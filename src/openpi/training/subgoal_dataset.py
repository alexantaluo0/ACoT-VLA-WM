"""Training-time subgoal (future / terminal / world-model top-head) image sampling."""

from __future__ import annotations

import copy
import dataclasses
import enum
import logging
import os
from pathlib import Path
from typing import SupportsIndex

import numpy as np
from PIL import Image

from openpi.training.sampler import get_base_dataset

logger = logging.getLogger(__name__)

_DEBUG_SUBGOAL_MAX_LOGS = 50
_debug_subgoal_log_count = 0


def _log_subgoal_sample(
    *,
    index: SupportsIndex,
    episode_index: int,
    frame_index: int,
    subtask_end: int,
    plan: "SubgoalSamplePlan",
    wm_store: "WorldModelSubgoalStore",
) -> None:
    global _debug_subgoal_log_count
    if _debug_subgoal_log_count >= _DEBUG_SUBGOAL_MAX_LOGS:
        return
    _debug_subgoal_log_count += 1
    if plan.kind == SubgoalSourceKind.FUTURE_FRAME:
        detail = f"FUTURE frame={plan.frame_index} (from current={frame_index}, subtask_end={subtask_end})"
    elif plan.kind == SubgoalSourceKind.TERMINAL_WM:
        detail = f"TERMINAL_WM step{plan.wm_step}.png (ep {episode_index:06d})"
    else:
        wm_exists = wm_store.has(episode_index, subtask_end)
        detail = (
            f"TERMINAL_LAST frame={plan.frame_index} (subtask_end={subtask_end}, "
            f"wm_step{subtask_end}.png exists={wm_exists})"
        )
    print(
        f"[subgoal { _debug_subgoal_log_count}/{_DEBUG_SUBGOAL_MAX_LOGS}] "
        f"idx={index} ep={episode_index} cur_frame={frame_index} -> {detail}",
        flush=True,
    )


def find_instruction_segment(segments: list[dict], frame_index: int) -> tuple[int, dict]:
    """Return (segment_id, segment) for the subtask that contains ``frame_index``."""
    segments = copy.deepcopy(segments)
    segments[0]["start_frame_index"] = 0
    segment_id = len(segments) - 1
    for i, segment in enumerate(segments):
        if frame_index >= segment["start_frame_index"] and frame_index <= segment["end_frame_index"]:
            segment_id = i
            break
    return segment_id, segments[segment_id]


def episode_last_frame_index(episode_start: int, episode_end_exclusive: int) -> int:
    """Last valid frame_index inside an episode (LeRobot ``to`` index is exclusive)."""
    return episode_end_exclusive - episode_start - 1


def resolve_wm_subgoal_root(repo_root: Path, explicit_root: str | None = None) -> Path | None:
    """Resolve world-model subgoal image root under a LeRobot dataset."""
    if explicit_root is not None:
        path = Path(explicit_root)
        return path if path.is_dir() else None

    images_root = repo_root / "images"
    if not images_root.is_dir():
        return None

    # Prefer chunk-000; extend if multi-chunk layout is needed later.
    for chunk in sorted(images_root.glob("chunk-*")):
        candidate = chunk / "observation.images.top_head"
        if candidate.is_dir():
            return candidate
    return None


class SubgoalSourceKind(enum.Enum):
    FUTURE_FRAME = "future_frame"
    TERMINAL_LAST_FRAME = "terminal_last_frame"
    TERMINAL_WM = "terminal_wm"


@dataclasses.dataclass(frozen=True)
class SubgoalSamplePlan:
    kind: SubgoalSourceKind
    frame_index: int
    wm_step: int | None = None


def plan_subgoal_sample(
    segment: dict,
    frame_index: int,
    *,
    fps: float,
    terminal_prob: float = 0.25,
    wm_in_terminal_prob: float = 0.5,
    future_horizon_s: float = 4.0,
    max_frame_index: int | None = None,
    wm_store: "WorldModelSubgoalStore | None" = None,
    episode_index: int = 0,
    rng: np.random.Generator | None = None,
) -> SubgoalSamplePlan:
    """Plan which subgoal image to load.

    - ``1 - terminal_prob`` (default 75%): uniform future frame in [t, t+4s] within the subtask.
    - ``terminal_prob`` (default 25%): subtask terminal subgoal, split by ``wm_in_terminal_prob``:
      - world-model ``step{subtask_end}.png`` when available;
      - otherwise dataset last frame of the subtask (also used when WM was chosen but missing).
    """
    rng = rng or np.random.default_rng()
    subtask_end = int(segment["success_frame_index"])
    if max_frame_index is not None:
        subtask_end = min(subtask_end, max_frame_index)

    if rng.random() < terminal_prob:
        use_wm = rng.random() < wm_in_terminal_prob
        if use_wm and wm_store is not None and wm_store.has(episode_index, subtask_end):
            return SubgoalSamplePlan(SubgoalSourceKind.TERMINAL_WM, subtask_end, wm_step=subtask_end)
        return SubgoalSamplePlan(SubgoalSourceKind.TERMINAL_LAST_FRAME, subtask_end)

    max_offset = max(0, int(future_horizon_s * fps))
    max_frame = min(frame_index + max_offset, subtask_end)
    if max_frame <= frame_index:
        # No future frame in subtask; fall back to terminal last frame.
        return SubgoalSamplePlan(SubgoalSourceKind.TERMINAL_LAST_FRAME, subtask_end)
    target = int(rng.integers(frame_index, max_frame + 1))
    return SubgoalSamplePlan(SubgoalSourceKind.FUTURE_FRAME, target)


class WorldModelSubgoalStore:
    """Lazy index of world-model subgoal PNGs: ``{root}/episode_XXXXXX/step{N}.png``."""

    def __init__(self, root: Path | None) -> None:
        self.root = Path(root) if root is not None else None
        self._steps_by_episode: dict[int, set[int]] = {}

    def __bool__(self) -> bool:
        return self.root is not None and self.root.is_dir()

    def has(self, episode_index: int, step: int) -> bool:
        if not self:
            return False
        if episode_index not in self._steps_by_episode:
            ep_dir = self.root / f"episode_{episode_index:06d}"
            if not ep_dir.is_dir():
                self._steps_by_episode[episode_index] = set()
            else:
                steps: set[int] = set()
                for path in ep_dir.glob("step*.png"):
                    try:
                        steps.add(int(path.stem[4:]))
                    except ValueError:
                        continue
                self._steps_by_episode[episode_index] = steps
        return step in self._steps_by_episode[episode_index]

    def load(self, episode_index: int, step: int):
        """Load image in LeRobot-like CHW float32 [0, 1] format."""
        if not self:
            raise FileNotFoundError("World-model subgoal root is not configured.")
        path = self.root / f"episode_{episode_index:06d}" / f"step{step}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        import torch

        img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return tensor


@dataclasses.dataclass
class SubgoalFrameDataset:
    """Wrap a LeRobot dataset and attach a sampled top-head subgoal image."""

    _dataset: object
    instruction_segments: dict
    fps: float
    subgoal_image_key: str = "observation.images.top_head"
    output_key: str = "subgoal_top_head"
    # Fraction using terminal subgoal (default 25%); the rest sample future frames in [t, t+4s].
    terminal_prob: float = 0.25
    # Within the terminal branch only: P(WM step image) vs P(dataset subtask last frame).
    wm_in_terminal_prob: float = 0.5
    future_horizon_s: float = 4.0
    wm_subgoal_root: str | None = None
    # Back-compat alias for ``terminal_prob``.
    end_prob: float | None = None

    def __post_init__(self) -> None:
        if self.end_prob is not None:
            object.__setattr__(self, "terminal_prob", self.end_prob)
        self._base = get_base_dataset(self._dataset)
        if not hasattr(self._base, "episode_data_index"):
            raise ValueError("SubgoalFrameDataset requires a LeRobotDataset with episode_data_index.")

        repo_root = Path(self._base.root) if hasattr(self._base, "root") else None
        wm_root = resolve_wm_subgoal_root(repo_root, self.wm_subgoal_root) if repo_root else None
        object.__setattr__(self, "_wm_store", WorldModelSubgoalStore(wm_root))
        if self._wm_store:
            logger.info("World-model subgoal images: %s", wm_root)
        else:
            logger.info(
                "World-model subgoal root not found; terminal branch will use dataset last frame only."
            )

    def __len__(self) -> int:
        return len(self._dataset)

    def _load_dataset_frame(self, episode_index: int, target_frame: int):
        episode_start = int(self._base.episode_data_index["from"][episode_index].item())
        episode_end = int(self._base.episode_data_index["to"][episode_index].item())
        last_frame = episode_last_frame_index(episode_start, episode_end)
        target_frame = min(max(target_frame, 0), last_frame)
        global_index = episode_start + target_frame
        return self._base[global_index][self.subgoal_image_key]

    def _resolve_terminal_image(self, episode_index: int, plan: SubgoalSamplePlan):
        """Load terminal subgoal; fall back to dataset last frame if WM image is missing."""
        subtask_end = plan.frame_index
        if plan.kind == SubgoalSourceKind.TERMINAL_WM and plan.wm_step is not None:
            try:
                if self._wm_store.has(episode_index, plan.wm_step):
                    return self._wm_store.load(episode_index, plan.wm_step)
            except FileNotFoundError:
                pass
        return self._load_dataset_frame(episode_index, subtask_end)

    def __getitem__(self, index: SupportsIndex):
        item = self._dataset[index]
        episode_index = int(item["episode_index"])
        frame_index = int(item["frame_index"])

        segments = self.instruction_segments.get(str(episode_index))
        if not segments:
            raise ValueError(f"No instruction_segments for episode {episode_index}")

        episode_start = int(self._base.episode_data_index["from"][episode_index].item())
        episode_end = int(self._base.episode_data_index["to"][episode_index].item())
        last_frame = episode_last_frame_index(episode_start, episode_end)

        _, segment = find_instruction_segment(segments, frame_index)
        subtask_end = min(int(segment["success_frame_index"]), last_frame)
        plan = plan_subgoal_sample(
            segment,
            frame_index,
            fps=self.fps,
            terminal_prob=self.terminal_prob,
            wm_in_terminal_prob=self.wm_in_terminal_prob,
            future_horizon_s=self.future_horizon_s,
            max_frame_index=last_frame,
            wm_store=self._wm_store,
            episode_index=episode_index,
        )

        if os.getenv("DEBUG_SUBGOAL_SAMPLING", "").lower() in ("1", "true", "yes"):
            _log_subgoal_sample(
                index=index,
                episode_index=episode_index,
                frame_index=frame_index,
                subtask_end=subtask_end,
                plan=plan,
                wm_store=self._wm_store,
            )

        if plan.kind == SubgoalSourceKind.FUTURE_FRAME:
            item[self.output_key] = self._load_dataset_frame(episode_index, plan.frame_index)
        else:
            item[self.output_key] = self._resolve_terminal_image(episode_index, plan)

        return item
