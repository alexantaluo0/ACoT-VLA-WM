"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.

LeRobot datasets store cameras as video; ``norm_stats.json`` only needs ``state`` /
``actions`` / ``coarse_actions``. By default this script skips MP4 decoding and feeds
black frames through the same transform chain so statistics stay correct for
proprioception and actions while running much faster.
"""

from __future__ import annotations

import contextlib
import pathlib

import numpy as np
import torch
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

_ORIG_QUERY_VIDEOS = LeRobotDataset._query_videos


def _query_videos_norm_stats_stub(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
    """Return zeros with the same layout as ``decode_video_frames`` (float32, CHW, value in [0, 1])."""
    out: dict[str, torch.Tensor] = {}
    for vid_key, query_ts in query_timestamps.items():
        n = len(query_ts)
        ft = self.meta.features.get(vid_key, {})
        shape = ft.get("shape", (3, 224, 224))
        shape_t = tuple(int(x) for x in shape)
        if len(shape_t) != 3:
            shape_t = (3, 224, 224)
        out[vid_key] = torch.zeros((n,) + shape_t, dtype=torch.float32)
    return out


@contextlib.contextmanager
def skip_video_decode_for_norm_stats(*, enabled: bool):
    """When enabled, patch LeRobotDataset to avoid ``decode_video_frames`` (major cost for norm_stats)."""
    if not enabled:
        yield
        return
    LeRobotDataset._query_videos = _query_videos_norm_stats_stub  # type: ignore[method-assign]
    try:
        yield
    finally:
        LeRobotDataset._query_videos = _ORIG_QUERY_VIDEOS  # type: ignore[method-assign]


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    max_frames: int | None = None,
    *,
    num_workers: int = 0,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # No ResizeImages: norm_stats only use state / actions / coarse_actions; skipping saves a lot of CPU.
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    # dataset = _data_loader.SafeDataset(dataset)
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False

    # num_workers must be 0 when skipping video decode: worker processes re-import LeRobotDataset and would
    # not see the main-process monkey-patch on _query_videos.
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def _default_robot_action_dim(config: _config.TrainConfig) -> int | None:
    """Use ``robot_action_dim`` from Go2-style data config when set (e.g. 24 for place_block_into_box)."""
    return getattr(config.data, "robot_action_dim", None)


def _default_output_dir(config: _config.TrainConfig, data_config: _config.DataConfig) -> pathlib.Path:
    aid = data_config.asset_id
    if isinstance(aid, list):
        aid = aid[0] if aid else None
    if aid:
        path = pathlib.Path(str(aid))
        if path.is_absolute():
            return path
        return (pathlib.Path(config.assets_base_dir) / config.name / path).resolve()
    return config.assets_dirs


def main(
    config_name: str,
    max_frames: int | None = None,
    output_dir: pathlib.Path | None = None,
    robot_action_dim: int | None = None,
    decode_video: bool = False,
) -> None:
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    # LeRobot: skip MP4 decode unless user explicitly needs real frames (slow; irrelevant to norm_stats).
    skip_video = data_config.rlds_data_dir is None and not decode_video
    with skip_video_decode_for_norm_stats(enabled=skip_video):
        if data_config.rlds_data_dir is not None:
            data_loader, num_batches = create_rlds_dataloader(
                data_config, config.model.action_horizon, config.batch_size, max_frames
            )
        else:
            workers = 0 if skip_video else 8
            data_loader, num_batches = create_torch_dataloader(
                data_config,
                config.batch_size,
                config.model,
                max_frames,
                num_workers=workers,
            )

        keys = ["state", "actions", "coarse_actions"]
        stats = {key: normalize.RunningStats() for key in keys}

        sample_ratio = 0.1
        max_batches = int(num_batches * sample_ratio)

        data_iter = iter(data_loader)
        pbar = tqdm.tqdm(total=max_batches, desc="Computing stats")
        valid_batches = 0
        while valid_batches < max_batches:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            except Exception as e:
                print(f"\n[Warning] Skipped a bad batch due to error: {e}")
                continue

            for key in keys:
                if key not in batch:
                    continue
                values = np.asarray(batch[key])
                stats[key].update(values.reshape(-1, values.shape[-1]))

            pbar.update(1)
            valid_batches += 1

        pbar.close()

        if valid_batches == 0:
            raise RuntimeError(
                "No batches were successfully loaded (all failed or dataset empty). "
                "If you see video timestamp tolerance errors, training uses DataConfig.video_tolerance_s "
                "(now passed into LeRobotDataset). When skipping video decode, num_workers must be 0 "
                "so the stub patch applies in the same process; use --decode-video only if you need real frames."
            )

    norm_stats = {key: rs.get_statistics() for key, rs in stats.items()}

    model_dim = config.model.action_dim
    rad = robot_action_dim if robot_action_dim is not None else _default_robot_action_dim(config)
    norm_stats = normalize.align_norm_stats_to_model_dim(
        norm_stats,
        model_action_dim=model_dim,
        robot_action_dim=rad,
    )

    out = output_dir if output_dir is not None else _default_output_dir(config, data_config)
    out = pathlib.Path(out)
    print(f"Writing stats ({model_dim} dims, robot_action_dim={rad!r}) to: {out}")
    normalize.save(out, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
