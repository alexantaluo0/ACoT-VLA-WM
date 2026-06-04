from collections.abc import Iterator, Sequence
import contextlib
import multiprocessing
import os
import pathlib
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)

_LEROBOT_VIDEO_KEY_FILTERS: dict[str, frozenset[str] | None] = {}
_LEROBOT_VIDEO_KEY_FILTER_PATCHED = False
_LEROBOT_DATASET_VIDEO_PATCHED = False


def _all_lerobot_video_keys(meta: lerobot_dataset.LeRobotDatasetMetadata) -> list[str]:
    return [key for key, ft in meta.features.items() if ft["dtype"] == "video"]


def resolve_lerobot_video_keys_for_data_config(
    meta: lerobot_dataset.LeRobotDatasetMetadata,
    data_config: _config.DataConfig,
) -> list[str] | None:
    """Return the subset of LeRobot video keys required by ``data_config``, or ``None`` to keep all."""
    filt = lerobot_observation_keys_from_data_config(data_config)
    if filt is None:
        return None
    return [key for key in _all_lerobot_video_keys(meta) if key in filt]


def _openpi_active_video_keys(dataset: lerobot_dataset.LeRobotDataset) -> list[str]:
    custom = getattr(dataset, "_openpi_video_keys", None)
    if custom is not None:
        return custom
    return dataset.meta.video_keys


def _install_lerobot_dataset_video_key_patch() -> None:
    global _LEROBOT_DATASET_VIDEO_PATCHED
    if _LEROBOT_DATASET_VIDEO_PATCHED:
        return
    cls = lerobot_dataset.LeRobotDataset

    def _get_query_timestamps(
        self: lerobot_dataset.LeRobotDataset,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps: dict[str, list[float]] = {}
        for key in _openpi_active_video_keys(self):
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        return query_timestamps

    def _query_hf_dataset(
        self: lerobot_dataset.LeRobotDataset,
        query_indices: dict[str, list[int]],
    ) -> dict:
        video_keys = set(_openpi_active_video_keys(self))
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in video_keys
        }

    cls._get_query_timestamps = _get_query_timestamps
    cls._query_hf_dataset = _query_hf_dataset
    _LEROBOT_DATASET_VIDEO_PATCHED = True


def _attach_openpi_video_keys(dataset: "Dataset", video_keys: list[str] | None) -> None:
    if video_keys is None:
        return
    while isinstance(dataset, TransformedDataset):
        dataset = dataset._dataset
    if isinstance(dataset, lerobot_dataset.LeRobotDataset):
        _install_lerobot_dataset_video_key_patch()
        dataset._openpi_video_keys = video_keys


def resolve_lerobot_repo_root(repo_id: str) -> tuple[str, pathlib.Path]:
    """Resolve LeRobot ``repo_id`` and on-disk ``root`` (supports absolute local paths)."""
    path = pathlib.Path(repo_id)
    if path.is_absolute():
        return repo_id, path
    root = pathlib.Path(lerobot_dataset.HF_LEROBOT_HOME) / repo_id
    return repo_id, root


def lerobot_observation_keys_from_data_config(data_config: _config.DataConfig) -> frozenset[str] | None:
    """LeRobot feature keys referenced by repack transforms (e.g. ``observation.images.top_head``)."""
    keys: set[str] = set()

    def _walk(value: object) -> None:
        if isinstance(value, str) and value.startswith("observation.images."):
            keys.add(value)
        elif isinstance(value, dict):
            for nested in value.values():
                _walk(nested)

    for transform in data_config.repack_transforms.inputs:
        if isinstance(transform, _transforms.RepackTransform):
            _walk(transform.structure)
    return frozenset(keys) if keys else None


def _install_lerobot_video_key_filter_patch() -> None:
    global _LEROBOT_VIDEO_KEY_FILTER_PATCHED
    if _LEROBOT_VIDEO_KEY_FILTER_PATCHED:
        return
    meta_cls = lerobot_dataset.LeRobotDatasetMetadata
    orig_get = meta_cls.video_keys.fget

    def _filtered_video_keys(self: lerobot_dataset.LeRobotDatasetMetadata) -> list[str]:
        keys = orig_get(self)
        filt = _LEROBOT_VIDEO_KEY_FILTERS.get(str(pathlib.Path(self.root).resolve()))
        if filt is None:
            return keys
        return [key for key in keys if key in filt]

    meta_cls.video_keys = property(_filtered_video_keys)
    _LEROBOT_VIDEO_KEY_FILTER_PATCHED = True


@contextlib.contextmanager
def lerobot_video_key_filter(root: pathlib.Path, keys: frozenset[str] | None):
    """Temporarily restrict which LeRobot ``video_keys`` are required/decoded for ``root``."""
    _install_lerobot_video_key_filter_patch()
    root_key = str(root.resolve())
    previous = _LEROBOT_VIDEO_KEY_FILTERS.get(root_key)
    _LEROBOT_VIDEO_KEY_FILTERS[root_key] = keys
    try:
        yield
    finally:
        if previous is None:
            _LEROBOT_VIDEO_KEY_FILTERS.pop(root_key, None)
        else:
            _LEROBOT_VIDEO_KEY_FILTERS[root_key] = previous


def register_lerobot_video_key_filter(root: pathlib.Path, keys: frozenset[str] | None) -> None:
    """Register a persistent LeRobot video-key filter for ``root`` (until process exit)."""
    _install_lerobot_video_key_filter_patch()
    root_key = str(root.resolve())
    if keys is None:
        _LEROBOT_VIDEO_KEY_FILTERS.pop(root_key, None)
    else:
        _LEROBOT_VIDEO_KEY_FILTERS[root_key] = keys


def _validate_local_lerobot_files(
    root: pathlib.Path,
    meta: lerobot_dataset.LeRobotDatasetMetadata,
    *,
    video_keys: Sequence[str],
) -> None:
    missing: list[str] = []
    for ep_idx in range(meta.total_episodes):
        parquet = root / meta.get_data_file_path(ep_idx)
        if not parquet.is_file():
            missing.append(str(parquet))
        for vid_key in video_keys:
            video = root / meta.get_video_file_path(ep_idx, vid_key)
            if not video.is_file():
                missing.append(str(video))
    if not missing:
        return
    preview = "\n".join(missing[:25])
    extra = len(missing) - min(len(missing), 25)
    suffix = f"\n... and {extra} more missing files" if extra > 0 else ""
    raise RuntimeError(
        f"Local LeRobot dataset at {root} is incomplete ({len(missing)} missing files). "
        f"Restore the missing parquet/video files or restrict training to cameras declared in "
        f"repack_transforms (unused depth videos can be omitted).\n"
        f"Missing examples:\n{preview}{suffix}"
    )


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class SafeDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: SupportsIndex):
        try:
            return self.dataset[index]
        except Exception as e:
            print(f"[Data Load Error] Skipping index {index} due to: {e}")
            return None
    
    def __getattr__(self, name):
        if name == 'dataset':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute 'dataset'")
        
        return getattr(self.dataset, name)


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        if not hasattr(self._dataset, "_datasets"):
            item = self._dataset[index]
            return self._transform(item)
        else:
            idx = index.__index__()
            for d in self._dataset._datasets:
                if idx < len(d):
                    item = d[idx]
                    return self._transform(item)
                idx -= len(d)
            raise IndexError("Index out of range")

    def __len__(self) -> int:
        if not hasattr(self._dataset, "_datasets"):
            length = len(self._dataset)
        else:
            length = 0
            for item in self._dataset._datasets:
                length += len(item)
        return length


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


def _validate_lerobot_observation_metadata(
    repo_id: str,
    mode: _config.LeRobotObservationMode,
    *,
    root: pathlib.Path | None = None,
    video_key_filter: frozenset[str] | None = None,
) -> None:
    with lerobot_video_key_filter(root or resolve_lerobot_repo_root(repo_id)[1], video_key_filter):
        meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
        has_video = len(meta.video_keys) > 0
        has_image = len(meta.image_keys) > 0
    if mode == _config.LeRobotObservationMode.video_mp4:
        if not has_video:
            raise ValueError(
                f"lerobot_observation_mode=video_mp4 but dataset {repo_id!r} has no LeRobot video features "
                f"(video_keys is empty). Point --data.repo-id at an MP4-backed dataset, or use "
                f"--data.lerobot-observation-mode parquet_images for parquet image columns."
            )
    elif mode == _config.LeRobotObservationMode.parquet_images:
        if has_video:
            raise ValueError(
                f"lerobot_observation_mode=parquet_images but dataset {repo_id!r} still declares video features "
                f"{meta.video_keys!r}. Use a predecoded dataset root, or set --data.predecoded-repo-id to that root."
            )
        if not has_image:
            raise ValueError(
                f"lerobot_observation_mode=parquet_images but dataset {repo_id!r} has no LeRobot image features."
            )
    else:
        raise AssertionError(mode)


def _create_single_lerobot_dataset(
    repo_id: str,
    data_config: _config.DataConfig,
    action_chunk_size: int,
) -> Dataset:
    repo_root = resolve_lerobot_repo_root(repo_id)[1]
    root_key = str(repo_root.resolve())
    if root_key not in _LEROBOT_VIDEO_KEY_FILTERS:
        register_lerobot_video_key_filter(repo_root, lerobot_observation_keys_from_data_config(data_config))
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=repo_root)
    if pathlib.Path(repo_id).is_absolute() or repo_root.is_dir():
        _validate_local_lerobot_files(repo_root, dataset_meta, video_keys=dataset_meta.video_keys)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=repo_root,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_chunk_size)]
            for key in data_config.action_sequence_keys
        },
        tolerance_s=float(data_config.video_tolerance_s),
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])
    if data_config.prompt_from_hl_instruction:
        dataset = TransformedDataset(
            dataset,
            [_transforms.PromptFromHighlevelInstruction(dataset_meta.info["instruction_segments"])],
        )
    if data_config.enable_subgoal_training:
        from openpi.training.subgoal_dataset import SubgoalFrameDataset

        terminal_prob = (
            data_config.subgoal_end_prob
            if data_config.subgoal_end_prob is not None
            else data_config.subgoal_terminal_prob
        )
        dataset = SubgoalFrameDataset(
            _dataset=dataset,
            instruction_segments=dataset_meta.info["instruction_segments"],
            fps=float(dataset_meta.fps),
            subgoal_image_key=data_config.subgoal_image_key,
            output_key=data_config.subgoal_output_key,
            terminal_prob=terminal_prob,
            wm_in_terminal_prob=data_config.subgoal_wm_in_terminal_prob,
            future_horizon_s=data_config.subgoal_future_horizon_s,
            wm_subgoal_root=data_config.subgoal_wm_root,
        )
    active_video_keys = resolve_lerobot_video_keys_for_data_config(dataset_meta, data_config)
    if active_video_keys is not None:
        skipped = sorted(set(_all_lerobot_video_keys(dataset_meta)) - set(active_video_keys))
        if skipped:
            print(f"LeRobot dataset {repo_root}: skipping unused video keys {skipped}")
        _attach_openpi_video_keys(dataset, active_video_keys)
    return dataset


def create_torch_dataset(
    data_config: _config.DataConfig, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training (LeRobot). Observation policy is ``lerobot_observation_mode``."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    mode = data_config.lerobot_observation_mode
    video_key_filter = lerobot_observation_keys_from_data_config(data_config)
    if isinstance(repo_id, list):
        for r in repo_id:
            _, root = resolve_lerobot_repo_root(r)
            _validate_lerobot_observation_metadata(
                r, mode, root=root, video_key_filter=video_key_filter
            )
    else:
        _, root = resolve_lerobot_repo_root(repo_id)
        _validate_lerobot_observation_metadata(
            repo_id, mode, root=root, video_key_filter=video_key_filter
        )

    if model_config.model_type == _model.ModelType.ACOT_VLA_PI0 or model_config.model_type == _model.ModelType.ACOT_VLA_PI05:

        acot_action_horizons = jnp.array((model_config.coarse_action_horizon, model_config.action_horizon))
        joint_action_shifts = jnp.array((data_config.joint_action_shifts))
        action_chunk_size = max(acot_action_horizons * joint_action_shifts).item()

    else:
        action_chunk_size = model_config.action_horizon

    if isinstance(repo_id, list):
        # If repo_id is a list, create a dataset for each repo_id and concatenate them.
        dataset_metas = [
            lerobot_dataset.LeRobotDatasetMetadata(r) for r in repo_id
        ]
        tol = float(data_config.video_tolerance_s)
        dataset = lerobot_dataset.MultiLeRobotDataset(
            repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(action_chunk_size)]
                for dataset_meta in dataset_metas
                for key in data_config.action_sequence_keys
            },
            tolerances_s={r: tol for r in repo_id},
        )
        if data_config.prompt_from_task:
            for n, d in enumerate(dataset._datasets):
                dataset._datasets[n] = TransformedDataset(
                    d, [_transforms.PromptFromLeRobotTask(dataset_metas[n].tasks)]
                )
        if data_config.prompt_from_hl_instruction:
            for n, d in enumerate(dataset._datasets):
                dataset._datasets[n] = TransformedDataset(
                    d,[_transforms.PromptFromHighlevelInstruction(dataset_metas[n].info['instruction_segments'])]
                )
        for i, d in enumerate(dataset._datasets):
            print(f"Dataset {i} has {len(d)} frames.")

    else:
        return _create_single_lerobot_dataset(repo_id, data_config, action_chunk_size)

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training."""
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    sampler = None
    if data_config.dataloader_sampler != '':
        from openpi.training.sampler import FrameSampler
        sampler = FrameSampler(dataset, data_config.dataloader_sampler)
        shuffle = False

    dataset = SafeDataset(dataset)
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        sampler = sampler
    )

    if model_config.model_type == _model.ModelType.ACOT_VLA_PI0 or model_config.model_type == _model.ModelType.ACOT_VLA_PI05:
        return DataLoaderACOTImpl(data_config, data_loader)
    else:
        return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        sampler = None,
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
            sampler=sampler,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                if batch is None:
                    continue
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    filter_items = [x for x in items if x is not None]
    if len(filter_items) != len(items):
        return None
    # return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *filter_items)

    def debug_stack(*args):
        arrays = [np.asarray(x) for x in args]
        try:
            return np.stack(arrays, axis=0)
        except ValueError as e:
            shapes = [x.shape for x in arrays]
            unique_shapes = set(shapes)
            print(f"\n======== DEBUG ERROR ========")
            print(f"Stacking failed!")
            print(f"Found varying shapes: {unique_shapes}")
            print(f"First 5 shapes: {shapes[:5]}")
            print(f"Sample data (first item): {arrays[0]}")
            print(f"=============================\n")
            raise e

    return jax.tree.map(debug_stack, *filter_items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    # Spawned DataLoader workers re-import modules; re-apply LeRobot patches here.
    _install_lerobot_dataset_video_key_patch()
    _install_lerobot_video_key_filter_patch()


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]

class DataLoaderACOTImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"], batch["coarse_actions"]


_install_lerobot_dataset_video_key_patch()
_install_lerobot_video_key_filter_patch()