"""Debug logging for G2-style 24-dim state / action through the training data pipeline."""

from __future__ import annotations

import logging
import pathlib

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np

import openpi.models.model as _model
import openpi.models.acot_vla as _acot_vla
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
from openpi.models.pi0 import make_attn_mask

G2_JOINT_NAMES_24: tuple[str, ...] = (
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
    "gripper_l",
    "gripper_r",
    "head_joint1",
    "head_joint2",
    "head_joint3",
    "waist_joint1",
    "waist_joint2",
    "waist_joint3",
    "waist_joint4",
    "waist_joint5",
)


def setup_robot_io_log_file(log_path: pathlib.Path | str, *, mode: str = "w") -> logging.Logger:
    """Attach a dedicated file handler (file only; does not print to the terminal)."""
    path = pathlib.Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    abs_path = str(path.resolve())
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == abs_path:
            return logger
    file_handler = logging.FileHandler(path, mode=mode, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(file_handler)
    return logger


def _robot_action_dim(config: _config.TrainConfig) -> int:
    return int(getattr(config.data, "robot_action_dim", 24) or 24)


def _slice_acot_batch(
    observation: _model.Observation,
    actions: jax.Array,
    coarse_actions: jax.Array,
    sample_idx: int,
) -> tuple[_model.Observation, jax.Array, jax.Array]:
    """Use a single batch element for memory-heavy ``sample_actions`` / flow debug."""
    sl = slice(sample_idx, sample_idx + 1)
    obs_1 = _model.Observation(
        images={k: v[sl] for k, v in observation.images.items()},
        image_masks={k: v[sl] for k, v in observation.image_masks.items()},
        state=observation.state[sl],
        tokenized_prompt=None
        if observation.tokenized_prompt is None
        else observation.tokenized_prompt[sl],
        tokenized_prompt_mask=None
        if observation.tokenized_prompt_mask is None
        else observation.tokenized_prompt_mask[sl],
        token_ar_mask=None if observation.token_ar_mask is None else observation.token_ar_mask[sl],
        token_loss_mask=None
        if observation.token_loss_mask is None
        else observation.token_loss_mask[sl],
    )
    return obs_1, actions[sl], coarse_actions[sl]


def loss_to_float(loss: jax.Array) -> float:
    """Convert a replicated scalar loss array to Python float."""
    arr = np.asarray(jax.device_get(loss)).reshape(-1)
    if arr.size != 1:
        raise ValueError(f"Expected scalar loss, got shape {arr.shape}")
    return float(arr[0])


_loss_to_float = loss_to_float  # backward-compatible alias


def make_sharded_acot_compute_loss_fn(
    *,
    mesh: jax.sharding.Mesh,
    train_state_sharding: jax.sharding.NamedSharding,
    data_sharding: jax.sharding.NamedSharding,
    replicated_sharding: jax.sharding.NamedSharding,
):
    """Build ``compute_loss`` jitted with the same shardings as ``acot_train_step`` in ``scripts/train.py``."""

    def loss_fn(
        train_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
    ) -> jnp.ndarray:
        model = nnx.merge(train_state.model_def, train_state.params)
        model.train()
        return model.compute_loss(rng, observation, actions, coarse_actions, train=True)

    jitted = jax.jit(
        loss_fn,
        in_shardings=(
            train_state_sharding,
            replicated_sharding,
            data_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=replicated_sharding,
    )

    def evaluate(
        train_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
    ) -> jnp.ndarray:
        with sharding.set_mesh(mesh):
            return jitted(train_state, rng, observation, actions, coarse_actions)

    return evaluate


def make_sharded_acot_loss_action_fn(
    *,
    mesh: jax.sharding.Mesh,
    train_state_sharding: jax.sharding.NamedSharding,
    data_sharding: jax.sharding.NamedSharding,
    replicated_sharding: jax.sharding.NamedSharding,
    robot_action_dim: int = 24,
    num_sample_steps: int = 5,
):
    """Monitor-only MAE between ``sample_actions`` expert output and GT (no gradients).

    Evaluates mean |pred - gt| over batch, action horizon, and the first ``robot_action_dim`` dims
    in normalized action space. Shardings match ``acot_train_step``.
    """

    def metric_fn(
        train_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        gt_actions: _model.Actions,
    ) -> jnp.ndarray:
        model = nnx.merge(train_state.model_def, train_state.params)
        preds = model.sample_actions(rng, observation, num_steps=num_sample_steps)
        pred_actions = preds["actions"] if isinstance(preds, dict) else preds
        dim = min(robot_action_dim, int(pred_actions.shape[-1]), int(gt_actions.shape[-1]))
        return jnp.mean(jnp.abs(pred_actions[..., :dim] - gt_actions[..., :dim]))

    jitted = jax.jit(
        metric_fn,
        in_shardings=(
            train_state_sharding,
            replicated_sharding,
            data_sharding,
            data_sharding,
        ),
        out_shardings=replicated_sharding,
    )

    def evaluate(
        train_state: training_utils.TrainState,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        gt_actions: _model.Actions,
    ) -> jnp.ndarray:
        with sharding.set_mesh(mesh):
            return jitted(train_state, rng, observation, gt_actions)

    return evaluate


def _jit_compute_loss_train(model: nnx.Module):
    """Unsharded single-sample ``compute_loss`` for lightweight diagnostics only."""
    graphdef, state = nnx.split(model)

    def fun(
        st: nnx.State,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
    ):
        mod = nnx.merge(graphdef, st)
        return mod.compute_loss(rng, observation, actions, coarse_actions, train=True)

    jitted = jax.jit(fun)

    def wrapper(
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
    ):
        return jitted(state, rng, observation, actions, coarse_actions)

    return wrapper


def _vec24(arr, *, robot_action_dim: int = 24, time_idx: int = 0) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim > 1:
        a = a[time_idx]
    return a.reshape(-1)[:robot_action_dim]


def _log_vector_block(vec_1d: np.ndarray, *, label: str, robot_action_dim: int) -> None:
    logger = logging.getLogger(__name__)
    arr = _vec24(vec_1d, robot_action_dim=robot_action_dim)
    logger.info("-" * 57)
    logger.info("%s (%d dims):", label, arr.shape[-1])
    for line in _named_dim_lines(arr, G2_JOINT_NAMES_24):
        logger.info(line)
    for line in _summary_lines(arr, label=label):
        logger.info(line)


def log_robot_vector(vec, *, header: str | None = None, robot_action_dim: int = 24) -> None:
    """Log a 1D state or action vector with G2 joint names."""
    if vec is None:
        return
    logger = logging.getLogger(__name__)
    if header is not None:
        logger.info("%s", header)
    _log_vector_block(vec, label=header or "vector", robot_action_dim=robot_action_dim)


def _named_dim_lines(vec_1d, names: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    d = min(int(vec_1d.shape[-1]), len(names))
    for i in range(d):
        out.append(f'    "{names[i]}": {float(vec_1d[i]):.6f},')
    for i in range(d, int(vec_1d.shape[-1])):
        out.append(f'    "pad_{i - d}": {float(vec_1d[i]):.6f},')
    return out


def _summary_lines(vec_1d, *, label: str) -> list[str]:
    arr = np.asarray(vec_1d, dtype=np.float32).reshape(-1)
    abs_arr = np.abs(arr)
    finite_ratio = float(np.isfinite(arr).mean())
    gt_3_ratio = float((abs_arr > 3.0).mean())
    gt_5_ratio = float((abs_arr > 5.0).mean())

    lines = [
        (
            f"{label} stats: mean={float(arr.mean()):.6f}, std={float(arr.std()):.6f}, "
            f"min={float(arr.min()):.6f}, max={float(arr.max()):.6f}"
        ),
        f"{label} quality: finite_ratio={finite_ratio:.3f}, |x|>3 ratio={gt_3_ratio:.3f}, |x|>5 ratio={gt_5_ratio:.3f}",
    ]
    if finite_ratio < 1.0:
        lines.append(f"{label} warning: non-finite values detected.")
    if gt_5_ratio > 0.10:
        lines.append(f"{label} warning: too many large normalized values (|x|>5).")
    elif gt_3_ratio > 0.25:
        lines.append(f"{label} warning: many normalized values exceed |x|>3.")
    return lines


def fetch_pre_norm_sample(
    config: _config.TrainConfig,
    *,
    sample_index: int = 0,
) -> dict:
    """One sample after repack + data_transforms (slice/delta/mask), before Normalize."""
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.rlds_data_dir is not None:
        raise ValueError("robot_io_debug only supports LeRobot torch datasets.")
    dataset = _data_loader.create_torch_dataset(data_config, config.model)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
        ],
    )
    return dataset[sample_index]


def log_pre_norm_robot_sample(
    sample: dict,
    *,
    config_name: str,
    robot_action_dim: int = 24,
    sample_index: int = 0,
) -> None:
    """Log 24-dim state / actions before quantile normalization."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 57)
    logger.info(
        "PRE-NORM (repack + data_transforms, before Normalize) | config=%s | dataset_index=%d",
        config_name,
        sample_index,
    )
    _log_vector_block(
        sample["state"],
        label="pre_norm state",
        robot_action_dim=robot_action_dim,
    )
    if "actions" in sample:
        _log_vector_block(
            sample["actions"],
            label="pre_norm action chunk t=0",
            robot_action_dim=robot_action_dim,
        )
    if "coarse_actions" in sample:
        _log_vector_block(
            sample["coarse_actions"],
            label="pre_norm coarse_action chunk t=0",
            robot_action_dim=robot_action_dim,
        )


def print_normed_robot_batch(
    batch: tuple,
    *,
    config_name: str = "",
    sample_idx: int = 0,
    action_time_idx: int = 0,
    robot_action_dim: int = 24,
    stage_label: str = "MODEL INPUT (after Normalize + model transforms)",
) -> None:
    """Log state and action from the training batch (model input)."""
    if len(batch) == 3:
        obs, actions, coarse_actions = batch
    elif len(batch) == 2:
        obs, actions = batch
        coarse_actions = None
    else:
        raise ValueError(f"Expected batch tuple of length 2 or 3, got {len(batch)}")

    st = jax.device_get(obs.state)
    act = jax.device_get(actions)
    state_vec = _vec24(st[sample_idx], robot_action_dim=robot_action_dim)
    action_vec = _vec24(act[sample_idx], robot_action_dim=robot_action_dim, time_idx=action_time_idx)

    logger = logging.getLogger(__name__)
    logger.info("=" * 57)
    logger.info(
        "%s | config=%s | batch sample_idx=%d action_t=%d",
        stage_label,
        config_name,
        sample_idx,
        action_time_idx,
    )
    _log_vector_block(state_vec, label="model_input state", robot_action_dim=robot_action_dim)
    _log_vector_block(action_vec, label="model_input action chunk t=0", robot_action_dim=robot_action_dim)

    if coarse_actions is not None:
        coarse = jax.device_get(coarse_actions)
        coarse_vec = _vec24(coarse[sample_idx], robot_action_dim=robot_action_dim, time_idx=action_time_idx)
        _log_vector_block(
            coarse_vec,
            label="model_input coarse_action chunk t=0",
            robot_action_dim=robot_action_dim,
        )

    if coarse_actions is not None:
        act_full = _vec24(act[sample_idx], robot_action_dim=robot_action_dim, time_idx=0)
        coarse_full = _vec24(coarse[sample_idx], robot_action_dim=robot_action_dim, time_idx=0)
        diff = np.abs(act_full - coarse_full)
        logger.info(
            "action vs coarse_action t=0: max_abs_diff=%.6f mean_abs_diff=%.6f",
            float(diff.max()),
            float(diff.mean()),
        )


def log_training_pipeline_debug(
    config: _config.TrainConfig,
    batch: tuple,
    *,
    log_path: pathlib.Path | str | None = None,
    pre_norm_index: int = 0,
    batch_sample_idx: int = 0,
) -> pathlib.Path:
    """Log pre-norm (24-d) and model-input tensors; write to ``robot_io_debug.log`` under checkpoint dir."""
    if log_path is None:
        log_path = config.checkpoint_dir / "robot_io_debug.log"
    log_path = pathlib.Path(log_path)
    setup_robot_io_log_file(log_path)

    robot_dim = _robot_action_dim(config)
    logger = logging.getLogger(__name__)
    logger.info("Robot I/O debug log: %s", log_path.resolve())
    data_config = config.data.create(config.assets_dirs, config.model)
    logger.info("config=%s robot_action_dim=%d batch_size=%d", config.name, robot_dim, config.batch_size)
    logger.info("use_quantile_norm=%s asset_id=%s", data_config.use_quantile_norm, data_config.asset_id)

    try:
        pre_sample = fetch_pre_norm_sample(config, sample_index=pre_norm_index)
        log_pre_norm_robot_sample(
            pre_sample,
            config_name=config.name,
            robot_action_dim=robot_dim,
            sample_index=pre_norm_index,
        )
    except Exception as e:
        logger.exception("Failed to fetch pre-norm sample at index %d: %s", pre_norm_index, e)

    print_normed_robot_batch(
        batch,
        config_name=config.name,
        sample_idx=batch_sample_idx,
        robot_action_dim=robot_dim,
    )
    return log_path


def _acot_flow_matching_tensors(
    model: _acot_vla.ACOT_VLA,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    coarse_actions: _model.CoarseActions,
    *,
    train: bool = False,
) -> dict[str, jax.Array]:
    """Mirror ``ACOT_VLA.compute_loss`` but return per-dim flow-matching errors (before batch mean)."""
    preprocess_rng, time_rng, coarse_action_noise_rng, expert_action_noise_rng = jax.random.split(rng, 4)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    batch_shape = actions.shape[:-2]
    coarse_action_noise = jax.random.normal(coarse_action_noise_rng, coarse_actions.shape)
    expert_action_noise = jax.random.normal(expert_action_noise_rng, actions.shape)

    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]

    x_ref_t = time_expanded * coarse_action_noise + (1.0 - time_expanded) * coarse_actions
    u_ref_t = coarse_action_noise - coarse_actions
    x_expert_t = time_expanded * expert_action_noise + (1.0 - time_expanded) * actions
    u_expert_t = expert_action_noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions_prefix = _acot_vla._positions_from_attn_mask(prefix_mask)
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None, None], mask=prefix_attn_mask, positions=positions_prefix)

    if model.adopt_explicit_action_reasoner:
        suffix_ref_action_tokens, suffix_ref_action_mask, suffix_ref_action_ar_mask, adarms_ref_action_cond = (
            model.embed_suffix(observation, x_ref_t, time, suf_type="reasoner")
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_ref_action_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ref_action_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = _acot_vla._positions_from_attn_mask(input_mask)
        (_, suffix_ref_action_out, _), _ = model.PaliGemma.llm(
            [prefix_tokens, suffix_ref_action_tokens, None],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_ref_action_cond, None],
        )
        v_ref_t = model.coarse_action_out_proj(suffix_ref_action_out[:, -model.coarse_action_horizon :])
    else:
        v_ref_t = None

    if model.adopt_implicit_action_reasoner:
        import einops

        K_all, V_all = kv_cache
        K_rearranged = einops.rearrange(K_all, "L B T 1 D -> B L T D")
        V_rearranged = einops.rearrange(V_all, "L B T 1 D -> B L T D")
        implicit_action_reason = model.implicit_action_reasoner(K_rearranged, V_rearranged)
    else:
        implicit_action_reason = None

    explicit_action_reason = coarse_actions if model.adopt_explicit_action_reasoner else None
    suffix_expert_tokens, suffix_expert_mask, suffix_expert_ar_mask, adarms_expert_cond = model.embed_suffix(
        observation,
        x_expert_t,
        time,
        explicit_action_reason=explicit_action_reason,
        implicit_action_reason=implicit_action_reason,
        suf_type="expert",
    )
    input_mask = jnp.concatenate([prefix_mask, suffix_expert_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_expert_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = _acot_vla._positions_from_attn_mask(input_mask)
    (_, _, suffix_expert_out), _ = model.PaliGemma.llm(
        [prefix_tokens, None, suffix_expert_tokens],
        mask=attn_mask,
        positions=positions,
        adarms_cond=[None, None, adarms_expert_cond],
    )
    v_expert_t = model.action_out_proj(suffix_expert_out[:, -model.action_horizon :])

    out: dict[str, jax.Array] = {
        "u_expert": u_expert_t,
        "v_expert": v_expert_t,
        "expert_sq": jnp.square(u_expert_t - v_expert_t),
    }
    if v_ref_t is not None:
        out["u_ref"] = u_ref_t
        out["v_ref"] = v_ref_t
        out["ref_sq"] = jnp.square(u_ref_t - v_ref_t)
    return out


def _log_pred_vs_gt(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    label: str,
    robot_action_dim: int,
) -> None:
    logger = logging.getLogger(__name__)
    pred24 = _vec24(pred, robot_action_dim=robot_action_dim)
    gt24 = _vec24(gt, robot_action_dim=robot_action_dim)
    diff = np.abs(pred24 - gt24)
    logger.info("-" * 57)
    logger.info("%s pred vs GT (t=0): max_abs_diff=%.6f mean_abs_diff=%.6f", label, diff.max(), diff.mean())
    _log_vector_block(pred24, label=f"{label} PRED", robot_action_dim=robot_action_dim)
    _log_vector_block(gt24, label=f"{label} GT", robot_action_dim=robot_action_dim)
    _log_top_dims(diff, label=f"{label} |pred-GT|")


def _dim_label(idx: int) -> str:
    if idx < len(G2_JOINT_NAMES_24):
        return G2_JOINT_NAMES_24[idx]
    return f"pad_{idx - len(G2_JOINT_NAMES_24)}"


def _log_top_dims(values: np.ndarray, *, label: str, top_k: int = 8, max_dim: int | None = None) -> None:
    logger = logging.getLogger(__name__)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    d = len(arr) if max_dim is None else min(len(arr), max_dim)
    order = np.argsort(-np.abs(arr[:d]))[:top_k]
    logger.info("%s top-%d dims by |value|:", label, top_k)
    for i in order:
        logger.info("  [%2d] %-14s = %+.6f", int(i), _dim_label(int(i)), float(arr[i]))


def _log_branch_loss_summary(
    sq: np.ndarray,
    *,
    label: str,
    robot_action_dim: int,
    model_action_dim: int,
) -> None:
    """Split flow-matching MSE into robot dims vs padding dims."""
    per_dim = sq.mean(axis=(0, 1))
    robot_part = per_dim[:robot_action_dim]
    pad_part = per_dim[robot_action_dim:model_action_dim]
    logger = logging.getLogger(__name__)
    logger.info(
        "%s branch: all_dims_mean=%.6f robot_%dd_mean=%.6f pad_%dd_mean=%.6f",
        label,
        float(per_dim.mean()),
        robot_action_dim,
        float(robot_part.mean()),
        int(pad_part.shape[0]),
        float(pad_part.mean()) if pad_part.size else 0.0,
    )


def _log_flow_sq_per_dim(sq: np.ndarray, *, label: str, robot_action_dim: int) -> None:
    """sq: (batch, horizon, dim) -> per-dim mean sq error."""
    per_dim = sq.mean(axis=(0, 1))
    logger = logging.getLogger(__name__)
    logger.info("-" * 57)
    logger.info(
        "%s flow-matching MSE per dim (mean over batch+time): total=%.6f max=%.6f",
        label,
        float(per_dim[:robot_action_dim].mean()),
        float(per_dim[:robot_action_dim].max()),
    )
    _log_top_dims(per_dim[:robot_action_dim], label=f"{label} MSE")


def _log_tensor_abs_max(tensors: dict[str, np.ndarray], *, label: str, robot_action_dim: int) -> None:
    logger = logging.getLogger(__name__)
    logger.info("-" * 57)
    logger.info("%s abs-max per dim (batch+time mean of |.|):", label)
    for key, arr in tensors.items():
        if arr is None:
            continue
        per_dim = np.abs(arr).mean(axis=(0, 1))[:robot_action_dim]
        argmax = int(per_dim.argmax())
        logger.info(
            "  %s: global_max=%.6f dim_max=%.6f at [%d] %s",
            key,
            float(np.abs(arr).max()),
            float(per_dim.max()),
            argmax,
            _dim_label(argmax),
        )


def log_model_forward_debug(
    config: _config.TrainConfig,
    train_state: training_utils.TrainState,
    batch: tuple,
    rng: jax.Array,
    *,
    sharded_compute_loss,
    log_path: pathlib.Path | str | None = None,
    sample_idx: int = 0,
    action_time_idx: int = 0,
    num_sample_steps: int = 10,
) -> None:
    """Log model ``sample_actions`` outputs and per-dim flow-matching errors (append to robot_io_debug.log)."""
    if log_path is None:
        log_path = config.checkpoint_dir / "robot_io_debug.log"
    setup_robot_io_log_file(log_path, mode="a")

    robot_dim = _robot_action_dim(config)
    logger = logging.getLogger(__name__)
    logger.info("=" * 57)
    logger.info("MODEL FORWARD DEBUG | config=%s | sample_idx=%d", config.name, sample_idx)

    if len(batch) != 3:
        logger.warning("Expected ACOT batch (obs, actions, coarse_actions); got len=%d", len(batch))
        return
    if sharded_compute_loss is None:
        logger.warning("sharded_compute_loss is required for ACOT full-batch loss logging")
        return

    observation, actions, coarse_actions = batch
    model = nnx.merge(train_state.model_def, train_state.params)
    obs_1, act_1, coarse_1 = _slice_acot_batch(observation, actions, coarse_actions, sample_idx)

    # --- scalar loss (FSDP shardings match acot_train_step) ---
    loss_rng, sample_rng, flow_rng = jax.random.split(rng, 3)
    loss_full = loss_to_float(
        sharded_compute_loss(train_state, loss_rng, observation, actions, coarse_actions)
    )
    loss_one = loss_to_float(_jit_compute_loss_train(model)(loss_rng, obs_1, act_1, coarse_1))
    logger.info("compute_loss full batch (train=True, training shardings): %.6f", loss_full)
    logger.info("compute_loss single sample[%d] (unsharded diagnostic): %.6f", sample_idx, loss_one)

    # --- flow matching per-dim (one random noise/time draw, single sample) ---
    if isinstance(model, _acot_vla.ACOT_VLA):
        flow = jax.device_get(_acot_flow_matching_tensors(model, flow_rng, obs_1, act_1, coarse_1, train=True))
        _log_flow_sq_per_dim(flow["expert_sq"], label="expert actions", robot_action_dim=robot_dim)
        _log_tensor_abs_max(
            {
                "u_expert (target vel)": flow["u_expert"],
                "v_expert (model vel)": flow["v_expert"],
            },
            label="expert flow tensors",
            robot_action_dim=robot_dim,
        )
        if "ref_sq" in flow:
            _log_flow_sq_per_dim(flow["ref_sq"], label="coarse_actions", robot_action_dim=robot_dim)
            _log_tensor_abs_max(
                {
                    "u_ref (target vel)": flow["u_ref"],
                    "v_ref (model vel)": flow["v_ref"],
                },
                label="coarse flow tensors",
                robot_action_dim=robot_dim,
            )
    else:
        logger.warning("Model is not ACOT_VLA; skip per-dim flow debug.")

    # --- sample_actions (inference-style denoised output, single sample to save memory) ---
    sample_fn = nnx_utils.module_jit(model.sample_actions)
    preds = jax.device_get(sample_fn(sample_rng, obs_1, num_steps=num_sample_steps))
    gt_act = jax.device_get(act_1)[0, action_time_idx]
    if isinstance(preds, dict):
        pred_act = preds["actions"][0, action_time_idx]
        _log_pred_vs_gt(pred_act, gt_act, label="sample_actions.actions", robot_action_dim=robot_dim)
        if "coarse_actions" in preds:
            pred_coarse = preds["coarse_actions"][0, action_time_idx]
            gt_coarse = jax.device_get(coarse_1)[0, action_time_idx]
            _log_pred_vs_gt(
                pred_coarse, gt_coarse, label="sample_actions.coarse_actions", robot_action_dim=robot_dim
            )
    else:
        pred_act = preds[0, action_time_idx]
        _log_pred_vs_gt(pred_act, gt_act, label="sample_actions", robot_action_dim=robot_dim)

    logger.info("=" * 57)
    logger.info("End robot I/O debug")


def log_step_action_debug(
    config: _config.TrainConfig,
    train_state: training_utils.TrainState,
    batch: tuple,
    rng: jax.Array,
    *,
    sharded_compute_loss,
    step: int,
    log_path: pathlib.Path | str | None = None,
    sharded_loss_action=None,
    loss_action: float | None = None,
    sample_idx: int = 0,
    action_time_idx: int = 0,
    num_sample_steps: int = 5,
) -> None:
    """Per log_interval: flow-matching per-dim loss + ``sample_actions`` pred vs GT."""
    if log_path is None:
        log_path = config.checkpoint_dir / "robot_io_debug.log"
    setup_robot_io_log_file(log_path, mode="a")

    robot_dim = _robot_action_dim(config)
    model_dim = int(getattr(config.model, "action_dim", 32) or 32)
    logger = logging.getLogger(__name__)
    logger.info("=" * 57)
    logger.info("TRAIN STEP %d ACTION DEBUG | config=%s | sample_idx=%d", step, config.name, sample_idx)

    if len(batch) != 3:
        logger.warning("Expected ACOT batch (obs, actions, coarse_actions); got len=%d", len(batch))
        return
    if sharded_compute_loss is None:
        logger.warning("sharded_compute_loss is required for ACOT full-batch loss logging")
        return

    observation, actions, coarse_actions = batch
    model = nnx.merge(train_state.model_def, train_state.params)
    obs_1, act_1, coarse_1 = _slice_acot_batch(observation, actions, coarse_actions, sample_idx)

    loss_rng, sample_rng, flow_rng = jax.random.split(rng, 3)
    loss_full = loss_to_float(
        sharded_compute_loss(train_state, loss_rng, observation, actions, coarse_actions)
    )
    logger.info("compute_loss full batch (train=True, training shardings): %.6f", loss_full)
    if loss_action is not None:
        logger.info(
            "loss_action (sample_actions vs GT, mean |pred-GT|, robot %dd, monitor-only): %.6f",
            robot_dim,
            loss_action,
        )
    elif sharded_loss_action is not None:
        action_metric_rng = jax.random.fold_in(rng, 1)
        loss_action_val = loss_to_float(
            sharded_loss_action(train_state, action_metric_rng, observation, actions)
        )
        logger.info(
            "loss_action (sample_actions vs GT, mean |pred-GT|, robot %dd, monitor-only): %.6f",
            robot_dim,
            loss_action_val,
        )

    if isinstance(model, _acot_vla.ACOT_VLA):
        flow = jax.device_get(_acot_flow_matching_tensors(model, flow_rng, obs_1, act_1, coarse_1, train=True))
        _log_branch_loss_summary(
            flow["expert_sq"], label="expert", robot_action_dim=robot_dim, model_action_dim=model_dim
        )
        _log_flow_sq_per_dim(flow["expert_sq"], label="expert actions", robot_action_dim=robot_dim)
        _log_top_dims(
            flow["expert_sq"].mean(axis=(0, 1)),
            label="expert actions MSE (incl pad)",
            max_dim=model_dim,
        )
        _log_tensor_abs_max(
            {
                "u_expert (target vel)": flow["u_expert"],
                "v_expert (model vel)": flow["v_expert"],
            },
            label="expert flow tensors",
            robot_action_dim=model_dim,
        )
        if "ref_sq" in flow:
            _log_branch_loss_summary(
                flow["ref_sq"], label="coarse", robot_action_dim=robot_dim, model_action_dim=model_dim
            )
            _log_flow_sq_per_dim(flow["ref_sq"], label="coarse_actions", robot_action_dim=robot_dim)
            _log_top_dims(
                flow["ref_sq"].mean(axis=(0, 1)),
                label="coarse_actions MSE (incl pad)",
                max_dim=model_dim,
            )
            _log_tensor_abs_max(
                {
                    "u_ref (target vel)": flow["u_ref"],
                    "v_ref (model vel)": flow["v_ref"],
                },
                label="coarse flow tensors",
                robot_action_dim=model_dim,
            )
    else:
        logger.warning("Model is not ACOT_VLA; skip per-dim flow debug.")

    sample_fn = nnx_utils.module_jit(model.sample_actions)
    preds = jax.device_get(sample_fn(sample_rng, obs_1, num_steps=num_sample_steps))
    gt_act = jax.device_get(act_1)[0, action_time_idx]
    if isinstance(preds, dict):
        pred_act = preds["actions"][0, action_time_idx]
        _log_pred_vs_gt(pred_act, gt_act, label="sample_actions.actions", robot_action_dim=robot_dim)
        if model_dim > robot_dim:
            pred_full = np.asarray(pred_act, dtype=np.float32).reshape(-1)
            gt_full = np.asarray(gt_act, dtype=np.float32).reshape(-1)
            pad_diff = np.abs(pred_full[robot_dim:model_dim] - gt_full[robot_dim:model_dim])
            logger.info(
                "sample_actions.actions pad dims [%d:%d): max_abs_diff=%.6f mean_abs_diff=%.6f",
                robot_dim,
                model_dim,
                float(pad_diff.max()) if pad_diff.size else 0.0,
                float(pad_diff.mean()) if pad_diff.size else 0.0,
            )
            _log_top_dims(pad_diff, label="sample_actions.actions pad |pred-GT|", max_dim=pad_diff.size)
        if "coarse_actions" in preds:
            pred_coarse = preds["coarse_actions"][0, action_time_idx]
            gt_coarse = jax.device_get(coarse_1)[0, action_time_idx]
            _log_pred_vs_gt(
                pred_coarse, gt_coarse, label="sample_actions.coarse_actions", robot_action_dim=robot_dim
            )
    else:
        pred_act = preds[0, action_time_idx]
        _log_pred_vs_gt(pred_act, gt_act, label="sample_actions", robot_action_dim=robot_dim)

    logger.info("=" * 57)
