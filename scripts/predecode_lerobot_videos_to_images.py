#!/usr/bin/env python3
"""将 LeRobot 视频数据集离线解码为 image 列（嵌入 parquet），训练时不再走 _query_videos。

用法示例::

    export HF_LEROBOT_HOME=/data/dataset/Robotdataset/Robotdataset/G2_Robot
    uv run python scripts/predecode_lerobot_videos_to_images.py \\
        --input-repo-id longqi_place_block_into_box_20sample \\
        --output-repo-id longqi_place_block_into_box_20sample_images

若转换中断导致个别 episode parquet 缺失，可用 ``--repair-missing`` 从同一套 **video** 输入仅补缺失文件（``--resize`` / ``--image-format`` / ``--jpeg-quality`` 须与首次转换一致）。

保留原有 parquet 表格列（含 action.0..、observation.state.0.. 展开列），仅新增与 meta 中同名的相机 image 列。
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import av
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset, Features, Image as HFImage, Sequence as HFSequence, Value
from PIL import Image


def _load_info(repo_root: Path) -> dict[str, Any]:
    with open(repo_root / "meta" / "info.json") as f:
        return json.load(f)


def _video_keys_from_info(info: dict[str, Any]) -> list[str]:
    return [k for k, ft in info["features"].items() if ft.get("dtype") == "video"]


def _build_image_info(info: dict[str, Any], resize: tuple[int, int] | None) -> dict[str, Any]:
    """复制 info，并将 video 特征改为 image，关闭 video_path / total_videos。"""
    out = json.loads(json.dumps(info))  # deep copy (JSON-safe)
    out["total_videos"] = 0
    out["video_path"] = None
    for key, ft in list(out["features"].items()):
        if ft.get("dtype") == "video":
            new_ft = {k: v for k, v in ft.items() if k != "info"}
            new_ft["dtype"] = "image"
            if resize is not None:
                width, height = resize
                new_ft["shape"] = [3, height, width]
            out["features"][key] = new_ft
    return out


def _encode_image_bytes(
    image: Image.Image,
    *,
    image_format: str,
    jpeg_quality: int,
) -> bytes:
    buf = io.BytesIO()
    save_kwargs: dict[str, Any] = {}
    if image_format == "jpeg":
        save_kwargs = {"quality": jpeg_quality, "optimize": False}
    image.save(buf, format=image_format.upper(), **save_kwargs)
    return buf.getvalue()


def _decode_encoded_frames(
    video_path: Path,
    *,
    resize: tuple[int, int] | None,
    image_format: str,
    jpeg_quality: int,
) -> list[dict[str, bytes | None]]:
    """顺序解码整段 mp4，并立即编码成 datasets.Image 可嵌入的 bytes。"""
    frames: list[dict[str, bytes | None]] = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            if resize is not None:
                image = image.resize(resize, Image.BICUBIC)
            frames.append(
                {
                    "bytes": _encode_image_bytes(image, image_format=image_format, jpeg_quality=jpeg_quality),
                    "path": None,
                }
            )
    finally:
        container.close()
    return frames


def _pa_dtype_to_value(dtype: pa.DataType) -> Any:
    if pa.types.is_int64(dtype):
        return Value("int64")
    if pa.types.is_int32(dtype):
        return Value("int32")
    if pa.types.is_float32(dtype):
        return Value("float32")
    if pa.types.is_float64(dtype):
        return Value("float64")
    if pa.types.is_boolean(dtype):
        return Value("bool")
    if pa.types.is_string(dtype):
        return Value("string")
    if pa.types.is_list(dtype) or pa.types.is_large_list(dtype):
        return HFSequence(_pa_dtype_to_value(dtype.value_type))
    if pa.types.is_fixed_size_list(dtype):
        return HFSequence(_pa_dtype_to_value(dtype.value_type), length=dtype.list_size)
    raise ValueError(f"Unsupported parquet column dtype for expanded table: {dtype}")


def _features_from_table(
    schema: pa.Schema,
    image_keys: list[str],
) -> Features:
    hf: dict[str, Any] = {}
    for name in schema.names:
        hf[name] = _pa_dtype_to_value(schema.field(name).type)
    for key in image_keys:
        hf[key] = HFImage()
    return Features(hf)


def _format_path(tmpl: str, episode_index: int, chunks_size: int) -> str:
    ep_chunk = episode_index // chunks_size
    return tmpl.format(episode_index=episode_index, episode_chunk=ep_chunk)


def _convert_episode(
    *,
    input_root: Path,
    output_root: Path,
    source_episode_index: int,
    output_episode_index: int,
    output_start_index: int,
    video_keys: list[str],
    data_path_fmt: str,
    video_path_fmt: str,
    chunks_size: int,
    resize: tuple[int, int] | None,
    image_format: str,
    jpeg_quality: int,
) -> None:
    rel_data = _format_path(data_path_fmt, source_episode_index, chunks_size)
    in_parquet = input_root / rel_data
    if not in_parquet.is_file():
        raise FileNotFoundError(in_parquet)

    table = pq.read_table(in_parquet)
    n = table.num_rows

    col_dict: dict[str, list[Any]] = {name: table.column(name).to_pylist() for name in table.column_names}
    if "episode_index" in col_dict:
        col_dict["episode_index"] = [output_episode_index] * n
    if "frame_index" in col_dict:
        col_dict["frame_index"] = list(range(n))
    if "index" in col_dict:
        col_dict["index"] = list(range(output_start_index, output_start_index + n))

    for vk in video_keys:
        rel = Path(
            video_path_fmt.format(
                video_key=vk,
                episode_index=source_episode_index,
                episode_chunk=source_episode_index // chunks_size,
            )
        )
        video_file = input_root / rel
        if not video_file.is_file():
            raise FileNotFoundError(video_file)
        image_frames = _decode_encoded_frames(
            video_file,
            resize=resize,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
        if len(image_frames) != n:
            raise ValueError(
                f"帧数与 parquet 行数不一致: key={vk} episode={source_episode_index} "
                f"frames={len(image_frames)} rows={n} ({video_file})"
            )
        col_dict[vk] = image_frames

    features = _features_from_table(table.schema, video_keys)
    ds = Dataset.from_dict(col_dict, features=features)

    out_rel = Path(_format_path(data_path_fmt, output_episode_index, chunks_size))
    out_parquet = output_root / out_rel
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out_parquet)


def _copy_meta_files(input_root: Path, output_root: Path) -> None:
    (output_root / "meta").mkdir(parents=True, exist_ok=True)
    for name in ("episodes.jsonl", "tasks.jsonl", "stats.json"):
        src = input_root / "meta" / name
        if src.is_file():
            shutil.copy2(src, output_root / "meta" / name)
    # 部分数据集使用 episodes_stats.jsonl；若存在则一并复制
    for name in ("episodes_stats.jsonl", "stats.jsonl"):
        src = input_root / "meta" / name
        if src.is_file():
            shutil.copy2(src, output_root / "meta" / name)


def _load_jsonlines(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonlines(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _ensure_episodes_stats(output_root: Path) -> None:
    """LeRobot v2.1 wants episodes_stats.jsonl; older converted data may only have stats.json."""
    episodes_stats = output_root / "meta" / "episodes_stats.jsonl"
    if episodes_stats.is_file():
        try:
            rows = _load_jsonlines(episodes_stats)
            if rows and all("episode_index" in row and "stats" in row for row in rows):
                return
        except Exception:
            pass

        # Some datasets accidentally rename aggregate stats.json to episodes_stats.jsonl.
        with open(episodes_stats) as f:
            stats = json.load(f)
    else:
        stats_path = output_root / "meta" / "stats.json"
        if not stats_path.is_file():
            return
        with open(stats_path) as f:
            stats = json.load(f)

    episodes_path = output_root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        return

    episodes = _load_jsonlines(episodes_path)

    with open(episodes_stats, "w") as f:
        for episode in episodes:
            f.write(json.dumps({"episode_index": episode["episode_index"], "stats": stats}, ensure_ascii=False))
            f.write("\n")


def _write_output_info(output_root: Path, info: dict[str, Any]) -> None:
    with open(output_root / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)
        f.write("\n")


def _subset_meta_for_episodes(
    output_root: Path,
    info: dict[str, Any],
    episode_indices: list[int],
    image_dataset: bool,
) -> list[dict[str, Any]]:
    """在仅转换部分 episode 时，重编号为连续 episode 并更新 meta。"""
    ep_path = output_root / "meta" / "episodes.jsonl"
    if not ep_path.is_file():
        return []
    all_eps = _load_jsonlines(ep_path)
    episodes_by_index = {int(e["episode_index"]): e for e in all_eps}
    missing = [i for i in episode_indices if i not in episodes_by_index]
    if missing:
        raise ValueError(f"episodes.jsonl 中找不到 episode_index: {sorted(missing)}")

    filtered: list[dict[str, Any]] = []
    source_to_output: dict[int, int] = {}
    for output_episode_index, source_episode_index in enumerate(episode_indices):
        row = dict(episodes_by_index[source_episode_index])
        row["episode_index"] = output_episode_index
        filtered.append(row)
        source_to_output[source_episode_index] = output_episode_index

    _write_jsonlines(ep_path, filtered)

    stats_path = output_root / "meta" / "episodes_stats.jsonl"
    if stats_path.is_file():
        try:
            stats_rows = _load_jsonlines(stats_path)
        except json.JSONDecodeError:
            # Some converted datasets store aggregate stats JSON under this name.
            # Leave it for _ensure_episodes_stats(), which rewrites per-episode rows.
            stats_rows = []
        stats_by_index = {int(row["episode_index"]): row for row in stats_rows if "episode_index" in row}
        if stats_by_index:
            rewritten_stats = []
            for source_episode_index in episode_indices:
                if source_episode_index not in stats_by_index:
                    continue
                row = dict(stats_by_index[source_episode_index])
                row["episode_index"] = source_to_output[source_episode_index]
                rewritten_stats.append(row)
            _write_jsonlines(stats_path, rewritten_stats)

    total_frames = sum(int(e["length"]) for e in filtered)
    n = len(filtered)
    info["total_episodes"] = n
    info["total_frames"] = total_frames
    if image_dataset:
        info["total_videos"] = 0
    info["splits"] = {"train": f"0:{n}"}
    chunks_size = int(info.get("chunks_size", 1000))
    info["total_chunks"] = (n + chunks_size - 1) // chunks_size if n else 0
    return filtered


def convert_dataset(
    *,
    input_root: Path,
    output_root: Path,
    resize: tuple[int, int] | None,
    image_format: str,
    jpeg_quality: int,
    overwrite: bool,
    episode_indices: list[int] | None,
    num_workers: int,
) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在: {output_root}（使用 --overwrite 覆盖）")
        shutil.rmtree(output_root)

    info_in = _load_info(input_root)
    video_keys = _video_keys_from_info(info_in)
    if not video_keys:
        raise ValueError(f"未在 meta/info.json 中发现 dtype=video 的特征: {input_root}")

    data_path = info_in["data_path"]
    video_path = info_in.get("video_path")
    if not video_path:
        raise ValueError("输入 info.json 缺少 video_path，无法定位 mp4。")

    output_root.mkdir(parents=True)
    info_out = _build_image_info(info_in, resize)
    chunks_size = int(info_in.get("chunks_size", 1000))
    total_eps = info_in["total_episodes"]
    if episode_indices is not None:
        ep_list = sorted(set(episode_indices))
        bad = [i for i in ep_list if i < 0 or i >= total_eps]
        if bad:
            raise ValueError(f"episode 索引超出范围 [0, {total_eps - 1}]: {bad}")
    else:
        ep_list = list(range(total_eps))

    _copy_meta_files(input_root, output_root)
    output_episodes = (
        _subset_meta_for_episodes(output_root, info_out, ep_list, image_dataset=True)
        if episode_indices is not None
        else []
    )
    _write_output_info(output_root, info_out)
    _ensure_episodes_stats(output_root)

    n_plan = len(ep_list)
    jobs: list[dict[str, Any]] = []
    output_start_index = 0
    for position, source_episode_index in enumerate(ep_list):
        output_episode_index = int(output_episodes[position]["episode_index"]) if output_episodes else position
        rel_data = _format_path(data_path, source_episode_index, chunks_size)
        num_rows = pq.read_metadata(input_root / rel_data).num_rows
        jobs.append(
            {
                "input_root": input_root,
                "output_root": output_root,
                "source_episode_index": source_episode_index,
                "output_episode_index": output_episode_index,
                "output_start_index": output_start_index,
                "video_keys": video_keys,
                "data_path_fmt": data_path,
                "video_path_fmt": video_path,
                "chunks_size": chunks_size,
                "resize": resize,
                "image_format": image_format,
                "jpeg_quality": jpeg_quality,
            }
        )
        output_start_index += num_rows

    if num_workers <= 1:
        for i, job in enumerate(jobs, start=1):
            print(
                f"Converting episode {job['source_episode_index']:06d} -> {job['output_episode_index']:06d} "
                f"({i}/{n_plan}) ...",
                flush=True,
            )
            _convert_episode(**job)
    else:
        print(f"Converting {n_plan} episodes with {num_workers} workers ...", flush=True)
        with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_job = {executor.submit(_convert_episode, **job): job for job in jobs}
            for i, future in enumerate(futures.as_completed(future_to_job), start=1):
                job = future_to_job[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed converting episode {job['source_episode_index']:06d} "
                        f"-> {job['output_episode_index']:06d}"
                    ) from exc
                print(
                    f"Converted episode {job['source_episode_index']:06d} -> {job['output_episode_index']:06d} "
                    f"({i}/{n_plan})",
                    flush=True,
                )

    print(f"Done. Image dataset written to: {output_root}", flush=True)


def _global_row_offset_for_episode(output_root: Path, episode_index: int) -> int:
    """输出数据集中该 episode 第一行在全局 index 列上应有的起始值（前面所有 episode 的 length 之和）。"""
    rows = _load_jsonlines(output_root / "meta" / "episodes.jsonl")
    total = 0
    for row in rows:
        ei = int(row["episode_index"])
        if ei < episode_index:
            total += int(row["length"])
    return total


def repair_missing_episodes(
    *,
    input_root: Path,
    output_root: Path,
    output_repo_id: str,
    resize: tuple[int, int] | None,
    image_format: str,
    jpeg_quality: int,
    num_workers: int,
) -> None:
    """在已有 image 数据集上补写缺失的 episode parquet（输入须仍为带 mp4 的 video 数据集）。"""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    if not output_root.is_dir():
        raise FileNotFoundError(f"输出数据集不存在: {output_root}")

    info_in = _load_info(input_root)
    video_keys = _video_keys_from_info(info_in)
    if not video_keys:
        raise ValueError(f"输入不是 video 数据集（无 dtype=video）: {input_root}")

    data_path = info_in["data_path"]
    video_path = info_in.get("video_path")
    if not video_path:
        raise ValueError("输入 info.json 缺少 video_path。")
    chunks_size = int(info_in.get("chunks_size", 1000))

    meta_out = LeRobotDatasetMetadata(output_repo_id, root=output_root)
    if meta_out.video_keys:
        raise ValueError(f"输出仍含 video_keys，不适合 repair: {meta_out.video_keys}")

    missing: list[int] = []
    for ep_idx in range(meta_out.total_episodes):
        rel = meta_out.get_data_file_path(ep_idx)
        if not (output_root / rel).is_file():
            missing.append(ep_idx)

    if not missing:
        print("repair: 无缺失 parquet，跳过。", flush=True)
        return

    print(f"repair: 将补写 {len(missing)} 个缺失 episode: {missing[:30]}{' ...' if len(missing) > 30 else ''}", flush=True)

    jobs: list[dict[str, Any]] = []
    for ep_idx in missing:
        jobs.append(
            {
                "input_root": input_root,
                "output_root": output_root,
                "source_episode_index": ep_idx,
                "output_episode_index": ep_idx,
                "output_start_index": _global_row_offset_for_episode(output_root, ep_idx),
                "video_keys": video_keys,
                "data_path_fmt": data_path,
                "video_path_fmt": video_path,
                "chunks_size": chunks_size,
                "resize": resize,
                "image_format": image_format,
                "jpeg_quality": jpeg_quality,
            }
        )

    if num_workers <= 1:
        for i, job in enumerate(jobs, start=1):
            print(f"repair episode {job['source_episode_index']:06d} ({i}/{len(jobs)}) ...", flush=True)
            _convert_episode(**job)
    else:
        with futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_job = {executor.submit(_convert_episode, **job): job for job in jobs}
            for i, future in enumerate(futures.as_completed(future_to_job), start=1):
                job = future_to_job[future]
                try:
                    future.result()
                except Exception as exc:
                    raise RuntimeError(f"repair 失败 episode {job['source_episode_index']:06d}") from exc
                print(f"repair episode {job['source_episode_index']:06d} ({i}/{len(jobs)})", flush=True)

    print(f"repair: 完成，已写入 {len(missing)} 个 parquet。", flush=True)


def validate_image_dataset(repo_root: Path, repo_id: str) -> None:
    """检查 meta 与 LeRobotDataset 行为：无 video_keys、可读取样本。"""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=repo_root)
    assert not meta.video_keys, f"期望 video_keys 为空，实际: {meta.video_keys}"
    assert len(meta.image_keys) > 0, "期望存在 image_keys"
    print("metadata OK:", "video_keys=", meta.video_keys, "image_keys=", meta.image_keys)

    # LeRobotDataset 在本地 parquet 不完整时会走 Hub 下载，本地-only 的 repo_id 会 404；先显式报错。
    missing_parquets: list[str] = []
    for ep_idx in range(meta.total_episodes):
        rel = meta.get_data_file_path(ep_idx)
        if not (repo_root / rel).is_file():
            missing_parquets.append(str(repo_root / rel))
    if missing_parquets:
        preview = missing_parquets[:25]
        extra = len(missing_parquets) - len(preview)
        lines = "\n".join(preview)
        suffix = f"\n... 另有 {extra} 个缺失文件" if extra > 0 else ""
        raise RuntimeError(
            f"本地 episode parquet 不完整：缺失 {len(missing_parquets)} / {meta.total_episodes} 个文件。\n"
            f"可对本脚本使用 --repair-missing --input-repo-id <原始video数据集目录名> 仅补缺失文件；"
            f"或全量重转（勿中断）并确认 HF_LEROBOT_HOME 指向正确。\n"
            f"缺失示例:\n{lines}{suffix}"
        )

    # 确保 __getitem__ 不触发视频解码：无 mp4 亦可工作
    ds = LeRobotDataset(repo_id, root=repo_root, tolerance_s=30.0, video_backend="pyav")
    sample = ds[0]
    for k in meta.image_keys:
        assert k in sample, f"样本缺少 {k}"
        t = sample[k]
        assert hasattr(t, "shape"), f"{k} 应为张量，得到 {type(t)}"
    print("sample keys (subset):", [k for k in sample.keys() if "observation" in k or k in ("task", "timestamp")])
    print("first image tensor shape:", {k: sample[k].shape for k in meta.image_keys})


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--hf-lerobot-home",
        type=Path,
        default=None,
        help="LeRobot 根目录（默认环境变量 HF_LEROBOT_HOME）",
    )
    p.add_argument(
        "--input-repo-id",
        type=str,
        default=None,
        help="输入（video）数据集目录名，位于 HF_LEROBOT_HOME 下；--validate-only 时可省略；--repair-missing 时必填",
    )
    p.add_argument("--output-repo-id", type=str, required=True, help="输出数据集目录名")
    p.add_argument("--overwrite", action="store_true", help="若输出目录已存在则删除后重写")
    p.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="可选：解码后 resize 到该宽高（默认保留原始分辨率）",
    )
    p.add_argument(
        "--image-format",
        choices=("jpeg", "png"),
        default="jpeg",
        help="嵌入 parquet 的图片编码格式。默认 jpeg；如需无损可用 png，但会慢很多、文件更大。",
    )
    p.add_argument("--jpeg-quality", type=int, default=90, help="--image-format=jpeg 时的质量参数")
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="仅校验已存在的 image 数据集（需同时传 --output-repo-id）",
    )
    p.add_argument(
        "--repair-missing",
        action="store_true",
        help="在已有 image 输出上补写缺失的 episode parquet（需 --input-repo-id 指向原始 video 数据集；resize/编码参数须与当初一致）",
    )
    p.add_argument(
        "--episodes",
        type=int,
        nargs="*",
        default=None,
        metavar="INDEX",
        help="只转换指定 episode_index（可多个，如 --episodes 0 1）。省略则转换全部。",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="并行转换 episode 的进程数。默认 1；视频较多时可设为 8 或 16。",
    )
    args = p.parse_args(argv)

    hf_home = args.hf_lerobot_home or Path(os.environ.get("HF_LEROBOT_HOME", ""))
    if not hf_home or not hf_home.is_dir():
        print("请设置有效的 HF_LEROBOT_HOME 或使用 --hf-lerobot-home", file=sys.stderr)
        return 1

    out_root = hf_home / args.output_repo_id
    resize = tuple(args.resize) if args.resize else None

    if args.validate_only and args.repair_missing:
        print("--validate-only 与 --repair-missing 不能同时使用", file=sys.stderr)
        return 1

    if args.validate_only:
        if not out_root.is_dir():
            print(f"输出数据集不存在: {out_root}", file=sys.stderr)
            return 1
        validate_image_dataset(out_root, args.output_repo_id)
        return 0

    if args.repair_missing:
        if not args.input_repo_id:
            print("--repair-missing 需要同时提供 --input-repo-id（原始 video 数据集）", file=sys.stderr)
            return 1
        if not out_root.is_dir():
            print(f"输出数据集不存在: {out_root}", file=sys.stderr)
            return 1
        in_root = hf_home / args.input_repo_id
        if not in_root.is_dir():
            print(f"输入数据集不存在: {in_root}", file=sys.stderr)
            return 1
        repair_missing_episodes(
            input_root=in_root,
            output_root=out_root,
            output_repo_id=args.output_repo_id,
            resize=resize,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            num_workers=args.num_workers,
        )
        validate_image_dataset(out_root, args.output_repo_id)
        return 0

    if not args.input_repo_id:
        print("转换时必须提供 --input-repo-id", file=sys.stderr)
        return 1

    in_root = hf_home / args.input_repo_id
    if not in_root.is_dir():
        print(f"输入数据集不存在: {in_root}", file=sys.stderr)
        return 1

    ep_arg = list(args.episodes) if args.episodes else None

    convert_dataset(
        input_root=in_root,
        output_root=out_root,
        resize=resize,
        image_format=args.image_format,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
        episode_indices=ep_arg,
        num_workers=args.num_workers,
    )
    validate_image_dataset(out_root, args.output_repo_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
