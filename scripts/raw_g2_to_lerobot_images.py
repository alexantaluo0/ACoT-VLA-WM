#!/usr/bin/env python3
"""一步将 G2 原始录制转为 LeRobot images 数据集（训练直接读 parquet 内嵌图）。

流程:
  1. convert_to_challenge_format_real.py
     按 aligned_joints.h5 时间戳从 h265+txt 抽帧写 mp4，帧数与 parquet 一致
  2. scripts/predecode_lerobot_videos_to_images.py
     将 mp4 预解码为 parquet 的 image 列

用法示例::

    export HF_LEROBOT_HOME=/data/dataset/Robotdataset/Robotdataset/G2_Robot
    python scripts/raw_g2_to_lerobot_images.py \\
        --input-dir /data/dataset/.../G2_real_data/0603 \\
        --video-repo-id 0603_train_video \\
        --images-repo-id 0603_train_video_images \\
        --overwrite

仅重跑预解码（video 数据集已存在）::

    python scripts/raw_g2_to_lerobot_images.py \\
        --input-dir .../0603 \\
        --video-repo-id 0603_train_video \\
        --images-repo-id 0603_train_video_images \\
        --skip-convert
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONVERT_SCRIPT = PROJECT_ROOT / "scripts" / "convert_to_challenge_format_real.py"
DEFAULT_PREDECODE_SCRIPT = PROJECT_ROOT / "scripts" / "predecode_lerobot_videos_to_images.py"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}\n", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _resolve_python(explicit: str | None, fallback: Path) -> Path:
    if explicit:
        return Path(explicit)
    if fallback.is_file():
        return fallback
    return Path(sys.executable)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input-dir", type=Path, required=True, help="G2 原始任务目录（含 001/002/... episode）")
    p.add_argument(
        "--hf-lerobot-home",
        type=Path,
        default=None,
        help="LeRobot 根目录（默认环境变量 HF_LEROBOT_HOME）",
    )
    p.add_argument("--video-repo-id", type=str, required=True, help="输出 video 数据集目录名（位于 HF_LEROBOT_HOME 下）")
    p.add_argument("--images-repo-id", type=str, required=True, help="输出 images 数据集目录名")
    p.add_argument("--task-name", type=str, default=None, help="info.json 任务名（默认 input-dir 目录名）")
    p.add_argument("--instruction", type=str, default=None, help="info.json instruction_segments 默认文案")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-episodes", type=int, default=-1, help="-1 表示全部 episode")
    p.add_argument("--overwrite", action="store_true", help="预解码时若 images 目录已存在则删除重写")
    p.add_argument("--skip-convert", action="store_true", help="跳过步骤 1，仅执行预解码")
    p.add_argument("--skip-predecode", action="store_true", help="仅执行步骤 1，不写 images 数据集")
    p.add_argument(
        "--no-align-videos",
        action="store_true",
        help="传给 convert：整段拷贝 mp4，不做 h5 对齐（不推荐）",
    )
    p.add_argument(
        "--convert-python",
        type=str,
        default=None,
        help="运行 convert 的 Python（需 h5py、pandas、pyarrow、av；默认本项目 .venv）",
    )
    p.add_argument(
        "--predecode-python",
        type=str,
        default=None,
        help="运行 predecode 的 Python（默认本项目 .venv）",
    )
    p.add_argument(
        "--convert-script",
        type=Path,
        default=DEFAULT_CONVERT_SCRIPT,
        help="convert_to_challenge_format_real.py 路径",
    )
    p.add_argument(
        "--predecode-script",
        type=Path,
        default=DEFAULT_PREDECODE_SCRIPT,
        help="predecode_lerobot_videos_to_images.py 路径",
    )
    p.add_argument("--dry-run", action="store_true", help="仅打印将执行的命令")
    # predecode 常用参数
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="步骤2 预解码并行度；步骤1 convert 的 episode 并行（多 episode 时相机并行自动降为 1）",
    )
    p.add_argument(
        "--video-workers",
        type=int,
        default=3,
        help="步骤1 每 episode 内 RGB 相机并行（线程池 + ffmpeg，默认 3）",
    )
    p.add_argument("--resize", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=None)
    p.add_argument("--image-format", choices=("jpeg", "png"), default="jpeg")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument(
        "--strict-alignment",
        action="store_true",
        help="预解码时要求 parquet 行数与 mp4 帧数严格一致（对齐抽帧后应可通过）",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input dir not found: {input_dir}", file=sys.stderr)
        return 1

    hf_home = args.hf_lerobot_home or Path(os.environ.get("HF_LEROBOT_HOME", ""))
    if not hf_home or not hf_home.is_dir():
        print("请设置有效的 HF_LEROBOT_HOME 或使用 --hf-lerobot-home", file=sys.stderr)
        return 1
    hf_home = hf_home.resolve()

    task_name = args.task_name or input_dir.name
    video_root = hf_home / args.video_repo_id
    convert_python = _resolve_python(args.convert_python, DEFAULT_PYTHON)
    predecode_python = _resolve_python(args.predecode_python, DEFAULT_PYTHON)

    if not args.convert_script.is_file():
        print(f"ERROR: convert script not found: {args.convert_script}", file=sys.stderr)
        return 1
    if not args.skip_predecode and not args.predecode_script.is_file():
        print(f"ERROR: predecode script not found: {args.predecode_script}", file=sys.stderr)
        return 1

    print("=" * 60)
    print("  G2 raw -> LeRobot video -> LeRobot images")
    print("=" * 60)
    print(f"  Input          : {input_dir}")
    print(f"  HF_LEROBOT_HOME: {hf_home}")
    print(f"  Video repo     : {video_root}")
    print(f"  Images repo    : {hf_home / args.images_repo_id}")
    print(f"  Convert python : {convert_python}")
    print(f"  Predecode py   : {predecode_python}")
    print()

    convert_cmd = [
        str(convert_python),
        str(args.convert_script),
        "--input_dir",
        str(input_dir),
        "--output_dir",
        str(video_root),
        "--task_name",
        task_name,
        "--fps",
        str(args.fps),
    ]
    if args.instruction:
        convert_cmd.extend(["--instruction", args.instruction])
    if args.max_episodes > 0:
        convert_cmd.extend(["--max_episodes", str(args.max_episodes)])
    if args.no_align_videos:
        convert_cmd.append("--no-align-videos")
    convert_cmd.extend(["--num-workers", str(args.num_workers)])
    convert_cmd.extend(["--video-workers", str(args.video_workers)])
    if args.dry_run:
        convert_cmd.append("--dry_run")

    predecode_cmd = [
        str(predecode_python),
        str(args.predecode_script),
        "--hf-lerobot-home",
        str(hf_home),
        "--input-repo-id",
        args.video_repo_id,
        "--output-repo-id",
        args.images_repo_id,
        "--num-workers",
        str(args.num_workers),
        "--image-format",
        args.image_format,
        "--jpeg-quality",
        str(args.jpeg_quality),
    ]
    if args.resize:
        predecode_cmd.extend(["--resize", str(args.resize[0]), str(args.resize[1])])
    if args.overwrite:
        predecode_cmd.append("--overwrite")
    if args.strict_alignment:
        predecode_cmd.append("--strict-alignment")

    if args.dry_run:
        if not args.skip_convert:
            _run(convert_cmd)
        if not args.skip_predecode:
            print("\n# would run predecode:\n", " ".join(str(c) for c in predecode_cmd))
        return 0

    if not args.skip_convert:
        _run(convert_cmd)
    else:
        if not video_root.is_dir():
            print(f"ERROR: --skip-convert but video dataset missing: {video_root}", file=sys.stderr)
            return 1
        print(f"Skipping convert; using existing video dataset: {video_root}")

    if args.skip_predecode:
        print(f"\nDone (video only): {video_root}")
        return 0

    _run(predecode_cmd)
    print(f"\nDone: images dataset at {hf_home / args.images_repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
