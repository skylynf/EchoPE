#!/usr/bin/env python3
"""
从视频目录批量运行 EchoPrime（与 EchoPrimeDemo 流程一致）。

用法（在 EchoPrime 目录下，且已存在 model_data 等资源）：
  ../.venv/bin/python run_testset.py ../testset -o ../testset_predictions.json

若使用 uv，请显式指定本仓库的 venv，避免 uv 误用 Conda 的 Python 3.9 并扫描到损坏的 pyodbc egg-info：
  uv run --python ../.venv/bin/python run_testset.py ../testset -o ../testset_predictions.json

默认：INPUT 下所有 .mp4/.mkv/.avi/.mov 视为同一 study（一次 predict_metrics / 一次报告），
即「同一次超声检查里的多段录像」，模型会按切面权重把多段 clip 一起编码再检索。
若 INPUT 的直接子目录里均含有视频，则每个子目录单独作为一个 study。
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path


def _list_video_paths(folder: str) -> list[str]:
    exts = ("*.mp4", "*.mkv", "*.avi", "*.mov")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(f"{folder}/**/{ext}", recursive=True))
    return sorted(paths)


def _json_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _has_video_files(folder: Path) -> bool:
    exts = {".mp4", ".mkv", ".avi", ".mov"}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            return True
    return False


def discover_study_dirs(root: Path) -> list[Path]:
    """若存在「仅含视频的子文件夹」，则按子文件夹拆分；否则整个 root 为一次 study。"""
    subs = [p for p in root.iterdir() if p.is_dir()]
    if subs and all(_has_video_files(s) for s in subs):
        return sorted(subs)
    return [root]


def main() -> None:
    # echo_prime/utils 依赖相对仓库根的 model_data、assets —— 必须与 echo_paths 一致
    experiments_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(experiments_dir))
    from echo_paths import setup_echo_root_cwd

    setup_echo_root_cwd()

    ap = argparse.ArgumentParser(description="EchoPrime：对测试集视频目录跑 predict_metrics")
    ap.add_argument("input", type=Path, help="含视频文件的目录（或若干 study 子目录）")
    ap.add_argument("-o", "--output", type=Path, help="将预测写入 JSON 文件")
    ap.add_argument("--report", action="store_true", help="同时生成文本报告（较慢）")
    ap.add_argument(
        "--views",
        action="store_true",
        help="输出每段 clip 的粗粒度视角分类（与 ConvNeXt 切面头一致）",
    )
    args = ap.parse_args()

    root = args.input.resolve()
    if not root.is_dir():
        print(f"不是目录: {root}", file=sys.stderr)
        sys.exit(1)

    from echo_prime import EchoPrime

    ep = EchoPrime()
    study_dirs = discover_study_dirs(root)
    out: dict[str, dict] = {}

    for d in study_dirs:
        key = d.name if d != root else str(root)
        stack = ep.process_mp4s(str(d))
        enc = ep.encode_study(stack, visualize=False)
        preds = ep.predict_metrics(enc)
        entry = {
            "n_clips": int(stack.shape[0]),
            "predict_metrics": {k: _json_float(v) for k, v in preds.items()},
        }
        if args.views:
            vpaths = _list_video_paths(str(d))
            view_names = ep.get_views(stack, return_view_list=True)
            entry["clip_views"] = [
                {"clip_index": i, "view": vn} for i, vn in enumerate(view_names)
            ]
            if len(vpaths) == len(view_names):
                for i, p in enumerate(vpaths):
                    entry["clip_views"][i]["file"] = Path(p).name
            elif vpaths:
                entry["clip_views_note"] = (
                    f"glob 到 {len(vpaths)} 个视频文件，成功解码 {len(view_names)} 段；"
                    "file 字段省略（可能部分文件损坏被跳过）"
                )
        if args.report:
            entry["report"] = ep.generate_report(enc)
        out[key] = entry
        print(f"[{key}] clips={stack.shape[0]} phenotypes={len(preds)}")

    text = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print("已写入", args.output)
    else:
        print(text)


if __name__ == "__main__":
    main()
