#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


FOCUS_PHENOTYPES = [
    "pulmonary_artery_pressure_continuous",
    "rv_systolic_function_depressed",
    "right_ventricle_dilation",
    "left_atrium_dilation",
    "right_atrium_dilation",
]

VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="按单视频粒度运行 EchoPrime phenotype 推理，并与 PE/Normal 标签对齐。"
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("../dataset/preprocessed"),
        help="预处理数据根目录（相对 EchoPrime 仓库根）",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("../dataset/preprocessed/manifest_preprocessed.csv"),
        help="manifest_preprocessed.csv 路径",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/text_run/results"),
        help="输出目录（相对 EchoPrime 仓库根）",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅跑前 N 个样本，默认跑全部",
    )
    ap.add_argument(
        "--label",
        choices=["pe", "normal", "all"],
        default="all",
        help="按标签过滤待跑样本",
    )
    ap.add_argument(
        "--view",
        action="append",
        default=None,
        help="仅跑指定 parsed_view，可多次传入",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="忽略已有结果并从头运行",
    )
    ap.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="每处理多少个样本增量写一次 CSV",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=50,
        help="传给 EchoPrime.predict_metrics 的 top-k 候选数",
    )
    return ap.parse_args()


def _json_float(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def resolve_video_path(row: pd.Series, dataset_root: Path) -> Path | None:
    label_dir = "PE" if str(row["label"]).lower() == "pe" else "Normal"
    candidates = [
        Path(str(row.get("output_video", ""))),
        Path(str(row.get("source_video", ""))),
        dataset_root / label_dir / str(row["parsed_view"]) / f"{row['id']}.mp4",
        dataset_root / label_dir / str(row["parsed_view"]) / f"{row['id']}.MP4",
        dataset_root / label_dir / str(row["parsed_view"]) / f"{row['id']}.mov",
        dataset_root / label_dir / str(row["parsed_view"]) / f"{row['id']}.avi",
        dataset_root / label_dir / str(row["parsed_view"]) / f"{row['id']}.mkv",
    ]
    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def preprocess_single_video(ep, utils_module, video_path: Path) -> torch.Tensor:
    pixels = utils_module.read_video_rgb_numpy(str(video_path))
    if pixels is None or len(pixels) == 0:
        raise ValueError(f"空视频或无法读取: {video_path}")

    x = np.zeros((len(pixels), ep.video_size, ep.video_size, 3), dtype=np.float32)
    for idx in range(len(x)):
        x[idx] = utils_module.crop_and_scale(pixels[idx])

    tensor = torch.as_tensor(x, dtype=torch.float).permute([3, 0, 1, 2])
    tensor.sub_(ep.mean).div_(ep.std)

    if tensor.shape[1] < ep.frames_to_take:
        padding = torch.zeros(
            (
                3,
                ep.frames_to_take - tensor.shape[1],
                ep.video_size,
                ep.video_size,
            ),
            dtype=torch.float,
        )
        tensor = torch.cat((tensor, padding), dim=1)

    clip = tensor[:, 0 : ep.frames_to_take : ep.frame_stride, :, :]
    return torch.stack([clip], dim=0)


def build_result_row(
    row: pd.Series,
    video_path: Path | None,
    focus_preds: dict[str, float | None],
    elapsed_s: float | None,
    status: str,
    error: str | None,
) -> dict[str, object]:
    result = {
        "id": str(row["id"]),
        "label": str(row["label"]).lower(),
        "label_binary": 1 if str(row["label"]).lower() == "pe" else 0,
        "parsed_view": str(row["parsed_view"]),
        "status": status,
        "video_path": str(video_path) if video_path is not None else "",
        "elapsed_s": elapsed_s,
        "error": error or "",
    }
    result.update(focus_preds)
    return result


def save_partial(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.sort_values(["label", "parsed_view", "id"], inplace=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()

    experiments_dir = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(experiments_dir))
    from echo_paths import setup_echo_root_cwd

    echo_root = setup_echo_root_cwd()

    import utils  # noqa: WPS433
    from echo_prime import EchoPrime  # noqa: WPS433

    dataset_root = (echo_root / args.dataset_root).resolve()
    manifest_path = (echo_root / args.manifest).resolve()
    output_dir = (echo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_csv = output_dir / "pe_phenotype_predictions.csv"
    full_json_path = output_dir / "pe_phenotype_predictions_full.jsonl"
    run_summary_path = output_dir / "run_summary.json"

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"数据目录不存在: {dataset_root}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest 不存在: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    manifest = manifest.loc[manifest["status"].eq("ok")].copy()
    manifest["label"] = manifest["label"].astype(str).str.lower()

    if args.label != "all":
        manifest = manifest.loc[manifest["label"].eq(args.label)].copy()
    if args.view:
        keep_views = {v.upper() for v in args.view}
        manifest = manifest.loc[manifest["parsed_view"].astype(str).str.upper().isin(keep_views)].copy()
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    previous_rows: list[dict[str, object]] = []
    done_ids: set[str] = set()
    if predictions_csv.exists() and not args.overwrite:
        previous = pd.read_csv(predictions_csv)
        previous_rows = previous.to_dict(orient="records")
        done_ids = set(previous.loc[previous["status"].eq("ok"), "id"].astype(str))

    pending = manifest.loc[~manifest["id"].astype(str).isin(done_ids)].copy()
    pending.sort_values(["label", "parsed_view", "id"], inplace=True)

    print(f"echo_root={echo_root}")
    print(f"dataset_root={dataset_root}")
    print(f"manifest_rows={len(manifest)} pending_rows={len(pending)}")

    ep = EchoPrime()
    all_rows = list(previous_rows)
    full_rows: list[dict[str, object]] = []
    started = time.time()
    new_count = 0
    missing_count = 0
    failed_count = 0

    if args.overwrite and full_json_path.exists():
        full_json_path.unlink()

    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="EchoPrime inference"):
        sample_id = str(row["id"])
        video_path = resolve_video_path(row, dataset_root)
        if video_path is None:
            missing_count += 1
            result_row = build_result_row(
                row=row,
                video_path=None,
                focus_preds={name: None for name in FOCUS_PHENOTYPES},
                elapsed_s=None,
                status="missing_video",
                error="video_not_found",
            )
            all_rows.append(result_row)
            new_count += 1
            continue

        t0 = time.time()
        try:
            stack = preprocess_single_video(ep, utils, video_path)
            enc = ep.encode_study(stack, visualize=False)
            preds = ep.predict_metrics(enc, k=args.k)
            elapsed_s = round(time.time() - t0, 4)

            focus_preds = {name: _json_float(preds.get(name)) for name in FOCUS_PHENOTYPES}
            result_row = build_result_row(
                row=row,
                video_path=video_path,
                focus_preds=focus_preds,
                elapsed_s=elapsed_s,
                status="ok",
                error=None,
            )
            all_rows.append(result_row)

            full_entry = {
                "id": sample_id,
                "label": str(row["label"]).lower(),
                "parsed_view": str(row["parsed_view"]),
                "video_path": str(video_path),
                "elapsed_s": elapsed_s,
                "predict_metrics": {key: _json_float(value) for key, value in preds.items()},
            }
            full_rows.append(full_entry)
        except Exception as exc:  # noqa: BLE001
            failed_count += 1
            result_row = build_result_row(
                row=row,
                video_path=video_path,
                focus_preds={name: None for name in FOCUS_PHENOTYPES},
                elapsed_s=round(time.time() - t0, 4),
                status="error",
                error=str(exc),
            )
            all_rows.append(result_row)

        new_count += 1
        if new_count % args.save_every == 0:
            save_partial(predictions_csv, all_rows)
            if full_rows:
                with full_json_path.open("a", encoding="utf-8") as fh:
                    for entry in full_rows:
                        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                full_rows = []

    save_partial(predictions_csv, all_rows)
    if full_rows:
        with full_json_path.open("a", encoding="utf-8") as fh:
            for entry in full_rows:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = {
        "echo_root": str(echo_root),
        "dataset_root": str(dataset_root),
        "manifest_path": str(manifest_path),
        "predictions_csv": str(predictions_csv),
        "full_jsonl": str(full_json_path),
        "n_manifest_filtered": int(len(manifest)),
        "n_existing_ok_skipped": int(len(done_ids)),
        "n_new_rows_processed": int(new_count),
        "n_missing_video": int(missing_count),
        "n_error": int(failed_count),
        "wall_time_s": round(time.time() - started, 3),
        "focus_phenotypes": FOCUS_PHENOTYPES,
        "k": int(args.k),
    }
    run_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
