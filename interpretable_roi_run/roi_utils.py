from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.path import Path as MplPath

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
INTERPRET_ROOT = EXP_ROOT / "interpretable_run"
LORA_ROOT = EXP_ROOT / "lora_run"
FULL_RUN_ROOT = EXP_ROOT / "full_run"
ORIG_CWD = Path(os.environ.get("PWD", str(Path.cwd()))).resolve()
for path in (EXP_ROOT, INTERPRET_ROOT, LORA_ROOT, FULL_RUN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

AUTO_MASK_NAMES = ("foreground", "background", "sector_border", "probe_near_field")
ANATOMY_MASK_NAMES = ("rv", "lv", "septum", "d_sign_region", "rv_lv_boundary")
CONTEXT_MASK_NAMES = ("background", "sector_border", "probe_near_field")
MEAN = torch.tensor([29.110628, 28.076836, 29.096405]).reshape(3, 1, 1, 1)
STD = torch.tensor([47.989223, 46.456997, 47.20083]).reshape(3, 1, 1, 1)


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ORIG_CWD / candidate).resolve()


def load_predictions(output_dir: str | Path) -> pd.DataFrame:
    out_dir = resolve_path(output_dir)
    csv_path = out_dir / "embedding_analysis" / "sample_predictions.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing predictions CSV: {csv_path}")
    return pd.read_csv(csv_path)


def load_attribution_payloads(output_dir: str | Path) -> list[dict[str, object]]:
    attr_dir = resolve_path(output_dir) / "attributions"
    if not attr_dir.is_dir():
        raise FileNotFoundError(f"Missing attributions directory: {attr_dir}")
    payloads = []
    for path in sorted(attr_dir.glob("*.pt")):
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        payload["_source_path"] = str(path)
        payloads.append(payload)
    return payloads


def case_rows_from_attributions(output_dir: str | Path) -> pd.DataFrame:
    rows = []
    for payload in load_attribution_payloads(output_dir):
        record = payload["record"]
        rows.append(
            {
                "case_id": str(payload["case_id"]),
                "split": str(payload["split"]),
                "path": str(record["path"]),
                "label": int(record["label"]),
                "label_name": str(record["label_name"]),
                "pred_label": int(record.get("pred_label", -1)),
                "correct": int(record.get("correct", -1)),
                "raw_view": str(record["raw_view"]),
                "coarse_view": str(record["coarse_view"]),
                "attribution_path": str(payload["_source_path"]),
            }
        )
    return pd.DataFrame(rows)


def denormalize_video(video: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(video.device, dtype=video.dtype)
    std = STD.to(video.device, dtype=video.dtype)
    return video * std + mean


def load_preprocessed_video(video_path: str | Path) -> torch.Tensor:
    # Import lazily so metadata-only scripts do not require video dependencies such as cv2.
    from lora_run.train_lora import preprocess_single_video  # noqa: WPS433

    return preprocess_single_video(str(resolve_path(video_path)), training=False)


def _pool2d(mask: torch.Tensor, kernel_size: int, mode: str) -> torch.Tensor:
    if kernel_size <= 1:
        return mask.bool()
    pad = kernel_size // 2
    x = mask.float().unsqueeze(0).unsqueeze(0)
    if mode == "dilate":
        pooled = F.max_pool2d(x, kernel_size=kernel_size, stride=1, padding=pad)
    elif mode == "erode":
        pooled = -F.max_pool2d(-x, kernel_size=kernel_size, stride=1, padding=pad)
    else:
        raise ValueError(f"Unknown pooling mode: {mode}")
    return pooled[0, 0].bool()


def generate_context_masks(
    video: torch.Tensor,
    *,
    threshold: float = 8.0,
    border_width: int = 5,
    probe_fraction: float = 0.18,
) -> dict[str, torch.Tensor]:
    """Generate weak acquisition-context masks from a normalized video.

    Args:
        video: Tensor shaped `(3, T, H, W)`.
    """
    denorm = denormalize_video(video).float().clamp(min=0)
    image = denorm.mean(dim=(0, 1))
    foreground = image > float(threshold)
    if int(foreground.sum().item()) == 0:
        foreground = image > float(image.mean().item())
    dilated = _pool2d(foreground, kernel_size=max(1, 2 * border_width + 1), mode="dilate")
    eroded = _pool2d(foreground, kernel_size=max(1, 2 * border_width + 1), mode="erode")
    sector_border = (dilated ^ eroded) & dilated
    background = ~foreground

    ys = torch.where(foreground)[0]
    probe_near_field = torch.zeros_like(foreground)
    if ys.numel() > 0:
        y_min = int(ys.min().item())
        y_max = int(ys.max().item())
        y_stop = y_min + max(1, int(round((y_max - y_min + 1) * float(probe_fraction))))
        probe_near_field[: y_stop + 1] = foreground[: y_stop + 1]

    return {
        "foreground": foreground.cpu(),
        "background": background.cpu(),
        "sector_border": sector_border.cpu(),
        "probe_near_field": probe_near_field.cpu(),
    }


def save_mask_payload(
    path: str | Path,
    *,
    case_row: dict[str, object],
    masks: dict[str, torch.Tensor],
    source: str,
) -> None:
    out_path = resolve_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": str(case_row["case_id"]),
        "path": str(case_row["path"]),
        "split": str(case_row.get("split", "")),
        "label_name": str(case_row.get("label_name", "")),
        "coarse_view": str(case_row.get("coarse_view", "")),
        "source": source,
        "masks": {name: mask.bool().cpu() for name, mask in masks.items()},
    }
    torch.save(payload, out_path)


def load_mask_payload(mask_dir: str | Path, case_id: str) -> dict[str, object]:
    path = resolve_path(mask_dir) / f"{case_id}.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing mask payload for {case_id}: {path}")
    return torch.load(str(path), map_location="cpu", weights_only=False)


def load_or_generate_mask_payload(
    mask_dir: str | Path,
    *,
    case_id: str,
    record: dict[str, object],
    save_generated: bool = True,
) -> tuple[dict[str, object], bool]:
    """Load a mask payload, or generate weak context masks for an attribution case.

    This keeps ROI quantification usable when the selected case panel and the
    older `attributions/*.pt` detailed cases do not fully overlap.
    """
    try:
        return load_mask_payload(mask_dir, case_id), False
    except FileNotFoundError:
        video = load_preprocessed_video(str(record["path"]))
        masks = generate_context_masks(video)
        payload = {
            "case_id": str(case_id),
            "path": str(record["path"]),
            "split": str(record.get("split", "")),
            "label_name": str(record.get("label_name", "")),
            "coarse_view": str(record.get("coarse_view", "")),
            "source": "generated_weak_context",
            "masks": {name: mask.bool().cpu() for name, mask in masks.items()},
        }
        if save_generated:
            save_mask_payload(resolve_path(mask_dir) / f"{case_id}.pt", case_row=payload, masks=masks, source="generated_weak_context")
        return payload, True


def mask_case_lookup(mask_dir: str | Path) -> dict[str, str]:
    manifest_path = resolve_path(mask_dir) / "mask_manifest.csv"
    if not manifest_path.is_file():
        return {}
    manifest = pd.read_csv(manifest_path)
    if "path" not in manifest.columns or "case_id" not in manifest.columns:
        return {}
    return {str(row["path"]): str(row["case_id"]) for _, row in manifest.iterrows()}


def polygon_to_mask(points: list[list[float]] | list[tuple[float, float]], height: int, width: int) -> torch.Tensor:
    if len(points) < 3:
        return torch.zeros((height, width), dtype=torch.bool)
    polygon = MplPath(np.asarray(points, dtype=np.float32))
    yy, xx = np.mgrid[:height, :width]
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)
    mask = polygon.contains_points(coords).reshape(height, width)
    return torch.from_numpy(mask.astype(bool))


def load_annotations(path: str | Path | None, height: int = 224, width: int = 224) -> dict[str, dict[str, torch.Tensor]]:
    if path is None:
        return {}
    ann_path = resolve_path(path)
    if not ann_path.is_file():
        print(
            f"[WARN] ROI annotation file not found, continuing with weak masks only: {ann_path}",
            file=sys.stderr,
        )
        return {}
    text = ann_path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if ann_path.suffix.lower() == ".jsonl":
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        entries = payload.get("cases", payload if isinstance(payload, list) else [])
    annotations: dict[str, dict[str, torch.Tensor]] = {}
    for entry in entries:
        case_id = str(entry["case_id"])
        regions = entry.get("regions", {})
        masks: dict[str, torch.Tensor] = {}
        for name, value in regions.items():
            if isinstance(value, dict):
                points = value.get("polygon", [])
            else:
                points = value
            masks[str(name)] = polygon_to_mask(points, height=height, width=width)
        annotations[case_id] = masks
    return annotations


def spatialize_attribution(score_map: torch.Tensor, frame_index: int | None = None) -> torch.Tensor:
    score = score_map.detach().float().cpu()
    if score.ndim == 3:
        if frame_index is None:
            return score.mean(dim=0).clamp(min=0)
        frame_index = max(0, min(int(frame_index), int(score.shape[0]) - 1))
        return score[frame_index].clamp(min=0)
    if score.ndim == 2:
        return score.clamp(min=0)
    raise ValueError(f"Expected 2D or 3D attribution map, got shape {tuple(score.shape)}")


def normalized_entropy(values: torch.Tensor, eps: float = 1e-8) -> float:
    flat = values.float().clamp(min=0).flatten()
    total = float(flat.sum().item())
    if total <= eps or flat.numel() <= 1:
        return 0.0
    probs = flat / flat.sum()
    entropy = -(probs * (probs + eps).log()).sum().item()
    return float(entropy / math.log(flat.numel() + eps))


def roi_metrics(score_map: torch.Tensor, mask: torch.Tensor, *, top_ratio: float = 0.10) -> dict[str, float]:
    score = score_map.float().clamp(min=0)
    mask = mask.bool()
    if score.shape != mask.shape:
        mask = F.interpolate(
            mask.float().unsqueeze(0).unsqueeze(0),
            size=tuple(score.shape),
            mode="nearest",
        )[0, 0].bool()
    total = float(score.sum().item())
    area_fraction = float(mask.float().mean().item())
    mass = float(score[mask].sum().item() / total) if total > 0 else 0.0
    enrichment = float(mass / area_fraction) if area_fraction > 0 else float("nan")
    flat = score.flatten()
    mask_flat = mask.flatten()
    if flat.numel() == 0:
        top_overlap = 0.0
    else:
        k = max(1, int(round(flat.numel() * float(top_ratio))))
        top_idx = torch.topk(flat, k=k).indices
        top_overlap = float(mask_flat[top_idx].float().mean().item())
    return {
        "roi_mass": mass,
        "roi_area_fraction": area_fraction,
        "roi_enrichment": enrichment,
        "top10_overlap": top_overlap,
    }


def summarize_metric_rows(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return {"n_rows": 0}
    out: dict[str, object] = {"n_rows": int(len(df))}
    for col in ("roi_mass", "roi_enrichment", "top10_overlap", "delta_prob", "delta_logit"):
        if col in df.columns:
            out[col] = {
                "mean": float(df[col].mean()),
                "median": float(df[col].median()),
            }
    return out


def top_indices(curve: np.ndarray, k: int = 3) -> set[int]:
    if curve.size == 0:
        return set()
    values = np.clip(curve.astype(float), a_min=0.0, a_max=None)
    if float(values.max()) <= 0.0:
        values = np.abs(curve.astype(float))
    k = min(k, values.size)
    return set(int(idx) for idx in np.argsort(-values)[:k])


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    left_rank = np.argsort(np.argsort(left)).astype(np.float64)
    right_rank = np.argsort(np.argsort(right)).astype(np.float64)
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def jaccard(left: set[int], right: set[int]) -> float:
    if not left and not right:
        return float("nan")
    return float(len(left & right) / max(1, len(left | right)))

