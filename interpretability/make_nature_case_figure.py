#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, to_rgb

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
FULL_RUN_ROOT = EXP_ROOT / "classification"
LORA_RUN_ROOT = EXP_ROOT / "finetuning"
ORIG_CWD = Path.cwd().resolve()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
if str(FULL_RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(FULL_RUN_ROOT))
if str(LORA_RUN_ROOT) not in sys.path:
    sys.path.insert(0, str(LORA_RUN_ROOT))

from extract_attributions import (  # noqa: E402
    build_coarse_view_idx,
    collect_attention_map,
    compute_occlusion_evidence,
    denormalize_video,
    enable_attention_capture,
    disable_attention_capture,
    normalize_map,
    pe_logit_value,
    softmax_prob,
)
from load_interpret_model import load_interpret_artifacts  # noqa: E402
from finetuning.train_lora import preprocess_single_video  # noqa: E402


DEFAULT_CASE_ID = "test_936692778d9241eba7872e79e8b2af1b_f0e46537"
DEFAULT_MODELS = ("frozen_head", "lora_kd_combo", "full_finetune")

MODEL_SPECS = {
    "frozen_head": {
        "display_name": "Frozen head",
        "checkpoint": EXP_ROOT / "classification" / "outputs_frozen_head" / "seed-2024" / "pooled" / "best_checkpoint.pt",
    },
    "lora_kd_combo": {
        "display_name": "LoRA + KD",
        "checkpoint": EXP_ROOT / "classification" / "outputs_lora_kd_combo" / "seed-2024" / "pooled" / "best_checkpoint.pt",
    },
    "full_finetune": {
        "display_name": "Full finetune",
        "checkpoint": EXP_ROOT / "classification" / "outputs" / "seed-2024" / "pooled" / "best_checkpoint.pt",
    },
}

COLORS = {
    "blue_dark": "#3F74A3",
    "blue_mid": "#699BC5",
    "blue_light": "#D1DFE6",
    "red_dark": "#B54646",
    "red_mid": "#D18F90",
    "brown_dark": "#794E47",
    "brown_mid": "#9C7B57",
    "brown_light": "#A78F95",
    "orange": "#F18F2B",
    "purple": "#C994AF",
    "green": "#8FBB8F",
}


@dataclass
class CaseFigurePayload:
    model_key: str
    display_name: str
    split: str
    case_id: str
    record: dict[str, object]
    prob_pe: float
    pred_label: int
    video_gray: torch.Tensor
    grad_map: torch.Tensor
    attention_map: torch.Tensor
    temporal_attention: torch.Tensor
    occlusion_delta_prob: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-style case figures.")
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, default=HERE / "paper_figures")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--occlusion-span", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    parser.add_argument("--models", nargs="+", choices=sorted(MODEL_SPECS), default=list(DEFAULT_MODELS))
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    return path if path.is_absolute() else (ORIG_CWD / path).resolve()


def set_nature_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 7.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 6.0,
            "figure.titlesize": 8.0,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_cmap(*hex_colors: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("custom_map", list(hex_colors))


ATTN_CMAP = plt.get_cmap("jet")
GRAD_CMAP = plt.get_cmap("jet")


def hex_to_rgb01(hex_color: str) -> np.ndarray:
    return np.array(to_rgb(hex_color), dtype=np.float32)


def overlay_frame(
    frame: torch.Tensor,
    heatmap: torch.Tensor,
    cmap: LinearSegmentedColormap,
    alpha: float = 0.62,
) -> np.ndarray:
    frame_np = frame.detach().cpu().numpy().astype(np.float32)
    frame_np = frame_np - frame_np.min()
    if frame_np.max() > 0:
        frame_np = frame_np / frame_np.max()
    base = np.repeat(frame_np[..., None], 3, axis=-1)
    base = np.clip(0.12 + 0.88 * base, 0.0, 1.0)
    heat = normalize_map(heatmap).detach().cpu().numpy().astype(np.float32)
    color = cmap(heat)[..., :3].astype(np.float32)
    weight = np.power(heat[..., None], 0.9) * alpha
    overlay = base * (1.0 - weight) + color * weight
    return np.clip(overlay, 0.0, 1.0)


def enhance_heatmap(heatmap: torch.Tensor, percentile: float = 99.0, gamma: float = 0.55) -> torch.Tensor:
    heat = heatmap.detach().float().cpu()
    flat = heat.flatten()
    if flat.numel() == 0:
        return heat
    q = float(torch.quantile(flat, min(max(percentile / 100.0, 0.0), 1.0)).item())
    if q <= 1e-8:
        return normalize_map(heat)
    heat = (heat / q).clamp(0.0, 1.0)
    heat = heat.pow(gamma)
    return normalize_map(heat)


def grad_overlay_frame(frame: torch.Tensor, heatmap: torch.Tensor) -> np.ndarray:
    enhanced = enhance_heatmap(heatmap, percentile=98.8, gamma=0.42)
    return overlay_frame(frame, enhanced, GRAD_CMAP, alpha=0.82)


def attention_overlay_frame(frame: torch.Tensor, heatmap: torch.Tensor) -> np.ndarray:
    enhanced = enhance_heatmap(heatmap, percentile=99.3, gamma=0.8)
    return overlay_frame(frame, enhanced, ATTN_CMAP, alpha=0.72)


def parse_case_id(case_id: str) -> tuple[str, str]:
    if "_" not in case_id:
        raise ValueError(f"Invalid case id: {case_id}")
    split, remainder = case_id.split("_", 1)
    case_stem, _digest = remainder.rsplit("_", 1)
    return split, case_stem


def find_record(artifacts, case_id: str) -> tuple[str, dict[str, object]]:
    split, case_stem = parse_case_id(case_id)
    for record in artifacts.records_by_split[split]:
        if str(record["id"]) == case_stem:
            return split, record
    raise KeyError(f"Could not find case {case_id} in split {split}.")


def compute_case_payload(model_key: str, case_id: str, device: str, occlusion_span: int) -> CaseFigurePayload:
    spec = MODEL_SPECS[model_key]
    artifacts = load_interpret_artifacts(
        spec["checkpoint"],
        task="pooled",
        batch_size=1,
        num_workers=0,
        splits=("train", "val", "test"),
        device=device,
    )
    split, record = find_record(artifacts, case_id)
    patched = enable_attention_capture(artifacts.model.encoder)
    try:
        video = preprocess_single_video(str(record["path"])).to(artifacts.device)
        input_video = video.unsqueeze(0).detach()
        input_video.requires_grad_(True)
        artifacts.model.zero_grad(set_to_none=True)
        coarse_view_idx = None
        if getattr(artifacts.checkpoint_args, "head_view_mode", "none") == "coarse4":
            coarse_view_idx = build_coarse_view_idx(record, artifacts.device)

        out = artifacts.model(input_video, coarse_view_idx=coarse_view_idx)
        logits = out["logits"]
        pe_logit = logits[:, 1].sum()
        pe_logit.backward()

        grad = input_video.grad.detach()[0]
        grad_map = normalize_map((grad * input_video.detach()[0]).abs().mean(dim=0).cpu())
        base_prob = softmax_prob(logits)
        pred_label = int(torch.argmax(logits, dim=1).item())

        with torch.no_grad():
            attention_map, temporal_attention = collect_attention_map(
                artifacts.model.encoder,
                target_shape=(video.shape[1], video.shape[2], video.shape[3]),
            )
            temporal_evidence = compute_occlusion_evidence(
                artifacts,
                input_video=input_video,
                coarse_view_idx=coarse_view_idx,
                base_prob=base_prob,
                base_logit=pe_logit_value(logits),
                occlusion_spans=[occlusion_span],
            )

        occlusion_delta_prob = temporal_evidence[str(occlusion_span)]["delta_prob"].cpu()
        video_gray = denormalize_video(video).mean(dim=0).cpu()
        return CaseFigurePayload(
            model_key=model_key,
            display_name=str(spec["display_name"]),
            split=split,
            case_id=case_id,
            record=record,
            prob_pe=float(base_prob),
            pred_label=pred_label,
            video_gray=video_gray,
            grad_map=grad_map.cpu(),
            attention_map=attention_map.cpu(),
            temporal_attention=temporal_attention.cpu(),
            occlusion_delta_prob=occlusion_delta_prob,
        )
    finally:
        disable_attention_capture(patched)


def style_image_axis(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def format_prob(prob: float) -> str:
    return f"{prob:.3f}" if prob >= 0.001 else f"{prob:.2e}"


def build_caption(payload: CaseFigurePayload) -> str:
    label_name = str(payload.record["label_name"])
    pred_name = "PE" if payload.pred_label == 1 else "Normal"
    raw_view = str(payload.record["raw_view"])
    return (
        f"{payload.display_name}. Case-based interpretability panel for a {label_name} study in {raw_view} view "
        f"({payload.case_id}; predicted {pred_name}, P(PE)={format_prob(payload.prob_pe)}). "
        "Top row, frame-wise attention reference overlays. "
        "Middle row, frame-wise Grad x Input overlays. "
        "Bottom, raw occlusion delta probability and a scaled attention reference trace; "
        "the dashed horizontal line marks zero change. "
        "A separate companion panel reports the clip-averaged Mean Grad x Input overlay."
    )


def save_case_figure(payload: CaseFigurePayload, output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    n_frames = int(payload.video_gray.shape[0])
    fig_width = max(7.0, 0.39 * n_frames)
    fig_height = 4.15
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor="white")
    gs = fig.add_gridspec(
        3,
        n_frames,
        height_ratios=[1.0, 1.0, 0.74],
        hspace=0.005,
        wspace=0.05,
    )

    for idx in range(n_frames):
        attn_ax = fig.add_subplot(gs[0, idx])
        grad_ax = fig.add_subplot(gs[1, idx])

        attn_ax.imshow(attention_overlay_frame(payload.video_gray[idx], payload.attention_map[idx]))
        grad_ax.imshow(grad_overlay_frame(payload.video_gray[idx], payload.grad_map[idx]))

        style_image_axis(attn_ax)
        style_image_axis(grad_ax)

    curve_ax = fig.add_subplot(gs[2, :])
    x = np.arange(n_frames, dtype=np.float32)
    occlusion = payload.occlusion_delta_prob.detach().cpu().numpy().astype(np.float32)
    attn = payload.temporal_attention.detach().cpu().numpy().astype(np.float32)
    attn_max = float(attn.max()) if attn.size else 0.0
    occ_absmax = float(np.max(np.abs(occlusion))) if occlusion.size else 0.0
    scaled_attn = attn.copy()
    if attn_max > 0 and occ_absmax > 0:
        scaled_attn = (attn / attn_max) * (occ_absmax * 0.92)
    elif attn_max > 0:
        scaled_attn = attn / attn_max

    y_absmax = max(
        1e-4,
        float(np.max(np.abs(occlusion))) if occlusion.size else 0.0,
        float(np.max(np.abs(scaled_attn))) if scaled_attn.size else 0.0,
    )
    y_lim = y_absmax * 1.18

    for idx in range(n_frames):
        if idx % 2 == 0:
            curve_ax.axvspan(idx - 0.5, idx + 0.5, color=COLORS["blue_light"], alpha=0.12, linewidth=0)

    curve_ax.axhline(0.0, color=COLORS["brown_light"], linewidth=0.9, linestyle=(0, (3.2, 2.6)), zorder=1)
    curve_ax.plot(
        x,
        occlusion,
        color=COLORS["red_dark"],
        linewidth=1.35,
        marker="o",
        markersize=2.5,
        label="Occlusion raw delta",
        zorder=3,
    )
    curve_ax.plot(
        x,
        scaled_attn,
        color=COLORS["blue_dark"],
        linewidth=1.2,
        marker="o",
        markersize=2.2,
        label="Attention reference (scaled)",
        zorder=2,
    )
    curve_ax.set_xlim(-0.5, n_frames - 0.5)
    curve_ax.set_ylim(-y_lim, y_lim)
    curve_ax.set_xlabel("Frame")
    curve_ax.set_ylabel("Delta probability / scaled attention")
    tick_step = 2 if n_frames > 10 else 1
    tick_positions = np.arange(0, n_frames, tick_step)
    curve_ax.set_xticks(tick_positions)
    curve_ax.set_xticklabels([str(int(v + 1)) for v in tick_positions])
    curve_ax.legend(loc="lower right", frameon=False, ncol=1, handlelength=2.6, columnspacing=1.2)
    curve_ax.spines["top"].set_visible(False)
    curve_ax.spines["right"].set_visible(False)
    curve_ax.spines["left"].set_color("black")
    curve_ax.spines["bottom"].set_color("black")
    curve_ax.tick_params(axis="both", colors="black")
    curve_ax.yaxis.label.set_color("black")
    curve_ax.xaxis.label.set_color("black")

    fig.text(
        0.02,
        0.985,
        f"{payload.display_name} | {str(payload.record['raw_view'])} view | P(PE)={format_prob(payload.prob_pe)}",
        ha="left",
        va="top",
        fontsize=7.0,
        color="black",
    )
    fig.subplots_adjust(left=0.02, right=0.995, top=0.94, bottom=0.11)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in formats:
        out_path = output_dir / f"{payload.case_id}_{payload.model_key}.{fmt}"
        is_svg = fmt.lower() == "svg"
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=("none" if is_svg else "white"),
            transparent=is_svg,
        )
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def save_mean_grad_figure(payload: CaseFigurePayload, output_dir: Path, formats: list[str], dpi: int) -> list[Path]:
    fig = plt.figure(figsize=(2.35, 2.35), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    mean_overlay = overlay_frame(payload.video_gray.mean(dim=0), payload.grad_map.mean(dim=0), GRAD_CMAP, alpha=0.55)
    ax.imshow(mean_overlay)
    ax.set_title("Mean Grad x Input", pad=4, color="black")
    style_image_axis(ax)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for fmt in formats:
        out_path = output_dir / f"{payload.case_id}_{payload.model_key}_mean_grad.{fmt}"
        is_svg = fmt.lower() == "svg"
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=("none" if is_svg else "white"),
            transparent=is_svg,
        )
        saved_paths.append(out_path)
    plt.close(fig)
    return saved_paths


def main() -> None:
    args = parse_args()
    set_nature_style()
    output_dir = resolve_output_dir(args.output_dir)
    captions: list[str] = []

    for model_key in args.models:
        payload = compute_case_payload(
            model_key=model_key,
            case_id=args.case_id,
            device=args.device,
            occlusion_span=int(args.occlusion_span),
        )
        save_case_figure(payload, output_dir=output_dir, formats=list(args.formats), dpi=int(args.dpi))
        save_mean_grad_figure(payload, output_dir=output_dir, formats=list(args.formats), dpi=int(args.dpi))
        captions.append(build_caption(payload))

    captions_path = output_dir / f"{args.case_id}_captions.txt"
    captions_path.write_text("\n\n".join(captions) + "\n", encoding="utf-8")
    for caption in captions:
        print(caption)
        print()
    print(f"Saved captions to: {captions_path}")


if __name__ == "__main__":
    main()
