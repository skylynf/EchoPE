#!/usr/bin/env python3
"""

对比 LoRA 微调 与 冻结特征+分类头 的性能

在相同数据划分（多随机种子）下分别训练 LoRA 模型和冻结特征基线，汇总对比。

用法:
  ../.venv/bin/python compare_with_baseline.py                        # 默认参数
  ../.venv/bin/python compare_with_baseline.py --seeds 42 123 456     # 自定义 seed
  ../.venv/bin/python compare_with_baseline.py --rank 4 --head mlp    # 调整 LoRA 参数
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
from echo_paths import ECHO_ROOT, setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from train_lora import (  # noqa: E402
    DATASET_ROOT,
    HEAD_REGISTRY,
    VIEWS,
    LoRAEchoPrimeClassifier,
    PEVideoDataset,
    build_classifier,
    collect_samples,
    evaluate,
    inject_lora,
    samples_from_feature_cache,
)

# 从 baseline 导入冻结特征的基线组件
PE_RUN = HERE.parent / "baseline"
sys.path.insert(0, str(PE_RUN))
from train_pe_classifier import (  # noqa: E402
    CACHE_FILE,
    EmbeddingDataset,
    build_classifier as build_baseline_classifier,
    evaluate as evaluate_baseline,
    train_one_epoch as train_baseline_epoch,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── 冻结特征基线（单次运行）──────────────────────────────────────────────────────
def run_baseline(
    head_name: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """训练冻结特征 + 分类头，返回测试集指标。"""
    set_seed(seed)
    n_total = len(labels)
    indices = list(range(n_total))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels.numpy(), random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=labels[temp_idx].numpy(), random_state=seed,
    )

    def make_ds(idx):
        return EmbeddingDataset(embeddings[idx], labels[idx])

    train_labels = labels[train_idx]
    class_counts = torch.bincount(train_labels)
    sample_weights = (1.0 / class_counts.float())[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    train_loader = DataLoader(make_ds(train_idx), batch_size=128, sampler=sampler)
    val_loader = DataLoader(make_ds(val_idx), batch_size=128, shuffle=False)
    test_loader = DataLoader(make_ds(test_idx), batch_size=128, shuffle=False)

    # 使用 baseline 的分类头构建器（兼容其 args namespace）
    baseline_args = argparse.Namespace(
        hidden=args.hidden, dropout=args.head_dropout,
        num_blocks=args.num_blocks,
        attn_tokens=16, attn_heads=4, attn_layers=2,
    )
    model = build_baseline_classifier(head_name, baseline_args).to(device)

    class_weight = (n_total / (2.0 * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

    best_val_acc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, 61):
        train_baseline_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_y, val_pred, _ = evaluate_baseline(model, val_loader, device)
        val_acc = accuracy_score(val_y, val_pred)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 15:
            break

    model.load_state_dict(best_state)
    y_true, y_pred, y_prob = evaluate_baseline(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


# ── LoRA 微调（单次运行）─────────────────────────────────────────────────────────
def run_lora(
    samples: list[tuple[str, int, str]],
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    """训练 LoRA 微调模型，返回测试集指标。"""
    set_seed(seed)

    labels_all = [s[1] for s in samples]
    indices = list(range(len(samples)))

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=labels_all, random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5,
        stratify=[labels_all[i] for i in temp_idx],
        random_state=seed,
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]

    train_labels = torch.tensor([s[1] for s in train_samples])
    class_counts = torch.bincount(train_labels)
    sample_weights = (1.0 / class_counts.float())[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    train_loader = DataLoader(
        PEVideoDataset(train_samples), batch_size=args.batch,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        PEVideoDataset(val_samples), batch_size=args.batch,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        PEVideoDataset(test_samples), batch_size=args.batch,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # 构建模型
    weights_path = ECHO_ROOT / "model_data" / "weights" / "echo_prime_encoder.pt"
    ckpt = torch.load(str(weights_path), map_location="cpu")
    encoder = torchvision.models.video.mvit_v2_s()
    encoder.head[-1] = nn.Linear(encoder.head[-1].in_features, 512)
    encoder.load_state_dict(ckpt)
    for p in encoder.parameters():
        p.requires_grad = False

    inject_lora(encoder, rank=args.rank, alpha=args.alpha,
                lora_dropout=args.lora_dropout, target_modules=args.target_modules)

    classifier = build_classifier(
        args.head, args.hidden, args.head_dropout, num_blocks=args.num_blocks,
    )
    model = LoRAEchoPrimeClassifier(encoder, classifier).to(device)

    lora_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and ("lora_A" in n or "lora_B" in n)]
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": head_params, "lr": args.head_lr, "weight_decay": 1e-4},
    ])

    n_total = len(train_labels)
    class_weight = (n_total / (2.0 * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_auc = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, args.lora_epochs + 1):
        model.train()
        optimizer.zero_grad()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y) / args.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        val_y, val_pred, val_prob = evaluate(model, val_loader, device)
        try:
            val_auc = roc_auc_score(val_y, val_prob)
        except ValueError:
            val_auc = float("nan")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            break

    model.load_state_dict(best_state)
    model.to(device)
    y_true, y_pred, y_prob = evaluate(model, test_loader, device)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")
    return {"acc": acc, "f1": f1, "auc": auc}


# ── 汇总打印 ──────────────────────────────────────────────────────────────────────
def print_comparison(
    baseline_results: list[dict[str, float]],
    lora_results: list[dict[str, float]],
    seeds: list[int],
) -> None:
    metrics = ["acc", "f1", "auc"]
    col_w = 18

    print(f"\n{'='*72}")
    print("  LoRA vs 冻结特征基线 对比（测试集，均值±标准差）")
    print(f"{'='*72}")

    header = f"{'方法':<20}"
    for m in metrics:
        header += f"{m:>{col_w}}"
    print(header)
    print("-" * 72)

    for method_name, results in [("冻结特征+头", baseline_results), ("LoRA微调", lora_results)]:
        row = f"{method_name:<20}"
        for m in metrics:
            vals = [r[m] for r in results]
            mu, sd = float(np.nanmean(vals)), float(np.nanstd(vals))
            row += f"{mu:.4f}±{sd:.4f}".rjust(col_w)
        print(row)

    print(f"{'='*72}")

    # 逐 seed 详情
    print(f"\n── 逐 seed 测试集详情 ──")
    print(f"  {'seed':>6}  {'方法':<16}  {'acc':>8}  {'f1':>8}  {'auc':>8}")
    for i, seed in enumerate(seeds):
        b = baseline_results[i]
        l = lora_results[i]
        print(f"  {seed:6d}  {'冻结特征+头':<16}  {b['acc']:8.4f}  {b['f1']:8.4f}  {b['auc']:8.4f}")
        print(f"  {seed:6d}  {'LoRA微调':<16}  {l['acc']:8.4f}  {l['f1']:8.4f}  {l['auc']:8.4f}")

    # 逐指标 winner
    print(f"\n  逐指标最优:")
    for m in metrics:
        b_mean = float(np.nanmean([r[m] for r in baseline_results]))
        l_mean = float(np.nanmean([r[m] for r in lora_results]))
        winner = "LoRA微调" if l_mean > b_mean else "冻结特征+头"
        delta = abs(l_mean - b_mean)
        print(f"    {m}: {winner}  (Δ={delta:.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA vs 冻结特征基线 对比实验")
    parser.add_argument("--dataset", default=str(DATASET_ROOT))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    # LoRA 参数
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05, dest="lora_dropout")
    parser.add_argument("--target-modules", nargs="+", default=["qkv", "proj"],
                        dest="target_modules")
    # 共用
    parser.add_argument("--head", default="mlp", choices=list(HEAD_REGISTRY))
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--head-dropout", type=float, default=0.3, dest="head_dropout")
    parser.add_argument("--num-blocks", type=int, default=3, dest="num_blocks")
    # LoRA 训练
    parser.add_argument("--lora-epochs", type=int, default=30, dest="lora_epochs")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3, dest="head_lr")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    parser.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 加载冻结特征缓存
    cache_path = Path(CACHE_FILE)
    if not cache_path.exists():
        print(f"[ERROR] 特征缓存不存在: {cache_path}")
        print("请先运行 baseline/train_pe_classifier.py 提取特征。")
        sys.exit(1)

    cache = torch.load(cache_path, map_location="cpu")
    embeddings = cache["embeddings"]
    labels = cache["labels"]

    # LoRA 需要原始视频：优先用缓存里的 paths（与嵌入行对齐），否则扫描 hope_dataset
    samples = samples_from_feature_cache(cache)
    sample_src = "特征缓存(paths)"
    if samples is None:
        samples = collect_samples(Path(args.dataset))
        sample_src = f"目录扫描 (--dataset)"
    if len(samples) == 0:
        print(
            "\n[ERROR] LoRA 需要可用的视频路径，但当前样本数为 0。\n"
            f"  已尝试数据集目录: {Path(args.dataset).resolve()}\n"
            "  请任选其一：\n"
            "  1) 在 baseline 重新提取特征并写入 paths：\n"
            "       python train_pe_classifier.py --force-extract\n"
            "  2) 或指定 hope_dataset 根目录（默认在 EchoPrime 上一级）：  --dataset /path/to/hope_dataset"
        )
        sys.exit(1)
    if len(samples) != len(labels):
        print(
            f"[WARN] 视频列表条数 ({len(samples)}) 与缓存 labels 条数 ({len(labels)}) 不一致，"
            "LoRA 与基线可能不对齐；建议重新 --force-extract。"
        )

    print(f"\nLoRA 视频列表: {len(samples)} 条（来源: {sample_src}）")
    print(f"\n对比方法: 冻结特征+头 vs LoRA微调")
    print(f"Seeds: {args.seeds}\n")

    baseline_results = []
    lora_results = []

    for i, seed in enumerate(args.seeds):
        print(f"\n{'━'*60}")
        print(f"  Seed {seed}  ({i+1}/{len(args.seeds)})")
        print(f"{'━'*60}")

        print("  训练冻结特征基线...", end="", flush=True)
        b = run_baseline(args.head, embeddings, labels, args, seed, device)
        baseline_results.append(b)
        print(f"  acc={b['acc']:.4f}  f1={b['f1']:.4f}  auc={b['auc']:.4f}")

        print("  训练 LoRA 微调...")
        l = run_lora(samples, args, seed, device)
        lora_results.append(l)
        print(f"  acc={l['acc']:.4f}  f1={l['f1']:.4f}  auc={l['auc']:.4f}")

    print_comparison(baseline_results, lora_results, args.seeds)


if __name__ == "__main__":
    main()
