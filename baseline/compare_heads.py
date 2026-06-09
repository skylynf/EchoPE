#!/usr/bin/env python3
"""
分类头对比脚本
在相同数据划分（多随机种子）下训练所有分类头，汇总并对比指标。

用法（在项目根目录下，使用 .venv）：
  ../.venv/bin/python compare_heads.py                             # 默认：3 个头 × 3 个 seed
  ../.venv/bin/python compare_heads.py --heads mlp residual        # 只对比两个头
  ../.venv/bin/python compare_heads.py --seeds 42 123 456 789      # 自定义 seed 列表
  ../.venv/bin/python compare_heads.py --epochs 80 --patience 20   # 调整训练参数
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
from echo_paths import setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# 从 train_pe_classifier 导入共用组件
from train_pe_classifier import (  # noqa: E402
    CACHE_FILE,
    VIEWS,
    EmbeddingDataset,
    HEAD_REGISTRY,
    build_classifier,
    evaluate,
    print_metrics,
    train_one_epoch,
)


# ── 单次训练运行 ──────────────────────────────────────────────────────────────────
def run_single(
    head_name: str,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    all_views: list[str],
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """
    用指定 seed 训练一个分类头，返回验证集和测试集指标字典。
    返回: {"val": {acc, f1, auc}, "test": {acc, f1, auc}}
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_total = len(labels)
    indices = list(range(n_total))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels.numpy(),
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        stratify=labels[temp_idx].numpy(),
        random_state=seed,
    )

    def make_ds(idx: list[int]) -> EmbeddingDataset:
        return EmbeddingDataset(embeddings[idx], labels[idx])

    train_labels = labels[train_idx]
    class_counts = torch.bincount(train_labels)
    sample_weights = (1.0 / class_counts.float())[train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(make_ds(train_idx), batch_size=args.batch, sampler=sampler)
    val_loader = DataLoader(make_ds(val_idx), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(make_ds(test_idx), batch_size=args.batch, shuffle=False)

    model = build_classifier(head_name, args).to(device)

    class_weight = (n_total / (2.0 * class_counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        val_y, val_pred, _ = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_y, val_pred)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)

    def compute_metrics(loader: DataLoader, split_idx: list[int]) -> dict[str, float]:
        y_true, y_pred, y_prob = evaluate(model, loader, device)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc = float("nan")
        return {"acc": acc, "f1": f1, "auc": auc}

    return {
        "val": compute_metrics(val_loader, val_idx),
        "test": compute_metrics(test_loader, test_idx),
    }


# ── 汇总打印 ──────────────────────────────────────────────────────────────────────
def print_summary(results: dict[str, list[dict[str, dict[str, float]]]]) -> None:
    """
    results: {head_name: [run_result, ...]}
    每个 run_result = {"val": {acc,f1,auc}, "test": {acc,f1,auc}}
    """
    metrics = ["acc", "f1", "auc"]
    splits = ["val", "test"]
    col_w = 18

    # 表头
    header_parts = [f"{'分类头':<12}"]
    for split in splits:
        for m in metrics:
            label = f"{split}/{m}"
            header_parts.append(f"{label:>{col_w}}")
    print("\n" + "=" * (12 + col_w * len(metrics) * len(splits) + 4))
    print("  多 seed 对比汇总（均值 ± 标准差）")
    print("=" * (12 + col_w * len(metrics) * len(splits) + 4))
    print("".join(header_parts))
    print("-" * (12 + col_w * len(metrics) * len(splits) + 4))

    # 每个头的统计
    summary: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
    for head_name, run_list in results.items():
        summary[head_name] = {}
        row = [f"{head_name:<12}"]
        for split in splits:
            summary[head_name][split] = {}
            for m in metrics:
                vals = [r[split][m] for r in run_list]
                mu = float(np.nanmean(vals))
                sd = float(np.nanstd(vals))
                summary[head_name][split][m] = (mu, sd)
                cell = f"{mu:.4f}±{sd:.4f}"
                row.append(f"{cell:>{col_w}}")
        print("".join(row))

    print("=" * (12 + col_w * len(metrics) * len(splits) + 4))

    # 找出每列最优头（按均值）
    best_row = [f"{'[最优]':<12}"]
    for split in splits:
        for m in metrics:
            best_head = max(summary, key=lambda h: summary[h][split][m][0])
            best_row.append(f"{'→ ' + best_head:>{col_w}}")
    print("".join(best_row))
    print("=" * (12 + col_w * len(metrics) * len(splits) + 4))


# ── 详细结果（每个头每个 seed 的测试集指标）────────────────────────────────────────
def print_per_run(
    results: dict[str, list[dict[str, dict[str, float]]]],
    seeds: list[int],
) -> None:
    print("\n── 逐 seed 测试集结果 ──────────────────────────────────────")
    for head_name, run_list in results.items():
        print(f"\n  [{head_name}]")
        print(f"    {'seed':>6}  {'acc':>8}  {'f1':>8}  {'auc':>8}")
        for seed, r in zip(seeds, run_list):
            m = r["test"]
            print(f"    {seed:6d}  {m['acc']:8.4f}  {m['f1']:8.4f}  {m['auc']:8.4f}")


# ── main ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比不同分类头在 PE 任务上的性能（多 seed 均值±标准差）"
    )
    parser.add_argument(
        "--cache", default=str(CACHE_FILE), help="特征缓存 .pt 文件路径"
    )
    parser.add_argument(
        "--heads",
        nargs="+",
        default=list(HEAD_REGISTRY),
        choices=list(HEAD_REGISTRY),
        help="要对比的分类头列表（默认全部）",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="随机种子列表，每个种子独立训练一次",
    )
    parser.add_argument("--epochs", type=int, default=60, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch", type=int, default=128, help="batch size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout 比例")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    # mlp / residual 共用
    parser.add_argument("--hidden", type=int, default=256, help="MLP/残差块隐层维度")
    # residual 专用
    parser.add_argument("--num-blocks", type=int, default=3, dest="num_blocks",
                        help="[residual] 残差块数量")
    # attention 专用
    parser.add_argument("--attn-tokens", type=int, default=16, dest="attn_tokens",
                        help="[attention] token 数量（须整除 512）")
    parser.add_argument("--attn-heads", type=int, default=4, dest="attn_heads",
                        help="[attention] 注意力头数")
    parser.add_argument("--attn-layers", type=int, default=2, dest="attn_layers",
                        help="[attention] Transformer 层数")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个 seed 的训练进度",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载特征缓存
    cache_path = Path(args.cache)
    if not cache_path.exists():
        print(f"[ERROR] 特征缓存不存在: {cache_path}")
        print("请先运行 train_pe_classifier.py 提取特征，或使用 --force-extract 参数。")
        sys.exit(1)

    print(f"加载特征缓存: {cache_path}")
    cache = torch.load(cache_path, map_location="cpu")
    embeddings: torch.Tensor = cache["embeddings"]
    labels: torch.Tensor = cache["labels"]
    all_views: list[str] = cache["views"]

    n_total = len(labels)
    n_normal = (labels == 0).sum().item()
    n_pe = (labels == 1).sum().item()
    print(f"样本统计: 总计={n_total}, Normal={n_normal}, PE={n_pe}")
    print(f"\n将对比以下分类头: {args.heads}")
    print(f"随机种子列表: {args.seeds}")
    print(f"每头总训练次数: {len(args.seeds)}\n")

    results: dict[str, list[dict[str, dict[str, float]]]] = {}

    total_runs = len(args.heads) * len(args.seeds)
    run_idx = 0

    for head_name in args.heads:
        results[head_name] = []
        # 打印该头的可训练参数量
        dummy_args = argparse.Namespace(**vars(args))
        sample_model = build_classifier(head_name, dummy_args)
        n_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        print(f"{'─'*60}")
        print(f"分类头: {head_name}  |  可训练参数: {n_params:,}")
        print(f"{'─'*60}")

        for seed in args.seeds:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] head={head_name}  seed={seed}  ...", end="", flush=True)
            run_result = run_single(
                head_name, embeddings, labels, all_views, args, seed, device
            )
            results[head_name].append(run_result)
            m = run_result["test"]
            print(f"  测试集 acc={m['acc']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")

    # 汇总输出
    print_summary(results)
    print_per_run(results, args.seeds)


if __name__ == "__main__":
    main()
