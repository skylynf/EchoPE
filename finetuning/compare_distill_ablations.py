#!/usr/bin/env python3
"""
KD + view 辅助 消融对比实验

固定 7 个核心配置 × 5 折 CV，输出测试集 acc / f1 / auc 均值±std。
划分由 experiments/utils/split_cv.py 预先生成（与其他库实验保持一致）。
回答两个研究问题：
  RQ1: KD（蒸馏）单独用 vs 与 LoRA 联合用，哪个更好？
  RQ2: view 辅助监督对 LoRA / KD 的边际贡献？

配置（与 plan 一致）：
  A. Frozen + head                      —— 冻结 encoder，仅训练分类头（baseline）
  B. LoRA + head                        —— 现有 LoRA（对照）
  C. Full FT (partial-stage4) + head    —— 后段解冻（与 LoRA 同等"可训练量级"对照）
  D. Frozen + head + KD-cos             —— 验证 KD 对完全冻结 encoder 是否有意义
  E. Full FT (partial-stage4) + KD-combo —— "KD 单独用"（回答 RQ1）
  F. LoRA + KD-combo                    —— 联合（回答 RQ1）
  G. LoRA + KD-combo + view-aux=ce      —— 全开（回答 RQ2）

用法（在 finetuning 目录下）：
  ../../.venv/bin/python compare_distill_ablations.py
  ../../.venv/bin/python compare_distill_ablations.py --folds 0 1 2 --epochs 80
  ../../.venv/bin/python compare_distill_ablations.py --configs A B F G   # 只跑指定配置
  ../../.venv/bin/python compare_distill_ablations.py --smoke              # 端到端冒烟
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

HERE = Path(__file__).resolve().parent
EXP_ROOT = HERE.parent
if str(EXP_ROOT) not in sys.path:
    sys.path.insert(0, str(EXP_ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from echo_paths import DEFAULT_HOPE_DATASET, ECHO_ROOT, setup_echo_root_cwd  # noqa: E402

setup_echo_root_cwd()

from train_lora import (  # noqa: E402
    DEFAULT_CV_SPLITS_DIR,
    load_cv_fold_samples,
)
from train_lora_distill import (  # noqa: E402
    DATASET_ROOT,
    PEVideoDatasetWithView,
    build_classifier,
    build_encoder,
    evaluate,
    LoRAEchoPrimeDistillModel,
    train_one_epoch,
    view_collate,
)
from distill import load_frozen_teacher  # noqa: E402
from aux_heads import load_view_teacher  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
#  配置定义
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """单个消融配置；只覆盖与 default_args 不同的字段。"""

    name: str
    desc: str
    overrides: dict = field(default_factory=dict)


CONFIGS: dict[str, Config] = {
    "A": Config("A. Frozen+head",
                "完全冻结 encoder + 训练分类头",
                {"lora": "off", "ft_mode": "frozen",
                 "kd_mode": "none", "view_aux": "none"}),
    "B": Config("B. LoRA+head",
                "现有 LoRA + 分类头（对照）",
                {"lora": "on", "ft_mode": "lora",
                 "kd_mode": "none", "view_aux": "none"}),
    "C": Config("C. PartialFT+head",
                "后 4 个 block 解冻 + 分类头",
                {"lora": "off", "ft_mode": "partial-stage4",
                 "kd_mode": "none", "view_aux": "none"}),
    "D": Config("D. Frozen+KD-cos",
                "冻结 encoder + cos KD（理论上 ≈ A）",
                {"lora": "off", "ft_mode": "frozen",
                 "kd_mode": "cos", "view_aux": "none"}),
    "E": Config("E. PartialFT+KD-combo",
                "KD 单独用（无 LoRA，后段解冻 + combo KD）",
                {"lora": "off", "ft_mode": "partial-stage4",
                 "kd_mode": "combo", "view_aux": "none"}),
    "F": Config("F. LoRA+KD-combo",
                "LoRA + combo KD（联合）",
                {"lora": "on", "ft_mode": "lora",
                 "kd_mode": "combo", "view_aux": "none"}),
    "G": Config("G. LoRA+KD+view-ce",
                "全开：LoRA + combo KD + view-aux ce",
                {"lora": "on", "ft_mode": "lora",
                 "kd_mode": "combo", "view_aux": "ce"}),
}


def default_args(args_cli: argparse.Namespace) -> argparse.Namespace:
    """构造 train_lora_distill 期望的 Namespace 默认值，再由 Config 覆盖。"""
    return argparse.Namespace(
        # 数据
        dataset=args_cli.dataset,
        feature_cache="",
        # encoder
        lora="on",
        ft_mode="lora",
        rank=args_cli.rank,
        alpha=args_cli.alpha,
        lora_dropout=0.05,
        target_modules=["qkv", "proj"],
        # 主分类头
        head="mlp",
        hidden=256,
        head_dropout=0.3,
        num_blocks=3,
        # KD
        kd_mode="none",
        kd_weight=args_cli.kd_weight,
        kd_warmup_epochs=2,
        lam_rkd=1.0,
        # view
        view_aux="none",
        view_weight=args_cli.view_weight,
        view_kd_weight=0.3,
        view_kd_temperature=2.0,
        # 训练
        epochs=args_cli.epochs,
        lr=args_cli.lr,
        head_lr=args_cli.head_lr,
        batch=args_cli.batch,
        grad_accum=args_cli.grad_accum,
        weight_decay=0.01,
        warmup_epochs=args_cli.warmup_epochs,
        patience=args_cli.patience,
        seed=0,
        num_workers=args_cli.num_workers,
        teacher_device=args_cli.teacher_device,
        smoke=False,
        save_prefix="",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  单次运行
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single(
    cfg: Config,
    fold: int,
    cli_args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int]:
    """运行单个 (config, fold) 组合，返回测试集指标及 val 选 ckpt 信息。"""
    set_seed(cli_args.seed)

    args = default_args(cli_args)
    for k, v in cfg.overrides.items():
        setattr(args, k, v)
    args.seed = cli_args.seed

    # ── 数据划分（统一使用 experiments/utils/split_cv.py 的五折 CV）──
    fold_samples = load_cv_fold_samples(
        Path(cli_args.cv_splits), fold, Path(cli_args.dataset),
    )
    train_samples = fold_samples["train"]
    val_samples = fold_samples["val"]
    test_samples = fold_samples["test"]

    if cli_args.smoke:
        # 冒烟：每个 split 取若干条，保证两类都有
        def _take(items: list, n: int) -> list:
            cls0 = [s for s in items if s[1] == 0][: n // 2]
            cls1 = [s for s in items if s[1] == 1][: n - len(cls0)]
            return cls0 + cls1
        train_samples = _take(train_samples, 4)
        val_samples = _take(val_samples, 2)
        test_samples = _take(test_samples, 2)

    train_labels = torch.tensor([s[1] for s in train_samples])
    class_counts = torch.bincount(train_labels, minlength=2)
    sample_weights = (1.0 / class_counts.float().clamp(min=1))[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(train_labels), replacement=True)

    train_loader = DataLoader(
        PEVideoDatasetWithView(train_samples), batch_size=args.batch,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True,
        collate_fn=view_collate,
    )
    val_loader = DataLoader(
        PEVideoDatasetWithView(val_samples), batch_size=args.batch,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        collate_fn=view_collate,
    )
    test_loader = DataLoader(
        PEVideoDatasetWithView(test_samples), batch_size=args.batch,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
        collate_fn=view_collate,
    )

    # ── 模型 ──
    weights_path = ECHO_ROOT / "model_data" / "weights" / "echo_prime_encoder.pt"
    encoder, _ = build_encoder(args, weights_path)
    classifier = build_classifier(
        args.head, args.hidden, args.head_dropout, num_blocks=args.num_blocks,
    )
    model = LoRAEchoPrimeDistillModel(
        encoder, classifier,
        view_aux=args.view_aux, head_dropout=args.head_dropout,
    ).to(device)

    teacher_device = (torch.device(args.teacher_device)
                      if args.teacher_device else device)
    teacher_kd = (load_frozen_teacher(weights_path, teacher_device)
                  if args.kd_mode != "none" else None)
    teacher_view = (load_view_teacher(
        ECHO_ROOT / "model_data" / "weights" / "view_classifier.pt", teacher_device,
    ) if args.view_aux in ("kd", "both") else None)

    # ── 优化器 ──
    enc_params = [p for n, p in model.encoder.named_parameters() if p.requires_grad]
    head_params = [p for p in model.classifier.parameters() if p.requires_grad]
    aux_params: list[nn.Parameter] = []
    if model.view_ce_head is not None:
        aux_params.extend(model.view_ce_head.parameters())
    if model.view_kd_head is not None:
        aux_params.extend(model.view_kd_head.parameters())

    param_groups = []
    if enc_params:
        param_groups.append({"params": enc_params, "lr": args.lr,
                             "weight_decay": args.weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr,
                             "weight_decay": 1e-4})
    if aux_params:
        param_groups.append({"params": aux_params, "lr": args.head_lr,
                             "weight_decay": 1e-4})
    if not param_groups:
        # 全冻结 + 无 head（理论上不会发生，但兜底）
        return {"acc": float("nan"), "f1": float("nan"), "auc": float("nan")}

    optimizer = torch.optim.AdamW(param_groups)

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(0, args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    n_total = len(train_labels)
    class_weight = (n_total / (2.0 * class_counts.clamp(min=1).float())).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── 训练（早停依据：val_auc；每 epoch 打印 val_acc / val_auc）──
    best_val_auc = -1.0
    best_val_acc_at_best = float("nan")
    best_epoch = 0
    last_epoch = 0
    best_state: dict | None = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        train_one_epoch(
            epoch, model, train_loader, teacher_kd, teacher_view,
            criterion, optimizer, scheduler, scaler, device, args,
        )
        val_y, val_pred, val_prob = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_y, val_pred)
        try:
            val_auc = roc_auc_score(val_y, val_prob)
        except ValueError:
            val_auc = float("nan")

        improved = (val_auc > best_val_auc) if not math.isnan(val_auc) else False
        if improved:
            best_val_auc = val_auc
            best_val_acc_at_best = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        auc_s = f"{val_auc:.4f}" if not math.isnan(val_auc) else "nan"
        print(
            f"  epoch {epoch:>3}/{args.epochs}  val_acc={val_acc:.4f}  val_auc={auc_s}  "
            f"best_ep={best_epoch}  best_val_auc={best_val_auc:.4f}",
            flush=True,
        )

        if no_improve >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)

    bva = (f"{best_val_acc_at_best:.4f}"
           if not math.isnan(best_val_acc_at_best) else "nan")
    print(
        f"  [val 汇总] last_epoch={last_epoch}  best_epoch={best_epoch}  "
        f"best_val_auc={best_val_auc:.4f}  val_acc@best={bva}",
        flush=True,
    )

    test_y, test_pred, test_prob = evaluate(model, test_loader, device)
    acc = accuracy_score(test_y, test_pred)
    f1 = f1_score(test_y, test_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(test_y, test_prob)
    except ValueError:
        auc = float("nan")

    # 释放显存
    del model, encoder, classifier, teacher_kd, teacher_view
    torch.cuda.empty_cache()

    return {
        "acc": acc,
        "f1": f1,
        "auc": auc,
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "best_val_auc": float(best_val_auc),
        "best_val_acc": float(best_val_acc_at_best)
        if not math.isnan(best_val_acc_at_best) else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  汇总打印
# ═══════════════════════════════════════════════════════════════════════════════

def print_summary(
    results: dict[str, list[dict[str, float | int]]],
    folds: list[int],
) -> None:
    metrics = ["acc", "f1", "auc"]
    name_w = 24
    col_w = 16

    print(f"\n{'='*92}")
    print(f"  消融对比汇总（测试集，均值±std，N folds = {len(folds)}）")
    print(f"{'='*92}")

    header = f"{'配置':<{name_w}}"
    for m in metrics:
        header += f"{m:>{col_w}}"
    print(header)
    print("-" * 92)

    # 每个指标的最佳均值
    best_per_metric = {m: -1.0 for m in metrics}
    means = {}
    for cid, runs in results.items():
        means[cid] = {m: float(np.nanmean([r[m] for r in runs])) for m in metrics}
        for m in metrics:
            if means[cid][m] > best_per_metric[m]:
                best_per_metric[m] = means[cid][m]

    for cid, runs in results.items():
        cfg = CONFIGS[cid]
        row = f"{cfg.name:<{name_w}}"
        for m in metrics:
            vals = [r[m] for r in runs]
            mu = float(np.nanmean(vals))
            sd = float(np.nanstd(vals))
            cell = f"{mu:.4f}±{sd:.4f}"
            if abs(mu - best_per_metric[m]) < 1e-6:
                cell = "*" + cell
            row += cell.rjust(col_w)
        print(row)

    print(f"{'='*92}")
    print("  * 标注每列最佳均值")

    # 逐 fold 详情
    print(f"\n── 逐 fold 测试集详情 ──")
    print(f"  {'fold':>6}  {'config':<24}  {'acc':>8}  {'f1':>8}  {'auc':>8}")
    for cid, runs in results.items():
        for s, r in zip(folds, runs):
            print(f"  {s:6d}  {CONFIGS[cid].name:<24}  "
                  f"{r['acc']:8.4f}  {r['f1']:8.4f}  {r['auc']:8.4f}")

    # val 选 ckpt（与训练时早停一致：按 val_auc 取 best）
    print(f"\n── val 选 ckpt（best_val_auc，早停 patience）──")
    print(f"  {'fold':>6}  {'config':<24}  {'best_ep':>8}  {'last_ep':>8}  "
          f"{'val_acc@best':>12}  {'best_val_auc':>12}")
    for cid, runs in results.items():
        for s, r in zip(folds, runs):
            bva = r.get("best_val_acc", float("nan"))
            bvu = r.get("best_val_auc", float("nan"))
            bva_s = f"{float(bva):12.4f}" if not math.isnan(float(bva)) else f"{'nan':>12}"
            bvu_s = f"{float(bvu):12.4f}" if not math.isnan(float(bvu)) else f"{'nan':>12}"
            print(f"  {s:6d}  {CONFIGS[cid].name:<24}  "
                  f"{int(r.get('best_epoch', 0)):8d}  {int(r.get('last_epoch', 0)):8d}  "
                  f"{bva_s}  {bvu_s}")

    # RQ 分析
    print(f"\n── RQ 分析 ──")
    if "B" in results and "F" in results:
        for m in metrics:
            b = float(np.nanmean([r[m] for r in results["B"]]))
            f = float(np.nanmean([r[m] for r in results["F"]]))
            print(f"  RQ1 [LoRA vs LoRA+KD]    Δ{m} = {f - b:+.4f}  "
                  f"(B={b:.4f} → F={f:.4f})")
    if "E" in results and "F" in results:
        for m in metrics:
            e = float(np.nanmean([r[m] for r in results["E"]]))
            f = float(np.nanmean([r[m] for r in results["F"]]))
            print(f"  RQ1 [KD-only vs LoRA+KD] Δ{m} = {f - e:+.4f}  "
                  f"(E={e:.4f} → F={f:.4f})")
    if "F" in results and "G" in results:
        for m in metrics:
            f = float(np.nanmean([r[m] for r in results["F"]]))
            g = float(np.nanmean([r[m] for r in results["G"]]))
            print(f"  RQ2 [+view-aux]          Δ{m} = {g - f:+.4f}  "
                  f"(F={f:.4f} → G={g:.4f})")


# ═══════════════════════════════════════════════════════════════════════════════
#  多 GPU 并行 worker
# ═══════════════════════════════════════════════════════════════════════════════

def _gpu_worker(
    gpu_id: int,
    cli_args_dict: dict,
    task_queue,
    result_queue,
    log_dir: str,
) -> None:
    """spawn 子进程：绑定单张 GPU，持续从 task_queue 拉取 (cid, fold) 任务。

    - CUDA_VISIBLE_DEVICES 必须在 import torch 之前设置，否则该进程会看到所有 GPU
    - 每个 worker 把 stdout/stderr 重定向到 log_dir/gpu{N}.log，避免多进程 tqdm 串流
    """
    import os as _os
    import sys as _sys
    import traceback

    _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = _os.path.join(log_dir, f"gpu{gpu_id}.log")
    log_f = open(log_path, "w", buffering=1)
    _sys.stdout = log_f
    _sys.stderr = log_f

    import torch as _torch  # 触发 CUDA 初始化时只看到指定的那张卡
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    cli_args = argparse.Namespace(**cli_args_dict)
    print(f"[worker GPU={gpu_id}] device={device}", flush=True)

    while True:
        task = task_queue.get()
        if task is None:
            break
        cid, fold = task
        cfg = CONFIGS[cid]
        print(f"\n>>> 开始: {cfg.name}  fold={fold}", flush=True)
        try:
            r = run_single(cfg, fold, cli_args, device)
            print(f"<<< 完成: {cfg.name} fold={fold}  "
                  f"acc={r['acc']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}  "
                  f"best_ep={r['best_epoch']} last_ep={r['last_epoch']}",
                  flush=True)
            result_queue.put((cid, fold, r, None))
        except Exception:
            tb = traceback.format_exc()
            print(f"<<< 失败: {cfg.name} fold={fold}\n{tb}", flush=True)
            result_queue.put((cid, fold, None, tb))


def run_parallel(
    args: argparse.Namespace,
    gpus: list[int],
) -> dict[str, list[dict[str, float | int]]]:
    """按 (config, fold) 任务粒度，把工作分配给 len(gpus) 个 worker。"""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    log_dir = HERE / "_parallel_logs"
    log_dir.mkdir(exist_ok=True)

    tasks = [(cid, fold) for cid in args.configs for fold in args.folds]
    for t in tasks:
        task_queue.put(t)
    for _ in gpus:
        task_queue.put(None)  # poison pill

    cli_args_dict = vars(args).copy()

    workers = []
    for gid in gpus:
        p = ctx.Process(
            target=_gpu_worker,
            args=(gid, cli_args_dict, task_queue, result_queue, str(log_dir)),
        )
        p.start()
        workers.append(p)

    print(f"\n并行模式: {len(gpus)} GPUs ({gpus}) × {len(tasks)} 任务  "
          f"(子进程日志: {log_dir})")

    received: dict[tuple[str, int], dict[str, float]] = {}
    n_total = len(tasks)
    for i in range(n_total):
        cid, fold, r, err = result_queue.get()
        if err is None:
            received[(cid, fold)] = r
            print(f"  [{i+1}/{n_total}]  {CONFIGS[cid].name:<24} fold={fold:>4}  "
                  f"→ acc={r['acc']:.4f}  f1={r['f1']:.4f}  auc={r['auc']:.4f}  "
                  f"best_ep={r['best_epoch']} last_ep={r['last_epoch']}")
        else:
            received[(cid, fold)] = {
                "acc": float("nan"),
                "f1": float("nan"),
                "auc": float("nan"),
                "best_epoch": 0,
                "last_epoch": 0,
                "best_val_auc": float("nan"),
                "best_val_acc": float("nan"),
            }
            print(f"  [{i+1}/{n_total}]  {CONFIGS[cid].name:<24} fold={fold:>4}  "
                  f"FAILED  (见 {log_dir}/gpu*.log)")

    for p in workers:
        p.join()

    results: dict[str, list[dict[str, float]]] = {cid: [] for cid in args.configs}
    for cid in args.configs:
        for fold in args.folds:
            results[cid].append(received[(cid, fold)])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KD + view 辅助 消融对比")
    p.add_argument("--dataset", default=str(DATASET_ROOT))
    p.add_argument("--cv-splits", default=str(DEFAULT_CV_SPLITS_DIR),
                   dest="cv_splits",
                   help="五折 CV 划分目录（包含 fold_0..fold_4），由 "
                        "experiments/utils/split_cv.py 生成；与其他库实验对齐")
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   help="使用哪些 fold（默认 0..4 五折全跑）")
    p.add_argument("--seed", type=int, default=42,
                   help="模型初始化/采样种子（数据划分由 CV 决定，与该 seed 无关）")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                   choices=list(CONFIGS.keys()),
                   help="要跑哪些配置（默认全部 A-G）")
    # LoRA 参数
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=16.0)
    # KD / view 权重
    p.add_argument("--kd-weight", type=float, default=0.5, dest="kd_weight")
    p.add_argument("--view-weight", type=float, default=0.3, dest="view_weight")
    # 训练
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--warmup-epochs", type=int, default=3, dest="warmup_epochs")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3, dest="head_lr")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    p.add_argument("--patience", type=int, default=15,
                   help="早停：验证集 val_auc 连续不提升的 epoch 数（默认 15）")
    p.add_argument("--num-workers", type=int, default=4, dest="num_workers")
    p.add_argument("--teacher-device", default="", dest="teacher_device")
    # 多 GPU 并行（按 (config, seed) 任务派发；每张卡同时只跑一个 job）
    p.add_argument("--gpus", default="",
                   help="逗号分隔 GPU id（如 '0,1,2'）；留空 = 单 GPU 串行执行")
    # 输出
    p.add_argument("--save-json", default="", dest="save_json",
                   help="将结果保存为 JSON")
    p.add_argument("--smoke", action="store_true",
                   help="冒烟：每类 4 条 + 2 epoch + 单 seed，仅验证管线")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gpus: list[int] = []
    if args.gpus.strip():
        gpus = [int(x) for x in args.gpus.replace(" ", "").split(",") if x != ""]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if gpus:
        print(f"主进程设备: {device}（仅作打印；并行 worker 各自绑定 1 张 GPU）")
    else:
        print(f"设备: {device}")

    # 校验 CV 划分目录
    cv_dir = Path(args.cv_splits)
    if not cv_dir.is_dir():
        print(f"[ERROR] CV 划分目录不存在: {cv_dir}\n"
              f"  请先运行 experiments/utils/split_cv.py 生成五折划分。")
        sys.exit(1)
    missing_folds = [f for f in args.folds
                     if not (cv_dir / f"fold_{f}").is_dir()]
    if missing_folds:
        print(f"[ERROR] 缺少 fold 目录: {missing_folds}（在 {cv_dir} 下）")
        sys.exit(1)

    if args.smoke:
        args.epochs = 2
        args.patience = 999
        args.batch = min(args.batch, 2)
        args.grad_accum = 1
        args.folds = args.folds[:1]
        print(f"[SMOKE] {len(args.folds)} fold × {len(args.configs)} 配置 × "
              f"{args.epochs} epoch")

    print(f"\nCV: {cv_dir}  folds={args.folds}  配置: {args.configs}")
    print(f"训练: epochs={args.epochs}  warmup={args.warmup_epochs}  "
          f"patience={args.patience}  lr={args.lr}  head_lr={args.head_lr}")

    if gpus and len(gpus) > 1 and not args.smoke:
        results = run_parallel(args, gpus)
    else:
        if gpus and len(gpus) == 1:
            import os as _os
            _os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"  (单 GPU 模式，绑定 cuda:{gpus[0]} → {device})")

        results = {cid: [] for cid in args.configs}
        total_runs = len(args.configs) * len(args.folds)
        run_idx = 0
        for cid in args.configs:
            cfg = CONFIGS[cid]
            for fold in args.folds:
                run_idx += 1
                print(f"\n{'━'*70}")
                print(f"  [{run_idx}/{total_runs}]  {cfg.name}  fold={fold}")
                print(f"  {cfg.desc}")
                print(f"{'━'*70}")
                r = run_single(cfg, fold, args, device)
                results[cid].append(r)
                print(f"  → test acc={r['acc']:.4f}  f1={r['f1']:.4f}  auc={r['auc']:.4f}  "
                      f"(best_epoch={r['best_epoch']}, last_epoch={r['last_epoch']})")

    print_summary(results, args.folds)

    if args.save_json:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({
                "folds": args.folds,
                "cv_splits": str(cv_dir),
                "configs": {cid: {"name": CONFIGS[cid].name,
                                   "desc": CONFIGS[cid].desc,
                                   "overrides": CONFIGS[cid].overrides,
                                   "results": results[cid]}
                            for cid in args.configs},
            }, f, indent=2)
        print(f"\n结果已保存 → {save_path}")


if __name__ == "__main__":
    main()
