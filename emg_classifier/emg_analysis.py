#!/usr/bin/env python3
"""
emg_analysis.py – CLI for NinaPro EMG classification grid search and quantization.

Replaces explore_emg.ipynb without any Jupyter/multiprocessing incompatibilities.
Multi-GPU parallelism uses fork-based ProcessPoolExecutor, which is safe here
because no CUDA context exists in the main process before the fork.

Modes
-----
  train     Run / resume the hyperparameter grid search.
  results   Load finished MLflow runs and print a ranked summary.
  quantize  Quantize a selected trained model (PTQ or QAT) and log to MLflow.

Examples
--------
  # Train all subjects across all configs on 7 GPUs:
  python emg_analysis.py train

  # Resume (skips already-finished runs):
  python emg_analysis.py train

  # Show results:
  python emg_analysis.py results

  # Quantize best config for subject 1 with PTQ:
  python emg_analysis.py quantize --subject 1 --method ptq

  # Override config values on the command line:
  python emg_analysis.py train --db 1 --subjects 1 2 3 --epochs 50
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Make emg_worker importable regardless of working directory
# ---------------------------------------------------------------------------
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from emg_worker import (
    ConvBnRelu,
    NinaProDatasetByRepetition,
    load_subject,
    run_task,
)


# ===========================================================================
# Configuration
# ===========================================================================

def load_config(config_file: Path) -> dict:
    if config_file.exists():
        with open(config_file) as f:
            cfg = json.load(f)
        print(f"Loaded config from {config_file.resolve()}")
    else:
        cfg = {}
        print(f"WARNING: {config_file} not found — using built-in defaults.")
    return cfg


def build_config(args: argparse.Namespace) -> dict:
    cfg_file = Path(args.config) if hasattr(args, "config") else Path("config.json")
    file_cfg = load_config(cfg_file)

    ninapro_dir = Path(args.ninapro_dir if args.ninapro_dir else
                       file_cfg.get("ninapro_dir", "../data/ninapro"))
    mlruns_dir  = args.mlruns_dir  if args.mlruns_dir  else file_cfg.get("mlruns_dir",  "mlruns")
    experiment  = args.experiment  if args.experiment  else file_cfg.get("mlflow_experiment", "ninapro_emg")

    return dict(
        db             = args.db,
        subjects       = args.subjects,
        ninapro_dir    = ninapro_dir,
        mlruns_dir     = mlruns_dir,
        experiment     = experiment,
        conv_config    = [tuple(k) for k in args.conv_config],
        window_sizes   = args.window_sizes,
        num_stages     = args.num_stages,
        train_reps     = args.train_reps,
        test_reps      = args.test_reps,
        window_step    = args.window_step,
        epochs         = args.epochs,
        lr             = args.lr,
        weight_decay   = args.weight_decay,
        batch_size     = args.batch_size,
        fs             = args.fs,
    )


# ===========================================================================
# GPU detection (without initialising the CUDA context)
# ===========================================================================

def detect_gpus() -> list[str]:
    """Return a list of device strings using nvidia-smi (no CUDA init)."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        if r.returncode == 0:
            names = [l for l in r.stdout.strip().splitlines() if l.strip()]
            if names:
                return [f"cuda:{i}" for i in range(len(names))]
    except Exception as e:
        print(f"GPU detection failed: {e}")

    if torch.backends.mps.is_available():
        return ["mps"]
    return ["cpu"]


# ===========================================================================
# MLflow helpers
# ===========================================================================

def get_finished_runs(mlruns_dir: str, experiment: str) -> set[str]:
    mlflow.set_tracking_uri(mlruns_dir)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return set()
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
    )
    return {r.info.run_name for r in runs}


# ===========================================================================
# Train mode
# ===========================================================================

def run_name_for(db, subject, window_size, kernels, num_stages,
                 model_type: str = "multiscale") -> str:
    if model_type == "conventional":
        return f"db{db}_s{subject}_w{window_size}_conventional"
    return f"db{db}_s{subject}_w{window_size}_k{'_'.join(map(str, kernels))}_stages{num_stages}"


def _build_pending(cfg: dict, model_type: str, finished: set) -> list:
    """Return list of (subject, kernels, window_size, num_stages, model_type) tuples."""
    pending = []
    db = cfg["db"]

    if model_type in ("multiscale", "both"):
        combos = list(itertools.product(cfg["conv_config"],
                                        cfg["window_sizes"],
                                        cfg["num_stages"]))
        for subject in cfg["subjects"]:
            for kernels, window_size, num_stages in combos:
                if run_name_for(db, subject, window_size, kernels, num_stages,
                                "multiscale") not in finished:
                    pending.append((subject, kernels, window_size, num_stages, "multiscale"))

    if model_type in ("conventional", "both"):
        for subject in cfg["subjects"]:
            for window_size in cfg["window_sizes"]:
                if run_name_for(db, subject, window_size, None, None,
                                "conventional") not in finished:
                    # kernels/num_stages unused for conventional; pass placeholders
                    pending.append((subject, (3, 5, 11), window_size, 1, "conventional"))

    return pending


def _total_runs(cfg: dict, model_type: str) -> int:
    n_ms = (len(cfg["subjects"]) * len(cfg["conv_config"])
            * len(cfg["window_sizes"]) * len(cfg["num_stages"]))
    n_cv = len(cfg["subjects"]) * len(cfg["window_sizes"])
    if model_type == "multiscale":
        return n_ms
    if model_type == "conventional":
        return n_cv
    return n_ms + n_cv   # "both"


def cmd_train(cfg: dict, model_type: str, n_workers: int | None = None,
              split_mode: str = "repetition") -> None:
    devices = detect_gpus()
    n_workers = min(n_workers, len(devices)) if n_workers else len(devices)
    print(f"Available devices : {devices}")

    mlflow.set_tracking_uri(cfg["mlruns_dir"])
    mlflow.set_experiment(cfg["experiment"])

    finished = get_finished_runs(cfg["mlruns_dir"], cfg["experiment"])
    pending  = _build_pending(cfg, model_type, finished)
    total    = _total_runs(cfg, model_type)

    print(f"Total runs        : {total}")
    print(f"Already done      : {total - len(pending)}  (skipped)")
    print(f"Remaining         : {len(pending)}  across {n_workers} worker(s)")

    if not pending:
        print("Nothing to do.")
        return

    # Build full task tuples (device assigned round-robin)
    tasks = [
        (
            cfg["db"], p[0], p[1], p[2], p[3],
            devices[idx % n_workers],
            str(cfg["ninapro_dir"]),
            cfg["train_reps"], cfg["test_reps"], cfg["window_step"],
            cfg["epochs"], cfg["lr"], cfg["weight_decay"], cfg["batch_size"],
            cfg["fs"], cfg["mlruns_dir"], cfg["experiment"], p[4], split_mode,
        )
        for idx, p in enumerate(pending)
    ]

    if n_workers == 1:
        for i, t in enumerate(tasks, 1):
            subject, kernels, window_size, num_stages, mtype, smode = \
                t[1], t[2], t[3], t[4], t[17], t[18]
            extra = f"  kernels={kernels}  stages={num_stages}" if mtype == "multiscale" else ""
            print(f"[{i}/{len(tasks)}] subject={subject}  model={mtype}  "
                  f"split={smode}  window={window_size}{extra}  device={t[5]}")
            r = run_task(t)
            print(f"  → acc={r['final_acc']:.4f}")
    else:
        # Multi-GPU: fork workers BEFORE any CUDA is initialised → safe.
        # Limit BLAS/OpenMP threads to 1 before fork: pywt and numpy use
        # multi-threaded libraries whose internal threads cannot survive a fork
        # (the child inherits a deadlocked state and gets killed by the OS).
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(run_task, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                t = futures[fut]
                try:
                    r = fut.result()
                    print(f"[{i}/{len(tasks)}] done  "
                          f"subject={r['subject']}  model={r['model_type']}  "
                          f"acc={r['final_acc']:.4f}")
                except Exception as e:
                    print(f"[{i}/{len(tasks)}] ERROR  task={t[:5]}  {e}")

    print("All runs complete.")


# ===========================================================================
# Results mode
# ===========================================================================

def cmd_results(cfg: dict, plot: bool = False, save_plot: str | None = None) -> None:
    try:
        import pandas as pd
    except ImportError:
        sys.exit("pandas is required for the results command: pip install pandas")

    mlflow.set_tracking_uri(cfg["mlruns_dir"])
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(cfg["experiment"])
    if exp is None:
        sys.exit(f"MLflow experiment '{cfg['experiment']}' not found.")

    all_runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
    )
    if not all_runs:
        print("No finished runs found.")
        return

    records = []
    for r in all_runs:
        p = r.data.params
        mtype = p.get("model_type", "multiscale")
        records.append({
            "model_type":  mtype,
            "subject":     int(p.get("subject", -1)),
            "kernels":     p.get("kernels", "") if mtype == "multiscale" else "—",
            "window_size": int(p.get("window_size", -1)),
            "num_stages":  int(p.get("num_stages", -1)) if mtype == "multiscale" else 0,
            "final_acc":   r.data.metrics.get("final_test_acc", float("nan")),
            "n_params":    int(p.get("n_params", 0)),
            "run_name":    r.info.run_name,
        })

    df = pd.DataFrame(records).sort_values(
        ["model_type", "subject", "kernels", "window_size", "num_stages"]
    )
    print(f"Loaded {len(df)} finished runs\n")
    print(df.to_string(index=False))

    pivot = (
        df.groupby(["model_type", "kernels", "window_size", "num_stages"])["final_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    print("\nRanked by mean accuracy across subjects:")
    print(pivot.to_string(index=False))

    if plot or save_plot:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(14, 5))
        labels_bar = [
            (f"{r['model_type']}\nk={r['kernels']} s={r['num_stages']}"
             if r["model_type"] == "multiscale"
             else f"conventional\nw={r['window_size']}")
            + f"\nw={r['window_size']}"
            for _, r in pivot.iterrows()
        ]
        ax.bar(range(len(pivot)), pivot["mean"], yerr=pivot["std"], capsize=4)
        ax.set_xticks(range(len(pivot)))
        ax.set_xticklabels(labels_bar, fontsize=7)
        ax.set_ylabel("Mean Test Accuracy")
        ax.set_title("Hyperparameter comparison (mean ± std over subjects)")
        plt.tight_layout()
        if save_plot:
            plt.savefig(save_plot, dpi=150)
            print(f"Plot saved to {save_plot}")
        if plot:
            plt.show()

    print(f"\nMLflow UI:  mlflow ui --port 5000  (experiment: '{cfg['experiment']}')")


# ===========================================================================
# Quantize mode
# ===========================================================================

def _build_datasets(cfg: dict, subject: int, window_size: int):
    tmp = load_subject(cfg["db"], subject, cfg["ninapro_dir"])
    tmp_labels = tmp["restimulus"] if tmp["restimulus"].size else tmp["stimulus"]
    tmp_reps = tmp["repetition"].astype(int)
    train_mask = np.isin(tmp_reps, cfg["train_reps"])
    all_labels = sorted(int(c) for c in np.unique(tmp_labels[train_mask]) if c > 0)
    lmap = {orig: new for new, orig in enumerate(all_labels)}

    tr = NinaProDatasetByRepetition(
        cfg["db"], subject, cfg["train_reps"],
        window_size=window_size, step=cfg["window_step"],
        label_map=lmap, ninapro_dir=cfg["ninapro_dir"], fs=cfg["fs"],
    )
    te = NinaProDatasetByRepetition(
        cfg["db"], subject, cfg["test_reps"],
        window_size=window_size, step=cfg["window_step"],
        label_map=lmap, ninapro_dir=cfg["ninapro_dir"], fs=cfg["fs"],
        channel_stats=tr.channel_stats,
    )
    return tr, te, lmap


def _evaluate_cpu(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x).argmax(1))
            trues.append(y)
    return accuracy_score(
        torch.cat(trues).numpy(), torch.cat(preds).numpy()
    )


def _load_fp32_model(cfg: dict, subject: int, window_size: int,
                     kernels: tuple, num_stages: int,
                     model_type: str = "multiscale") -> nn.Module:
    name = run_name_for(cfg["db"], subject, window_size, kernels, num_stages, model_type)
    mlflow.set_tracking_uri(cfg["mlruns_dir"])
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(cfg["experiment"])
    if exp is None:
        raise RuntimeError(f"Experiment '{cfg['experiment']}' not found.")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"attributes.run_name = '{name}' "
                      "and attributes.status = 'FINISHED'",
    )
    if not runs:
        raise RuntimeError(f"No finished run '{name}'. Train it first.")
    run_id = runs[0].info.run_id
    uri = f"{cfg['mlruns_dir']}/{exp.experiment_id}/{run_id}/artifacts/model"
    return mlflow.pytorch.load_model(uri, map_location="cpu").eval()


def _model_size_kb(model: nn.Module) -> float:
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size = os.path.getsize(f.name) / 1024
    os.unlink(f.name)
    return size


def _throughput_ms(model: nn.Module, loader: DataLoader, n_batches: int = 20) -> float:
    model.eval()
    times = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
            if i + 1 >= n_batches:
                break
    return float(np.mean(times))


def quantize_ptq(fp32_model: nn.Module, calib_loader: DataLoader,
                 n_calib_batches: int) -> nn.Module:
    import torch.ao.quantization as tq
    model = copy.deepcopy(fp32_model).cpu().eval()
    for _, module in model.named_modules():
        if isinstance(module, ConvBnRelu):
            tq.fuse_modules(module, [["0", "1", "2"]], inplace=True)
    model.qconfig = tq.get_default_qconfig("x86")
    tq.prepare(model, inplace=True)
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            model(x)
            if i + 1 >= n_calib_batches:
                break
    tq.convert(model, inplace=True)
    return model


def quantize_qat(fp32_model: nn.Module, train_loader: DataLoader,
                 test_loader: DataLoader, epochs: int, lr: float) -> nn.Module:
    import torch.ao.quantization as tq
    model = copy.deepcopy(fp32_model).cpu().train()
    for _, module in model.named_modules():
        if isinstance(module, ConvBnRelu):
            tq.fuse_modules(module, [["0", "1", "2"]], inplace=True)
    model.qconfig = tq.get_default_qat_qconfig("x86")
    tq.prepare_qat(model, inplace=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(1, epochs + 1):
        model.train()
        correct = n = 0
        for x, y in train_loader:
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
        scheduler.step()
        if epoch == max(1, int(epochs * 0.6)):
            model.apply(tq.disable_observer)
        if epoch == max(1, int(epochs * 0.8)):
            model.apply(nn.intrinsic.qat.freeze_bn_stats)
        val_acc = _evaluate_cpu(model, test_loader)
        print(f"  QAT epoch {epoch:3d}/{epochs}  train={correct/n:.3f}  val={val_acc:.3f}")

    model.eval()
    tq.convert(model, inplace=True)
    return model


def cmd_quantize(cfg: dict, subject: int, kernels: tuple, window_size: int,
                 num_stages: int, method: str, qat_epochs: int, qat_lr: float,
                 ptq_calib_batches: int, model_type: str = "multiscale") -> None:
    print(f"Loading FP32 model for subject={subject}, model={model_type}, "
          f"window={window_size}" +
          (f", kernels={kernels}, stages={num_stages}" if model_type == "multiscale" else "") +
          " …")
    fp32_model = _load_fp32_model(cfg, subject, window_size, kernels, num_stages, model_type)

    train_ds, test_ds, _ = _build_datasets(cfg, subject, window_size)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, num_workers=0)

    fp32_acc = _evaluate_cpu(fp32_model, test_loader)
    print(f"FP32 baseline accuracy : {fp32_acc:.4f}")

    print(f"\nQuantization method: {method.upper()}")
    if method == "ptq":
        q_model = quantize_ptq(fp32_model, train_loader, ptq_calib_batches)
    elif method == "qat":
        q_model = quantize_qat(fp32_model, train_loader, test_loader,
                               qat_epochs, qat_lr)
    else:
        sys.exit(f"Unknown method '{method}'. Use 'ptq' or 'qat'.")

    int8_acc = _evaluate_cpu(q_model, test_loader)

    fp32_kb = _model_size_kb(fp32_model)
    int8_kb = _model_size_kb(q_model)
    fp32_ms = _throughput_ms(fp32_model, test_loader)
    int8_ms = _throughput_ms(q_model,    test_loader)

    print(f"\n{'':20s}  {'FP32':>10}  {'Int8':>10}  {'Ratio':>8}")
    print("-" * 55)
    print(f"{'Model size (KB)':20s}  {fp32_kb:>10.1f}  {int8_kb:>10.1f}"
          f"  {int8_kb/fp32_kb:>7.2f}x")
    print(f"{'Batch latency (ms)':20s}  {fp32_ms:>10.2f}  {int8_ms:>10.2f}"
          f"  {int8_ms/fp32_ms:>7.2f}x")
    print(f"{'Test accuracy':20s}  {fp32_acc:>10.4f}  {int8_acc:>10.4f}"
          f"  {int8_acc - fp32_acc:>+7.4f}")

    mlflow.set_tracking_uri(cfg["mlruns_dir"])
    mlflow.set_experiment(cfg["experiment"])
    with mlflow.start_run(run_name=f"quant_{method}_s{subject}_w{window_size}"):
        mlflow.log_params({
            "method":       method,
            "base_subject": subject,
            "window_size":  window_size,
            "kernels":      str(kernels),
            "num_stages":   num_stages,
        })
        mlflow.log_metrics({
            "fp32_acc":      fp32_acc,
            "int8_acc":      int8_acc,
            "acc_delta":     int8_acc - fp32_acc,
            "size_ratio":    int8_kb / fp32_kb,
            "latency_ratio": int8_ms / fp32_ms,
        })
        mlflow.pytorch.log_model(q_model, artifact_path="model_int8")
    print("\nQuantized model logged to MLflow.")


# ===========================================================================
# Argument parser
# ===========================================================================

def _kernels(s: str):
    """Parse '3,5,11' → (3, 5, 11)."""
    parts = [int(x) for x in s.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("kernels must be three comma-separated ints, e.g. 3,5,11")
    return tuple(parts)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config",      default="config.json",
                   help="Path to config.json  (default: config.json)")
    p.add_argument("--n-workers", dest="n_workers", type=int, default=None,
                   help="Number of parallel workers (default: one per GPU). "
                        "Reduce if you run out of memory.")
    p.add_argument("--ninapro-dir", dest="ninapro_dir", default=None,
                   help="Override ninapro_dir from config")
    p.add_argument("--mlruns-dir",  dest="mlruns_dir",  default=None,
                   help="Override mlruns_dir from config")
    p.add_argument("--experiment",  default=None,
                   help="Override mlflow_experiment from config")

    # Dataset / search space
    p.add_argument("--db",          type=int, default=2)
    p.add_argument("--subjects",    type=int, nargs="+",
                   default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p.add_argument("--conv-config", dest="conv_config", type=_kernels, nargs="+",
                   default=[(3,5,11), (5,9,19), (7,9,13), (3,11,23)])
    p.add_argument("--window-sizes",dest="window_sizes", type=int, nargs="+",
                   default=[512])
    p.add_argument("--num-stages",  dest="num_stages",  type=int, nargs="+",
                   default=[1])

    # Training hyperparameters
    p.add_argument("--train-reps",  dest="train_reps",  type=int, nargs="+",
                   default=[1, 3, 4, 6],
                   help="Train repetitions (DB2/DB5 default: 1 3 4 6; "
                        "DB1: 1 2 4 6 7 9 10; DB7: 1 2 4 5)")
    p.add_argument("--test-reps",   dest="test_reps",   type=int, nargs="+",
                   default=[2, 5],
                   help="Test repetitions  (DB2/DB5 default: 2 5; "
                        "DB1: 3 5 8; DB7: 3 6)")
    p.add_argument("--window-step", dest="window_step", type=int, default=32)
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--weight-decay",dest="weight_decay", type=float, default=1e-4)
    p.add_argument("--batch-size",  dest="batch_size",   type=int,   default=64)
    p.add_argument("--fs",          type=int,   default=2000,
                   help="Sampling frequency in Hz (default 2000 for DB2-5, use 100 for DB1)")

    sub = p.add_subparsers(dest="command")
    sub.required = True

    # ── train ──────────────────────────────────────────────────────────────
    tp = sub.add_parser("train", help="Run / resume the grid search.")
    tp.add_argument("--model", choices=["multiscale", "conventional", "both"],
                    default="both",
                    help="Which model(s) to train (default: both)")
    tp.add_argument("--split", dest="split_mode",
                    choices=["repetition", "sample"], default="repetition",
                    help="'repetition' (default): train/test split by rep index — "
                         "realistic. 'sample': random 70/30 window-level split "
                         "matching the paper (data leakage, higher accuracy).")

    # ── results ────────────────────────────────────────────────────────────
    rp = sub.add_parser("results", help="Print ranked results from MLflow.")
    rp.add_argument("--plot",      action="store_true",
                    help="Show interactive bar chart")
    rp.add_argument("--save-plot", dest="save_plot", default=None, metavar="FILE",
                    help="Save bar chart to FILE (e.g. results.png)")

    # ── quantize ───────────────────────────────────────────────────────────
    qp = sub.add_parser("quantize", help="Quantize a trained model.")
    qp.add_argument("--subject",    type=int, default=1)
    qp.add_argument("--kernels",    type=_kernels, default=(3, 5, 11),
                    help="Kernel sizes, e.g. 3,5,11")
    qp.add_argument("--window",     type=int, default=256,
                    dest="window_size")
    qp.add_argument("--stages",     type=int, default=1,
                    dest="num_stages")
    qp.add_argument("--method",     choices=["ptq", "qat"], default="ptq")
    qp.add_argument("--qat-epochs", dest="qat_epochs", type=int,   default=10)
    qp.add_argument("--qat-lr",     dest="qat_lr",     type=float, default=1e-4)
    qp.add_argument("--ptq-calib-batches", dest="ptq_calib_batches",
                    type=int, default=32)
    qp.add_argument("--model", choices=["multiscale", "conventional"],
                    default="multiscale",
                    help="Which model to quantize (default: multiscale)")

    return p


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    cfg = build_config(args)

    print(f"  db             = {cfg['db']}")
    print(f"  subjects       = {cfg['subjects']}")
    print(f"  ninapro_dir    = {cfg['ninapro_dir']}")
    print(f"  mlruns_dir     = {cfg['mlruns_dir']}")
    print(f"  experiment     = {cfg['experiment']}")
    print()

    if args.command == "train":
        cmd_train(cfg, model_type=args.model, n_workers=args.n_workers,
                  split_mode=args.split_mode)

    elif args.command == "results":
        cmd_results(cfg, plot=args.plot, save_plot=args.save_plot)

    elif args.command == "quantize":
        cmd_quantize(
            cfg,
            subject=args.subject,
            kernels=args.kernels,
            window_size=args.window_size,
            num_stages=args.num_stages,
            method=args.method,
            qat_epochs=args.qat_epochs,
            qat_lr=args.qat_lr,
            ptq_calib_batches=args.ptq_calib_batches,
            model_type=args.model,
        )


if __name__ == "__main__":
    main()
