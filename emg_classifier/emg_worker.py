"""
emg_worker.py – spawn-safe worker for multi-GPU grid search.

All code that runs inside subprocess workers must live here so that
`spawn` can import it without re-executing the notebook.
"""

import glob
from collections import Counter
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_subject(db: int, subject: int, ninapro_dir: Path) -> dict:
    subject_dir = Path(ninapro_dir) / f"DB{db}" / f"DB{db}_s{subject}"
    patterns = [
        str(subject_dir / f"S{subject}_E*_A*.mat"),
        str(subject_dir / f"s{subject}_E*_A*.mat"),
        str(subject_dir / f"DB{db}_s{subject}_E*_A*.mat"),
        str(subject_dir / f"DB{db}_S{subject}_E*_A*.mat"),
    ]
    mat_files = []
    for p in patterns:
        mat_files.extend(sorted(glob.glob(p)))

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found for DB{db} subject {subject} in {subject_dir}.\n"
            f"Run: python get_ninapro.py --db {db} --subjects {subject}"
        )

    arrays: dict = {"emg": [], "stimulus": [], "restimulus": [], "repetition": []}
    for path in mat_files:
        mat = sio.loadmat(path, squeeze_me=True)
        for key in arrays:
            if key in mat:
                arrays[key].append(np.atleast_1d(mat[key]))

    result = {k: np.concatenate(v) if v else np.array([]) for k, v in arrays.items()}
    lengths = [a.shape[0] for a in result.values() if a.size > 0]
    if lengths:
        min_len = min(lengths)
        result = {k: a[:min_len] if a.size > 0 else a for k, a in result.items()}
    return result


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def bandpass_filter(
    emg: np.ndarray, fs: int = 2000, low: float = 10.0, high: float = 500.0
) -> np.ndarray:
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, emg, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NinaProDatasetByRepetition(Dataset):
    def __init__(
        self,
        db,
        subject,
        reps,
        window_size,
        step=32,
        drop_rest=True,
        ninapro_dir=None,
        label_map=None,
        channel_stats=None,
        fs=2000,
    ):
        self.window_size = window_size
        self.label_map = label_map

        raw = load_subject(db, subject, ninapro_dir)
        emg = bandpass_filter(raw["emg"].astype(np.float32), fs=fs)
        labels = raw["restimulus"] if raw["restimulus"].size else raw["stimulus"]
        repetition = raw["repetition"].astype(int)
        min_len = min(len(emg), len(labels), len(repetition))
        emg = emg[:min_len]
        labels = labels[:min_len]
        repetition = repetition[:min_len]

        mask = np.isin(repetition, reps)
        emg, labels = emg[mask], labels[mask]

        if channel_stats is None:
            self.mean = emg.mean(axis=0, keepdims=True)
            self.std = emg.std(axis=0, keepdims=True).clip(min=1e-8)
        else:
            self.mean, self.std = channel_stats
        emg = (emg - self.mean) / self.std

        self._emg = emg
        self._windows = []
        for s in range(0, len(emg) - window_size + 1, step):
            majority = Counter(labels[s : s + window_size].tolist()).most_common(1)[0][0]
            if drop_rest and majority == 0:
                continue
            self._windows.append((s, majority))

    @property
    def channel_stats(self):
        return self.mean, self.std

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        start, majority = self._windows[idx]
        window = self._emg[start : start + self.window_size]
        label = self.label_map[majority] if self.label_map else majority
        return torch.from_numpy(window.T), torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel):
        super().__init__(
            nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )


class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernels: tuple, dropout: float = 0.0):
        super().__init__()
        branch_ch = out_ch // 3
        remainder = out_ch - branch_ch * 3
        self.branches = nn.ModuleList([
            ConvBnRelu(in_ch, branch_ch + (1 if i < remainder else 0), k)
            for i, k in enumerate(kernels)
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([b(x) for b in self.branches], dim=1))


class MultiScaleEMGNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        num_classes: int,
        kernels: tuple = (3, 7, 15),
        num_stages: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        base = 192

        stages = []
        in_ch = n_channels
        for i in range(num_stages):
            stages.append(MultiScaleBlock(
                in_ch, base, kernels,
                dropout=dropout if i == num_stages - 1 else 0.0,
            ))
            if i < num_stages - 1:
                stages.append(nn.MaxPool1d(2))
            in_ch = base
        self.backbone = nn.Sequential(*stages)

        self.refine = nn.Sequential(
            ConvBnRelu(base, 256, 3),
            nn.MaxPool1d(2),
            ConvBnRelu(256, 256, 3),
            nn.Dropout(dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.refine(x)
        return self.classifier(self.gap(x).squeeze(-1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_evaluate(
    db: int,
    subject: int,
    window_size: int,
    kernels: tuple,
    num_stages: int,
    device: torch.device,
    ninapro_dir: Path,
    train_reps: list,
    test_reps: list,
    window_step: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    fs: int,
    mlruns_dir: str,
    experiment: str,
) -> dict:
    run_name = (
        f"db{db}_s{subject}_w{window_size}"
        f"_k{'_'.join(map(str, kernels))}_stages{num_stages}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "db": db,
            "subject": subject,
            "window_size": window_size,
            "kernels": str(kernels),
            "num_stages": num_stages,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "train_reps": str(train_reps),
            "test_reps": str(test_reps),
        })

        tmp = load_subject(db, subject, ninapro_dir)
        tmp_labels = tmp["restimulus"] if tmp["restimulus"].size else tmp["stimulus"]
        tmp_reps = tmp["repetition"].astype(int)
        train_mask = np.isin(tmp_reps, train_reps)
        all_labels = sorted(int(c) for c in np.unique(tmp_labels[train_mask]) if c > 0)
        label_map = {orig: new for new, orig in enumerate(all_labels)}
        num_classes = len(label_map)

        train_ds = NinaProDatasetByRepetition(
            db, subject, train_reps,
            window_size=window_size, step=window_step,
            label_map=label_map, ninapro_dir=ninapro_dir, fs=fs,
        )
        test_ds = NinaProDatasetByRepetition(
            db, subject, test_reps,
            window_size=window_size, step=window_step,
            label_map=label_map, ninapro_dir=ninapro_dir, fs=fs,
            channel_stats=train_ds.channel_stats,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)

        mlflow.log_params({
            "num_classes": num_classes,
            "train_windows": len(train_ds),
            "test_windows": len(test_ds),
        })

        n_channels = train_ds[0][0].shape[0]
        model = MultiScaleEMGNet(
            n_channels, num_classes, kernels=kernels, num_stages=num_stages
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mlflow.log_param("n_params", n_params)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        history = {"train_acc": [], "test_acc": []}

        for epoch in range(1, epochs + 1):
            for phase, loader, train in [
                ("train", train_loader, True),
                ("test", test_loader, False),
            ]:
                model.train(train)
                correct = n = 0
                with torch.set_grad_enabled(train):
                    for x, y in loader:
                        x, y = x.to(device), y.to(device)
                        logits = model(x)
                        loss = criterion(logits, y)
                        if train:
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()
                        correct += (logits.argmax(1) == y).sum().item()
                        n += len(y)
                acc = correct / n
                history[f"{phase}_acc"].append(acc)
                mlflow.log_metric(f"{phase}_acc", acc, step=epoch)
            scheduler.step()

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for x, y in test_loader:
                all_preds.append(model(x.to(device)).argmax(1).cpu())
                all_true.append(y)
        all_preds = torch.cat(all_preds).numpy()
        all_true = torch.cat(all_true).numpy()
        final_acc = accuracy_score(all_true, all_preds)

        mlflow.log_metric("final_test_acc", final_acc)
        mlflow.pytorch.log_model(model, artifact_path="model")
        print(f"  [{run_name}]  acc={final_acc:.4f}  params={n_params:,}")

    return {
        "subject": subject,
        "window_size": window_size,
        "kernels": kernels,
        "num_stages": num_stages,
        "history": history,
        "preds": all_preds,
        "true": all_true,
        "label_map": label_map,
        "final_acc": final_acc,
    }


# ---------------------------------------------------------------------------
# Spawn-safe top-level worker
# ---------------------------------------------------------------------------

def run_task(args: tuple) -> dict:
    """Entry point for each subprocess worker (must be a top-level function
    in an importable module so that `spawn` can pickle and call it).

    args tuple layout:
        db, subject, kernels, window_size, num_stages, device_str,
        ninapro_dir, train_reps, test_reps, window_step, epochs, lr,
        weight_decay, batch_size, fs, mlruns_dir, experiment
    """
    (
        db, subject, kernels, window_size, num_stages, device_str,
        ninapro_dir, train_reps, test_reps, window_step, epochs, lr,
        weight_decay, batch_size, fs, mlruns_dir, experiment,
    ) = args

    device = torch.device(device_str)
    mlflow.set_tracking_uri(mlruns_dir)
    mlflow.set_experiment(experiment)

    result = train_and_evaluate(
        db=db,
        subject=subject,
        window_size=window_size,
        kernels=kernels,
        num_stages=num_stages,
        device=device,
        ninapro_dir=Path(ninapro_dir),
        train_reps=train_reps,
        test_reps=test_reps,
        window_step=window_step,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        fs=fs,
        mlruns_dir=mlruns_dir,
        experiment=experiment,
    )
    result.pop("model", None)   # not needed cross-process
    return result
