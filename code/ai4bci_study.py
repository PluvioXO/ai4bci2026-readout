"""Self-contained AI4BCI experiment engine, embedded in the deliverable notebook."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt
import torch
from torch import nn
from torch.nn import functional as F

METHODS = ("ordinary", "matched", "sam", "sast", "direct", "eegnet")
LABELS = dict(zip(METHODS, ("Ordinary smooth", "Ordinary, 2x updates", "Same-batch SAM",
                          "Independent SAST", "Direct hard", "EEGNet")))
CHANNELS = "Fz FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4 P1 Pz P2 POz".split()
CLASS_NAMES = ["left hand", "right hand", "feet", "tongue"]
ENGINE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class BCIConfig:
    subjects: tuple = tuple(range(1, 10))
    seeds: tuple = tuple(range(5))
    epochs: int = 100
    batch_size: int = 32
    widths: tuple = (16, 32)
    leak: float = .9
    samples: int = 500
    learning_rates: tuple = (.0003, .001)
    radii: tuple = (.05, .2)
    weight_decays: tuple = (.0001, .001)
    thresholds: tuple = (.25, .5, .75, 1., 1.25, 1.5, 2., 3.)
    ridge_alphas: tuple = (.001, .01, .1, 1., 10., 100., 1000.)
    budgets: tuple = (0, 4, 8, 12)
    pilot_seed: int = 1000
    bootstrap: int = 10000
    synthetic: bool = False

    def validate(self):
        assert self.subjects and len(set(self.subjects)) == len(self.subjects)
        assert set(self.subjects) <= set(range(1, 10))
        assert self.seeds and len(set(self.seeds)) == len(self.seeds)
        assert self.pilot_seed not in self.seeds
        assert self.epochs > 0 and self.batch_size > 0 and self.samples >= 64
        assert len(self.widths) == 2 and min(self.widths) > 0 and 0 < self.leak < 1
        assert len(self.radii) == len(self.weight_decays) and len(self.radii) > 0
        assert all(v > 0 for seq in (self.learning_rates, self.radii, self.weight_decays,
                                     self.thresholds, self.ridge_alphas) for v in seq)
        assert len(set(self.thresholds)) == len(self.thresholds) and 1. in self.thresholds
        assert self.budgets[0] == 0 and tuple(sorted(set(self.budgets))) == self.budgets
        assert self.bootstrap >= 100
        if not self.synthetic:
            assert self.samples == 500 and self.budgets == (0, 4, 8, 12)
        return self


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)


def atomic_torch(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp); os.replace(tmp, path)


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def select_device():
    # CPU fallback is portable; CUDA is the intended full-study execution target.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def environment():
    return {"python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__}


def prepare_root(cfg, output):
    cfg.validate()
    identity = {"config": asdict(cfg), "engine": ENGINE_HASH, "software": environment()}
    name = ("SMOKE_" if cfg.synthetic else "BNCI2014_001_") + digest(identity)[:16]
    root = Path(output) / name; root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "study.json", identity)
    # Store the implementation itself so the output can be audited independently.
    (root / "ai4bci_study.py").write_text(Path(__file__).read_text())
    return root


# Data: the MATLAB trial markers precede the cue by TWO seconds.
def download_session(subject, session, cache):
    name = f"A{subject:02d}{session}.mat"
    path = Path(cache) / name; path.parent.mkdir(parents=True, exist_ok=True)
    record = path.with_suffix(".sha256.json")
    url = "https://lampx.tugraz.at/~bci/database/001-2014/" + name
    if path.exists() and record.exists():
        info = json.loads(record.read_text())
        if sha_file(path) != info["sha256"]:
            raise RuntimeError(f"Corrupt cached download: {path}; move it aside and rerun.")
        return path
    if not path.exists():
        for attempt in range(3):
            tmp = path.with_suffix(".part")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "AI4BCI-reproduction/1"})
                with urllib.request.urlopen(request, timeout=90) as response, tmp.open("wb") as out:
                    expected = response.headers.get("Content-Length")
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                if expected and tmp.stat().st_size != int(expected):
                    raise IOError("Incomplete dataset download")
                loadmat(tmp, variable_names=["data"])
                os.replace(tmp, path); break
            except Exception:
                if attempt == 2: raise
                time.sleep(2 ** attempt)
    loadmat(path, variable_names=["data"])
    atomic_json(record, {"url": url, "sha256": sha_file(path), "bytes": path.stat().st_size})
    return path


def read_session(path, subject, session):
    runs = np.atleast_1d(loadmat(path, struct_as_record=False, squeeze_me=True)["data"])
    arrays, rows = [], []
    sos = butter(4, [4, 38], btype="bandpass", fs=250, output="sos")
    run_id = 0
    for raw_id, run in enumerate(runs):
        markers = np.atleast_1d(run.trial).astype(int)
        if not markers.size: continue  # Initial EOG/baseline recordings are not task runs.
        run_id += 1
        labels = np.atleast_1d(run.y).astype(int) - 1
        flags = np.atleast_1d(run.artifacts).astype(int)
        x = np.asarray(run.X, dtype=np.float64)
        if float(run.fs) != 250 or x.ndim != 2 or x.shape[1] != 25:
            raise ValueError("Unexpected sampling rate or channel count")
        if len(markers) != 48 or len(flags) != 48 or not np.array_equal(np.bincount(labels, minlength=4), [12]*4):
            raise ValueError("Unexpected trial/class/artifact counts; do not silently drop trials")
        if not np.isfinite(x[:, :22]).all():
            raise ValueError(f"Nonfinite EEG in {path}, task run {run_id}; inspect before analysis")
        filtered = sosfiltfilt(sos, x[:, :22], axis=0, padtype="odd")
        for trial, (marker, label, artifact) in enumerate(zip(markers, labels, flags)):
            start = int(marker) - 1 + 500  # MATLAB one-based trial start + 2 s to cue.
            epoch = filtered[start:start + 1000]
            if epoch.shape != (1000, 22): raise ValueError("Incomplete 0--4 s cue epoch")
            epoch = resample_poly(epoch, 1, 2, axis=0, window=("kaiser", 5.), padtype="constant")
            arrays.append(epoch.T.astype(np.float32))
            rows.append(dict(subject=subject, session=session, run=run_id, raw_run=raw_id,
                             trial=trial, trial_start_sample=int(marker)-1, cue_sample=start,
                             label=int(label), artifact=int(artifact),
                             trial_id=f"A{subject:02d}{session}-r{run_id}-t{trial:02d}"))
    if run_id != 6: raise ValueError(f"Expected six task runs; found {run_id}")
    return np.stack(arrays), pd.DataFrame(rows)


def synthetic_session(cfg, subject, session):
    rng = np.random.default_rng(9000 + subject * 2 + (session == "E"))
    rows, xs = [], []
    for run in range(1, 7):
        labels = np.repeat(np.arange(4), 2); rng.shuffle(labels)
        for j, label in enumerate(labels):
            x = rng.normal(0, .2, (22, cfg.samples))
            x[label*4:(label+1)*4] += np.sin(np.arange(cfg.samples) * (.1 + label*.03))
            xs.append(x.astype(np.float32))
            rows.append(dict(subject=subject, session=session, run=run, raw_run=run-1, trial=j,
                             trial_start_sample=j*2000, cue_sample=j*2000+500, label=int(label),
                             artifact=int(j == 0), trial_id=f"SYN-{subject}{session}-r{run}-t{j}"))
    return np.stack(xs), pd.DataFrame(rows)


def prepare_subject(cfg, subject, cache, root, target=False):
    directory = root / f"subject_{subject:02d}"; directory.mkdir(exist_ok=True)
    def session(letter):
        if cfg.synthetic: return (*synthetic_session(cfg, subject, letter), "synthetic")
        path = download_session(subject, letter, cache)
        checksum = sha_file(path)
        prepared = directory / f"epochs_{letter}.npz"
        meta = directory / f"trials_{letter}.csv"
        if prepared.exists() and meta.exists():
            with np.load(prepared, allow_pickle=False) as z:
                if str(z["raw_sha"]) != checksum: raise RuntimeError("Dataset changed within a study")
                return z["x"], pd.read_csv(meta), checksum
        x, frame = read_session(path, subject, letter)
        with prepared.with_suffix(".tmp").open("wb") as f: np.savez_compressed(f, x=x, raw_sha=checksum)
        os.replace(prepared.with_suffix(".tmp"), prepared)
        frame.to_csv(meta, index=False)
        return x, frame, checksum
    x, frame, checksum = session("T")
    train = frame.run.to_numpy() <= 5
    mean = x[train].mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    scale = x[train].std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
    scale = np.maximum(scale, 1e-6)
    x = ((x - mean) / scale).astype(np.float32)
    np.savez(directory / "source_scaler.npz", mean=mean, scale=scale)
    frame.to_csv(directory / "trials_T.csv", index=False)
    data = {"train": (x[train], frame[train].reset_index(drop=True)),
            "val": (x[~train], frame[~train].reset_index(drop=True)), "source_sha": checksum}
    if target:
        xt, ft, target_sha = session("E")
        ft.to_csv(directory / "trials_E.csv", index=False)
        data["test"] = (((xt-mean)/scale).astype(np.float32), ft)
        data["target_sha"] = target_sha
    return data


# Network definitions preserve the existing arctan and subtractive-reset semantics.
@torch.jit.script
def lif_sequence(drive: torch.Tensor, leak: float, threshold: float, mode: int):
    state = torch.zeros_like(drive[:, :, 0]); previous = torch.zeros_like(state)
    outputs = torch.jit.annotate(list[torch.Tensor], [])
    for t in range(drive.size(2)):
        state = leak * state + drive[:, :, t] - threshold * previous
        distance = state - threshold
        smooth = .5 + torch.atan(distance) / math.pi
        if mode == 0:
            previous = smooth
        else:
            hard = (distance >= 0).to(drive.dtype)
            previous = hard + (smooth - smooth.detach()) if mode == 2 else hard
        outputs.append(previous)
    return torch.stack(outputs, dim=2)


class EEGSpiking(nn.Module):
    def __init__(self, cfg):
        super().__init__(); a, b = cfg.widths; self.leak = cfg.leak
        self.temporal = nn.Conv2d(1, 8, (1, 31), padding=(0, 15), bias=False)
        self.spatial = nn.Conv2d(8, a, (22, 1), bias=True)
        self.norm1 = nn.GroupNorm(1, a)
        self.conv2 = nn.Conv1d(a, b, 5, padding=2)
        self.norm2 = nn.GroupNorm(1, b)
        self.head = nn.Linear(b, 4)

    def forward(self, x, mode=0, threshold=1., features=False):
        drive = self.spatial(self.temporal(x.unsqueeze(1))).squeeze(2)
        drive = self.norm1(F.avg_pool1d(drive, 4))
        z1 = lif_sequence(drive, self.leak, threshold, mode)
        z2 = lif_sequence(self.norm2(self.conv2(z1)), self.leak, threshold, mode)
        f1, f2 = z1.mean(2), z2.mean(2)
        logits = self.head(f2)
        return (logits, [f1, f2]) if features else logits


class EEGNet(nn.Module):
    """PyTorch EEGNet-8,2 reference: 32-sample MI kernel, 125 Hz, 500 samples.

    Architectural specification: Lawhern et al., EEGModels.py (BSD-3-Clause).
    This independently written port keeps BN/ELU/dropout/pooling/max-norm;
    uses PyTorch initialization and the common study optimizer/search.
    """
    def __init__(self, cfg):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(8, eps=.001, momentum=.01)
        self.spatial = nn.Conv2d(8, 16, (22, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16, eps=.001, momentum=.01)
        self.depth = nn.Conv2d(16, 16, (1, 16), groups=16, bias=False)
        self.point = nn.Conv2d(16, 16, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(16, eps=.001, momentum=.01)
        self.dropout = nn.Dropout(.5)
        self.head = nn.Linear(16 * (cfg.samples // 32), 4)

    def forward(self, x, mode=0, threshold=1., features=False):
        z = self.bn1(self.temporal(F.pad(x.unsqueeze(1), (15, 16))))
        z = self.dropout(F.avg_pool2d(F.elu(self.bn2(self.spatial(z))), (1, 4)))
        f1 = z.mean((2, 3))
        z = self.point(self.depth(F.pad(z, (7, 8))))
        z = self.dropout(F.avg_pool2d(F.elu(self.bn3(z)), (1, 8)))
        f2 = z.flatten(1); logits = self.head(f2)
        return (logits, [f1, f2]) if features else logits

    @torch.no_grad()
    def constrain(self):
        for p, limit in ((self.spatial.weight, 1.), (self.head.weight, .25)):
            flat = p.flatten(1); norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
            p.mul_((limit / norms).clamp(max=1).reshape(-1, *([1]*(p.ndim-1))))


def make_model(cfg, method, subject, seed, device):
    seed_all(subject * 10000 + seed)
    return (EEGNet(cfg) if method == "eegnet" else EEGSpiking(cfg)).to(device)


def balanced_accuracy(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    if not all(np.any(y == c) for c in range(4)): return float("nan")
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(4)]) * 100)


@torch.no_grad()
def infer(model, x, device, mode=0, threshold=1., batch=64):
    model.eval(); logits, feats = [], [[], []]
    for offset in range(0, len(x), batch):
        tensor = torch.as_tensor(x[offset:offset+batch], device=device)
        out, fs = model(tensor, mode, float(threshold), features=True)
        logits.append(out.cpu().numpy())
        for target, values in zip(feats, fs): target.append(values.cpu().numpy())
    result = np.concatenate(logits)
    if not np.isfinite(result).all(): raise FloatingPointError("Nonfinite evaluation output")
    return result, [np.concatenate(v) for v in feats]


def training_step(model, optimizer, xd, yd, xa=None, ya=None, rho=0., hard=False):
    model.train(); optimizer.zero_grad(set_to_none=True)
    mode = 2 if hard else 0
    def backward(x, y):
        loss = F.cross_entropy(model(x, mode), y)
        if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None]))
        if not torch.isfinite(norm): raise FloatingPointError("Nonfinite gradient")
        return loss, norm
    if rho:
        _, norm = backward(xa, ya)
        originals = [(p, p.detach().clone()) for p in model.parameters()]
        try:
            with torch.no_grad():
                for p, _ in originals:
                    if p.grad is not None: p.add_(p.grad * (rho/(norm+1e-12)))
            optimizer.zero_grad(set_to_none=True)
            loss, _ = backward(xd, yd)
        finally:
            with torch.no_grad():
                for p, old in originals: p.copy_(old)
    else:
        loss, _ = backward(xd, yd)
    optimizer.step()
    if isinstance(model, EEGNet): model.constrain()
    return float(loss.detach())


def candidate_settings(cfg, method):
    if method in ("sam", "sast"):
        return [dict(lr=lr, rho=r, wd=.0001) for lr in cfg.learning_rates for r in cfg.radii]
    return [dict(lr=lr, rho=0., wd=w) for lr in cfg.learning_rates for w in cfg.weight_decays]


def fit_model(cfg, method, subject, seed, settings, data, directory, device, stop_after=None):
    directory.mkdir(parents=True, exist_ok=True)
    epochs = cfg.epochs * (2 if method == "matched" else 1)
    identity = digest(dict(config=asdict(cfg), method=method, subject=subject, seed=seed,
                           settings=settings, source=data["source_sha"], engine=ENGINE_HASH,
                           software=environment(), device=device.type))
    model = make_model(cfg, method, subject, seed, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], weight_decay=settings["wd"])
    descent = torch.Generator().manual_seed(subject*10000 + seed + 100000)
    ascent = torch.Generator().manual_seed(subject*10000 + seed + 200000)
    state_path = directory / "resume.pt"
    start, history, best, elapsed = 0, [], None, 0.
    if state_path.exists():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        if saved["identity"] != identity: raise RuntimeError(f"Resume identity changed: {directory}")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        start, history, best, elapsed = saved["epoch"], saved["history"], saved["best"], saved["seconds"]
        descent.set_state(saved["descent"]); ascent.set_state(saved["ascent"])
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda": torch.cuda.set_rng_state_all(saved["cuda_rng"])
    xt, ft = data["train"]; xv, fv = data["val"]
    x = torch.as_tensor(xt, device=device); y = torch.as_tensor(ft.label.to_numpy(), device=device)
    steps = math.ceil(len(x)/cfg.batch_size)
    validation_every = max(1, math.ceil(epochs/20))
    for epoch in range(start, epochs):
        tick = time.perf_counter(); losses = []
        for step in range(steps):
            progress = (epoch*steps+step)/max(1, epochs*steps-1)
            lr = settings["lr"] * (.01 + .99 * (1+math.cos(math.pi*progress))/2)
            for group in optimizer.param_groups: group["lr"] = lr
            ids = torch.randint(len(x), (cfg.batch_size,), generator=descent).to(device)
            aid = (torch.randint(len(x), (cfg.batch_size,), generator=ascent).to(device)
                   if method == "sast" else ids)
            losses.append(training_step(model, optimizer, x[ids], y[ids], x[aid], y[aid],
                                        settings["rho"], hard=method == "direct"))
        if device.type == "cuda": torch.cuda.synchronize()
        entry = dict(epoch=epoch+1, train_loss=float(np.mean(losses)), validation_ba=None,
                     validation_loss=None, gradient_evaluations=(epoch+1)*steps*(2 if settings["rho"] else 1))
        if (epoch+1) % validation_every == 0 or epoch+1 == epochs:
            logits, _ = infer(model, xv, device, mode=1 if method == "direct" else 0)
            ba = balanced_accuracy(fv.label, logits.argmax(1))
            vl = float(F.cross_entropy(torch.from_numpy(logits), torch.from_numpy(fv.label.to_numpy())))
            entry.update(validation_ba=ba, validation_loss=vl)
            if best is None or (ba, -vl) > (best["ba"], -best["loss"]):
                best = dict(ba=ba, loss=vl, epoch=epoch+1,
                            weights={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
        history.append(entry)
        elapsed += time.perf_counter()-tick  # Training plus validation; excludes checkpoint I/O.
        atomic_torch(state_path, dict(identity=identity, epoch=epoch+1, history=history, best=best,
                     seconds=elapsed, model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                     optimizer=optimizer.state_dict(), descent=descent.get_state(), ascent=ascent.get_state(),
                     rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else []))
        if (epoch+1) % max(1, epochs//5) == 0:
            print(f"  {directory.name}: epoch {epoch+1}/{epochs}, loss {entry['train_loss']:.4f}", flush=True)
        if stop_after is not None and epoch+1 >= stop_after and epoch+1 < epochs:
            return None
    model.load_state_dict(best["weights"])
    pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
    atomic_torch(directory / "best.pt", best["weights"])
    atomic_json(directory / "training.json", dict(settings=settings, best_epoch=best["epoch"],
                validation_ba=best["ba"], validation_loss=best["loss"], seconds=elapsed,
                gradient_evaluations=history[-1]["gradient_evaluations"],
                parameters=sum(p.numel() for p in model.parameters()), seed=seed, method=method))
    return model, best, history, elapsed


def train_all(cfg, cache, root, device):
    # Finish every validation-only search and main run before opening target sessions.
    for subject in cfg.subjects:
        data = prepare_subject(cfg, subject, cache, root)
        for method in METHODS:
            directory = root/f"subject_{subject:02d}"/method
            print(f"SOURCE ONLY: participant {subject}, {method}", flush=True)
            candidates = []
            for j, settings in enumerate(candidate_settings(cfg, method)):
                model, best, history, seconds = fit_model(cfg, method, subject, cfg.pilot_seed,
                    settings, data, directory/f"pilot_{j}", device)
                candidates.append(dict(candidate=j, settings=settings, validation_ba=best["ba"],
                                       validation_loss=best["loss"], seconds=seconds))
                del model
            winner = min(candidates, key=lambda c:(-c["validation_ba"], c["validation_loss"], c["candidate"]))
            atomic_json(directory/"selection.json", dict(winner=winner, candidates=candidates))
            for seed in cfg.seeds:
                print(f"MAIN: participant {subject}, {method}, seed {seed}", flush=True)
                fit_model(cfg, method, subject, seed, winner["settings"], data,
                          directory/f"seed_{seed}", device)
    atomic_json(root/"training_complete.json", {"subjects":list(cfg.subjects), "seeds":list(cfg.seeds)})


def select_threshold(predictions, labels, grid, reference=1.):
    return min(grid, key=lambda t:(-balanced_accuracy(labels, predictions[t]), abs(t-reference), t))


def linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    x = x-x.mean(0); y = y-y.mean(0)
    denominator = np.linalg.norm(x.T@x) * np.linalg.norm(y.T@y)
    return float(np.linalg.norm(x.T@y)**2/denominator) if denominator > 1e-20 else float("nan")


def fit_ridge(x, y, alpha, scaler=None):
    x = np.asarray(x, dtype=np.float64)
    if scaler is None: scaler = (x.mean(0), np.maximum(x.std(0), 1e-6))
    z = (x-scaler[0])/scaler[1]
    zm = z.mean(0); targets = np.eye(4)[np.asarray(y, dtype=int)]; ym = targets.mean(0)
    zc = z-zm
    weights = np.linalg.solve(zc.T@zc + alpha*np.eye(z.shape[1]), zc.T@(targets-ym))
    return dict(weights=weights, intercept=ym-zm@weights, mean=scaler[0], scale=scaler[1], alpha=alpha)


def ridge_predict(head, x):
    return (((x-head["mean"])/head["scale"])@head["weights"]+head["intercept"]).argmax(1)


def select_ridge(xt, yt, xv, yv, grid):
    candidates = [fit_ridge(xt, yt, a) for a in grid]
    return max(candidates, key=lambda h:(balanced_accuracy(yv, ridge_predict(h, xv)), h["alpha"]))


def calibration_indices(frame, k):
    pool = np.flatnonzero(frame.run.to_numpy() == 1)
    ids = np.concatenate([pool[frame.label.to_numpy()[pool] == c][:k] for c in range(4)])
    if len(ids) != 4*k: raise ValueError("Calibration pool has too few trials per class")
    return np.sort(ids)


def evaluate_checkpoint(cfg, model, method, subject, seed, data, directory, device):
    directory.mkdir(exist_ok=True)
    xt, ft = data["train"]; xv, fv = data["val"]; xe, fe = data["test"]
    yt, yv, ye = (f.label.to_numpy() for f in (ft, fv, fe))
    modes, val_grid, test_grid = {}, {}, {}
    if method != "eegnet":
        for threshold in cfg.thresholds:
            val_grid[threshold] = infer(model, xv, device, 1, threshold)[0].argmax(1)
        chosen = select_threshold(val_grid, yv, cfg.thresholds)
        modes.update(raw=(1, 1.), calibrated=(1, chosen))
    else: chosen = None
    if method != "direct": modes["continuous"] = (0, 1.)
    atomic_json(directory/"calibration.json", dict(threshold=chosen,
                validation_scores={str(t):balanced_accuracy(yv,p) for t,p in val_grid.items()}))
    rows1, rows2, rows3, predictions, features, heads = [], [], [], [], {}, {}
    def record(experiment, mode, branch, budget, pred, ids):
        frame = fe.iloc[ids].copy()
        frame["prediction"] = np.asarray(pred)[ids]; frame["correct"] = frame.prediction == frame.label
        frame["experiment"] = experiment; frame["mode"] = mode
        frame["intervention"] = branch; frame["budget"] = budget
        frame["method"] = method; frame["seed"] = seed; predictions.append(frame)
    def scores(pred, ids):
        clean = ids[fe.artifact.to_numpy()[ids] == 0]
        return dict(ba=balanced_accuracy(ye[ids], pred[ids]),
                    artifact_free_ba=balanced_accuracy(ye[clean], pred[clean]),
                    prediction_classes=int(len(np.unique(pred[ids]))))
    all_ids = np.arange(len(fe)); test_ids = np.flatnonzero(fe.run.to_numpy() >= 2)
    for mode, (flag, threshold) in modes.items():
        out_t, f_t = infer(model, xt, device, flag, threshold)
        out_v, f_v = infer(model, xv, device, flag, threshold)
        out_e, f_e = infer(model, xe, device, flag, threshold)
        pred = out_e.argmax(1); features[mode] = (f_t, f_v, f_e)
        rows1.append(dict(subject=subject, seed=seed, method=method, mode=mode,
                          threshold=threshold, **scores(pred, all_ids)))
        record("E1", mode, "original", 0, pred, all_ids)
        for layer in range(2):
            head = select_ridge(f_t[layer], yt, f_v[layer], yv, cfg.ridge_alphas)
            probe_pred = ridge_predict(head, f_e[layer]); heads[(mode, layer)] = head
            rows2.append(dict(subject=subject, seed=seed, method=method, mode=mode,
                layer=layer+1, alpha=head["alpha"], probe_ba=balanced_accuracy(ye,probe_pred),
                original_ba=balanced_accuracy(ye,pred), recovery_pp=balanced_accuracy(ye,probe_pred)-balanced_accuracy(ye,pred),
                artifact_free_probe_ba=balanced_accuracy(ye[fe.artifact==0],probe_pred[fe.artifact==0]), cka=np.nan))
            record("E2", mode, f"probe_L{layer+1}", 0, probe_pred, all_ids)
        if mode != "raw":
            base = heads[(mode, 1)]
            for k in cfg.budgets:
                ids = calibration_indices(fe, k)
                adapted = fit_ridge(np.concatenate([f_t[1], f_e[1][ids]]), np.concatenate([yt, ye[ids]]),
                                    base["alpha"], (base["mean"],base["scale"]))
                ap = ridge_predict(adapted, f_e[1])
                rows3.append(dict(subject=subject, seed=seed, method=method, mode=mode,
                    intervention="readout", budget=4*k, threshold=threshold, **scores(ap,test_ids)))
                record("E3",mode,"readout",4*k,ap,test_ids)
    if "continuous" in features and method != "eegnet":
        for row in rows2:
            row["cka"] = linear_cka(features["continuous"][2][row["layer"]-1],
                                    features[row["mode"]][2][row["layer"]-1])
    if chosen is not None:
        for t in cfg.thresholds: test_grid[t] = infer(model, xe, device, 1, t)[0].argmax(1)
        calibration_manifest = []
        for k in cfg.budgets:
            ids = calibration_indices(fe, k)
            t = chosen if k == 0 else select_threshold({t:p[ids] for t,p in test_grid.items()},ye[ids],cfg.thresholds,chosen)
            pred = test_grid[t]
            rows3.append(dict(subject=subject, seed=seed, method=method, mode="calibrated",
                intervention="threshold",budget=4*k,threshold=t,**scores(pred,test_ids)))
            record("E3","calibrated","threshold",4*k,pred,test_ids)
            calibration_manifest.append(dict(budget=4*k,trial_ids=fe.iloc[ids].trial_id.tolist()))
        atomic_json(directory/"calibration_trials.json", calibration_manifest)
    for name, rows in (("e1",rows1),("e2",rows2),("e3",rows3)):
        pd.DataFrame(rows).to_csv(directory/(name+".csv"),index=False)
    pd.concat(predictions, ignore_index=True).to_csv(directory/"predictions.csv.gz",index=False)
    np.savez_compressed(directory/"features.npz", **{f"{mode}_{split}_L{layer+1}":values[layer]
                        for mode, fs in features.items() for split,values in zip(("train","val","test"),fs) for layer in range(2)})
    np.savez_compressed(directory/"probes.npz", **{f"{mode}_L{layer+1}_{key}":value
                        for (mode,layer),head in heads.items() for key,value in head.items()})
    atomic_json(directory/"complete.json", dict(source_sha=data["source_sha"], target_sha=data["target_sha"]))


def analyze_all(cfg, cache, root, device):
    if not (root/"training_complete.json").exists(): raise RuntimeError("Complete source-only training first")
    for subject in cfg.subjects:
        data = prepare_subject(cfg, subject, cache, root, target=True)
        for method in METHODS:
            for seed in cfg.seeds:
                run = root/f"subject_{subject:02d}"/method/f"seed_{seed}"
                out = run/"analysis"
                if (out/"complete.json").exists():
                    old = json.loads((out/"complete.json").read_text())
                    if old != dict(source_sha=data["source_sha"],target_sha=data["target_sha"]):
                        raise RuntimeError("Data changed since analysis")
                    continue
                model = make_model(cfg, method, subject, seed, device)
                model.load_state_dict(torch.load(run/"best.pt",map_location=device,weights_only=True))
                print(f"E1/E2/E3: participant {subject}, {method}, seed {seed}",flush=True)
                evaluate_checkpoint(cfg,model,method,subject,seed,data,out,device)


def interval(values, cfg):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all(): return (float("nan"),)*3
    rng = np.random.default_rng(20261012)
    means = values[rng.integers(len(values),size=(cfg.bootstrap,len(values)))].mean(1)
    return float(values.mean()), *np.quantile(means,[.025,.975]).tolist()


def summarize(frame, groups, metrics, cfg):
    # Participant is the sampling unit; average all configured seeds FIRST.
    participant = frame.groupby(groups+["subject"],dropna=False)[metrics].agg(
        lambda values: values.mean() if np.isfinite(values).all() else np.nan).reset_index()
    rows=[]
    for keys, part in participant.groupby(groups,dropna=False):
        keys = keys if isinstance(keys,tuple) else (keys,)
        for metric in metrics:
            mean, low, high = interval(part[metric],cfg)
            rows.append(dict(zip(groups,keys),metric=metric,mean=mean,low=low,high=high,
                             n_subjects=len(part),n_finite=int(np.isfinite(part[metric]).sum())))
    return participant,pd.DataFrame(rows)


def report_all(cfg, root):
    reports=root/"reports";reports.mkdir(exist_ok=True)
    parts={key:[] for key in ("e1","e2","e3")}; cost=[]
    for subject,method,seed in itertools.product(cfg.subjects,METHODS,cfg.seeds):
        run=root/f"subject_{subject:02d}"/method/f"seed_{seed}"
        if not (run/"analysis/complete.json").exists(): raise RuntimeError(f"Missing result: {run}")
        for key in parts: parts[key].append(pd.read_csv(run/"analysis"/(key+".csv")))
        cost.append(dict(subject=subject,**json.loads((run/"training.json").read_text())))
    frames={k:pd.concat(v,ignore_index=True) for k,v in parts.items()}
    expected_e1=len(cfg.subjects)*len(cfg.seeds)*15
    if len(frames["e1"]) != expected_e1: raise RuntimeError("Incomplete/duplicate E1 mode matrix")
    if len(frames["e2"]) != 2*expected_e1 or len(frames["e3"]) != len(cfg.budgets)*expected_e1:
        raise RuntimeError("Incomplete E2/E3 matrix; do not report a partial experiment")
    for key, columns in {"e1":["subject","seed","method","mode"],
                         "e2":["subject","seed","method","mode","layer"],
                         "e3":["subject","seed","method","mode","intervention","budget"]}.items():
        if frames[key].duplicated(columns).any(): raise RuntimeError(f"Duplicate {key} results")
    definitions={"e1":(["method","mode"],["ba","artifact_free_ba"]),
                 "e2":(["method","mode","layer"],["probe_ba","original_ba","recovery_pp","cka","artifact_free_probe_ba"]),
                 "e3":(["method","mode","intervention","budget"],["ba","artifact_free_ba"])}
    summaries={}
    for key,frame in frames.items():
        frame.to_csv(reports/(key+"_per_seed.csv"),index=False)
        groups,metrics=definitions[key]
        participants,summary=summarize(frame,groups,metrics,cfg)
        participants.to_csv(reports/(key+"_participants.csv"),index=False)
        summary.to_csv(reports/(key+"_summary.csv"),index=False);summaries[key]=summary
    e1,e2,e3=(frames[k] for k in ("e1","e2","e3"))
    wide=e1.query("mode == 'calibrated'").pivot(index=["subject","seed"],columns="method",values="ba")
    paired=(wide.sast-wide.matched).groupby(level="subject").mean()
    mean,lo,hi=interval(paired.values,cfg)
    signs=np.array(list(itertools.product([-1,1],repeat=len(paired))))
    p=float(np.mean(np.abs((signs*paired.values).mean(1)) >= abs(mean)-1e-12))
    primary=dict(effect_pp=mean,low=lo,high=hi,sign_flip_p=p,n_subjects=len(paired),
                 seeds_per_subject=len(cfg.seeds),synthetic=cfg.synthetic)
    clean_wide=e1.query("mode == 'calibrated'").pivot(index=["subject","seed"],columns="method",values="artifact_free_ba")
    clean_paired=(clean_wide.sast-clean_wide.matched).groupby(level="subject").agg(
        lambda values:values.mean() if np.isfinite(values).all() else np.nan)
    clean_result=interval(clean_paired.values,cfg)
    primary["artifact_free_sensitivity"]={k:(v if np.isfinite(v) else None)
        for k,v in zip(("effect_pp","low","high"),clean_result)}
    clean_paired.rename("sast_minus_matched_pp").to_csv(reports/"artifact_free_primary_participant_effects.csv")
    atomic_json(reports/"primary_result.json",primary)
    paired.rename("sast_minus_matched_pp").to_csv(reports/"primary_participant_effects.csv")
    conversion=e1.pivot(index=["subject","seed","method"],columns="mode",values="ba").reset_index()
    conversion["smooth_minus_raw_pp"]=conversion.continuous-conversion.raw
    conversion["smooth_minus_calibrated_pp"]=conversion.continuous-conversion.calibrated
    conversion=conversion[conversion.method.isin(METHODS[:4])]
    conversion.to_csv(reports/"conversion_gaps_per_seed.csv",index=False)
    _,cg=summarize(conversion,["method"],["smooth_minus_raw_pp","smooth_minus_calibrated_pp"],cfg)
    cg.to_csv(reports/"conversion_gaps_summary.csv",index=False)
    baseline=e3.query("budget == 0")[["subject","seed","method","mode","intervention","ba"]].rename(columns={"ba":"zero_ba"})
    gains=e3.merge(baseline,on=["subject","seed","method","mode","intervention"],validate="many_to_one")
    gains["gain_pp"]=gains.ba-gains.zero_ba
    gains.to_csv(reports/"adaptation_gains_per_seed.csv",index=False)
    _,gs=summarize(gains,["method","mode","intervention","budget"],["gain_pp"],cfg)
    gs.to_csv(reports/"adaptation_gains_summary.csv",index=False)
    pd.DataFrame(cost).to_csv(reports/"training_cost.csv",index=False)
    tuning=[]
    for subject,method in itertools.product(cfg.subjects,METHODS):
        selection=json.loads((root/f"subject_{subject:02d}"/method/"selection.json").read_text())
        tuning.extend(dict(subject=subject,method=method,**c) for c in selection["candidates"])
    pd.DataFrame(tuning).to_csv(reports/"tuning_cost.csv",index=False)
    counts=e1.groupby(["method","mode"]).prediction_classes.agg(["min","max"])
    counts["single_class_runs"]=e1.groupby(["method","mode"]).prediction_classes.apply(lambda s:int((s==1).sum()))
    counts.to_csv(reports/"collapse_counts.csv")
    # Confusion matrices come directly from saved trial-level predictions.
    conf=[]
    for subject,method,seed in itertools.product(cfg.subjects,METHODS,cfg.seeds):
        file=root/f"subject_{subject:02d}"/method/f"seed_{seed}"/"analysis/predictions.csv.gz"
        pr=pd.read_csv(file)
        for keys,g in pr.groupby(["experiment","mode","intervention","budget"]):
            matrix=np.zeros((4,4),dtype=int);np.add.at(matrix,(g.label,g.prediction),1)
            for true,pred in itertools.product(range(4),repeat=2):
                conf.append(dict(subject=subject,method=method,seed=seed,experiment=keys[0],mode=keys[1],
                            intervention=keys[2],budget=keys[3],true=true,predicted=pred,count=matrix[true,pred]))
    pd.DataFrame(conf).to_csv(reports/"confusion_counts.csv",index=False)
    make_figures(cfg,frames,summaries,gs,reports)
    def cell(method,mode):
        row=summaries["e1"].query("method == @method and mode == @mode and metric == 'ba'")
        if row.empty:return "--"
        r=row.iloc[0];return f"{r['mean']:.2f} [{r.low:.2f}, {r.high:.2f}]"
    lines=[r"\begingroup\small\setlength{\tabcolsep}{4pt}",r"\begin{tabular}{lccc}",
           r"\toprule",r"Method & Continuous & Raw spikes & Calibrated spikes \\",r"\midrule"]
    for method in METHODS:
        lines.append(LABELS[method]+" & "+" & ".join(cell(method,m) for m in ("continuous","raw","calibrated"))+r" \\")
    lines += [r"\bottomrule",r"\end{tabular}\endgroup"]
    (reports/"table1.tex").write_text("\n".join(lines)+"\n")
    full = not cfg.synthetic and cfg.subjects==tuple(range(1,10)) and cfg.seeds==tuple(range(5))
    conclusion=("The interval excludes zero in favor of SAST." if lo>0 else
                "The interval excludes zero in favor of matched ordinary training." if hi<0 else
                "The interval includes zero; this does not establish an advantage or equivalence.")
    text=(f"# {'COMPLETE DATASET RUN' if full else 'SMOKE OR REDUCED RUN — NOT THE FULL PAPER RESULT'}\n\n"
          f"Calibrated SAST minus calibrated compute-matched ordinary training: **{mean:+.2f} pp** "
          f"(paired participant bootstrap 95% interval [{lo:+.2f}, {hi:+.2f}]); "
          f"{len(paired)} participants, {len(cfg.seeds)} seeds each. {conclusion}\n\n"
          f"Exact sign-flip sensitivity p={p:.4g}, conditional on sign exchangeability.\n\n"
          "- Table 1: `table1.tex` and `e1_summary.csv` (accuracy and artifact-free checks).\n"
          "- E1 within-checkpoint conversion loss: `conversion_gaps_summary.csv`.\n"
          "- E2: `e2_summary.csv`; L2 calibrated `cka`, `probe_ba`, and `recovery_pp` fill the draft.\n"
          "- E3: `e3_summary.csv` and `adaptation_gains_summary.csv`; budget 48 is the final comparison.\n"
          "- Main two-panel figure: `draft_results.pdf`; all curves: `adaptation_all.pdf`.\n"
          "- Training/search costs and collapsed runs have separate CSVs.\n\n"
          "Do not infer mechanism from CKA alone. Review absolute decoding quality and direct-hard/EEGNet controls. "
          "Ethics/data-use wording, manuscript status, public artifact URL, and the final interpretation require author review. "
          "Raw EEG is not included in the results ZIP. The notebook does not submit or publish anything.\n")
    (reports/"RESULTS_FOR_DRAFT.md").write_text(text)
    atomic_json(root/"complete.json",dict(full_protocol=full,primary=primary))
    return reports


def make_figures(cfg,frames,summaries,gains,reports):
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    e2=frames["e2"].query("layer == 2 and mode == 'calibrated'")
    for j,method in enumerate(METHODS[:4]):
        part=e2[e2.method==method].groupby("subject").recovery_pp.mean()
        mean,lo,hi=interval(part,cfg)
        axes[0].scatter(np.full(len(part),j),part,alpha=.4,s=15)
        axes[0].errorbar(j,mean,yerr=[[mean-lo],[hi-mean]],fmt="ko",capsize=3)
    axes[0].set_xticks(range(4),["Ordinary","2x updates","SAM","SAST"],rotation=15)
    axes[0].axhline(0,color="gray",lw=.8);axes[0].set_ylabel("Spike probe − original readout (pp)")
    axes[0].set_title("E2: Readout recovery")
    es=summaries["e3"].query("metric == 'ba'")
    for method,mode in [(m,"calibrated") for m in METHODS[:5]]+[("eegnet","continuous")]:
        g=es.query("method == @method and mode == @mode and intervention == 'readout'").sort_values("budget")
        axes[1].plot(g.budget,g["mean"],marker="o",label=LABELS[method])
        axes[1].fill_between(g.budget,g.low,g.high,alpha=.09)
    axes[1].set(xlabel="Labeled target trials",ylabel="Balanced accuracy (%)",title="E3: Readout adaptation")
    axes[1].legend(fontsize=7,ncol=2)
    if cfg.synthetic:fig.suptitle("SYNTHETIC PIPELINE CHECK — NOT EEG RESULTS",color="red")
    fig.savefig(reports/"draft_results.pdf");fig.savefig(reports/"draft_results.png",dpi=200);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout="constrained")
    for ax,branch in zip(axes,("threshold","readout")):
        for (method,mode),g in es[es.intervention==branch].groupby(["method","mode"]):
            g=g.sort_values("budget");ax.plot(g.budget,g["mean"],marker="o",label=LABELS[method]+" / "+mode)
        ax.set(title=branch+" adaptation",xlabel="Labeled target trials",ylabel="Balanced accuracy (%)")
        ax.legend(fontsize=6)
    fig.savefig(reports/"adaptation_all.pdf");plt.close(fig)


def export_zip(root):
    destination=root.parent/(root.name+"_results.zip")
    temp=destination.with_suffix(".tmp")
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            # Resume files and cached epochs can be large; best.pt and derived outputs are enough to audit.
            if path.is_file() and path.name != "resume.pt" and not path.name.startswith("epochs_") and path.suffix not in (".tmp",".part"):
                z.write(path,path.relative_to(root))
    os.replace(temp,destination)
    return destination


def run_study(cfg,cache,output,device=None):
    device=torch.device(device) if device else select_device()
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    torch.use_deterministic_algorithms(True)
    if device.type=="cuda":
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    root=prepare_root(cfg,output)
    atomic_json(root/f"session_{time.time_ns()}.json",dict(software=environment(),device=str(device),
        hardware=torch.cuda.get_device_name(device) if device.type=="cuda" else platform.processor()))
    print("Results:",root,flush=True)
    stage="training"
    try:
        train_all(cfg,cache,root,device)
        stage="analysis";analyze_all(cfg,cache,root,device)
        stage="reporting";report_all(cfg,root);archive=export_zip(root)
    except BaseException as error:
        atomic_json(root/f"failure_{time.time_ns()}.json",
                    dict(stage=stage,type=type(error).__name__,message=str(error)))
        raise
    print("Completed:",archive,flush=True)
    return root,archive


def smoke_config():
    return BCIConfig(subjects=(1,2),seeds=(0,),epochs=1,batch_size=16,widths=(2,4),samples=128,
        learning_rates=(.001,),radii=(.1,),weight_decays=(.0001,),thresholds=(.5,1.,2.),
        ridge_alphas=(.1,1.),budgets=(0,1,2),bootstrap=200,synthetic=True)
