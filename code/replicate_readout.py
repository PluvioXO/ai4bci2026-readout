"""Post-study, second-architecture readout replication; source-only selection.

Run from the release root: python code/replicate_readout.py --output replication/runs
The default is nine participants, five main seeds, four tuning candidates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

import ai4bci_study as b


class DepthwiseLIF(nn.Module):
    """One LIF block; depthwise electrode filters and four temporal-bin features."""
    def __init__(self, cfg):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 63), padding=(0, 31), bias=False)
        self.spatial = nn.Conv2d(8, 16, (22, 1), groups=8, bias=True)
        self.norm = nn.GroupNorm(1, 16)
        self.head = nn.Linear(64, 4)
        self.leak = cfg.leak

    def forward(self, x, mode=0, threshold=1., features=False):
        # Linear spatial/temporal filters commute; spatial first avoids 22-channel intermediates.
        mixed = F.conv1d(x, self.spatial.weight.reshape(16, 22, 1))
        kernels = self.temporal.weight.squeeze(2).repeat_interleave(2, dim=0)
        drive = F.conv1d(mixed, kernels, padding=31, groups=16) + self.spatial.bias[None, :, None]
        drive = self.norm(F.avg_pool1d(drive, 8))
        spikes = b.lif_sequence(drive, self.leak, threshold, mode)
        pooled = F.adaptive_avg_pool1d(spikes, 4).flatten(1)
        logits = self.head(pooled)
        return (logits, [spikes.mean(2), pooled]) if features else logits


def make_model(cfg, method, subject, seed, device):
    if method != "ordinary":
        raise ValueError("This replication tests readout recovery under ordinary training only")
    b.seed_all(subject * 10000 + seed)
    return DepthwiseLIF(cfg).to(device)


def configure_engine():
    # Give resumes a distinct identity without changing the archived study code.
    b.ENGINE_HASH = hashlib.sha256(
        (Path(b.__file__).read_bytes() + Path(__file__).read_bytes())).hexdigest()
    b.make_model = make_model


def protocol(cfg):
    return dict(
        analysis="Post-study exploratory replication; design fixed before its target evaluation",
        architecture="single-block depthwise LIF; temporal kernel 63; 8 temporal filters; "
                     "16 depthwise spatial filters; average pool 8; GroupNorm(1); "
                     "LIF leak .9, threshold 1; four temporal-bin averages; linear 64-to-4 head",
        comparison="Original smooth and raw-spike heads versus separately source-fitted ridge heads",
        training="Ordinary AdamW only; original source-validation optimizer grid and checkpoint rule",
        selection="All participant training and source-only ridge selection finish before target loading",
        config=asdict(cfg), original_engine_sha=b.sha_file(b.__file__),
        replication_sha=b.sha_file(__file__), combined_engine_sha=b.ENGINE_HASH,
        software=b.environment(), independent_dataset=False,
        limitations="Same nine participants and dataset; no replication of SAST superiority; "
                    "offline preprocessing and normalization; not preregistered")


def train_source(cfg, subject, cache, root, device, threads):
    configure_engine()
    torch.set_num_threads(threads)
    device = torch.device(device)
    data = b.prepare_subject(cfg, subject, cache, root, target=False)
    assert "test" not in data
    directory = root / f"subject_{subject:02d}" / "ordinary"
    candidates = []
    for j, settings in enumerate(b.candidate_settings(cfg, "ordinary")):
        print(f"SOURCE participant {subject}: pilot {j}", flush=True)
        fitted = b.fit_model(cfg, "ordinary", subject, cfg.pilot_seed, settings,
                             data, directory / f"pilot_{j}", device)
        candidates.append(dict(candidate=j, settings=settings, validation_ba=fitted[1]["ba"],
                               validation_loss=fitted[1]["loss"]))
    winner = min(candidates, key=lambda c: (-c["validation_ba"], c["validation_loss"], c["candidate"]))
    b.atomic_json(directory / "selection.json", dict(winner=winner, candidates=candidates))
    xt, ft = data["train"]
    xv, fv = data["val"]
    for seed in cfg.seeds:
        print(f"SOURCE participant {subject}: main seed {seed}", flush=True)
        folder = directory / f"seed_{seed}"
        model = b.fit_model(cfg, "ordinary", subject, seed, winner["settings"],
                            data, folder, device)[0]
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        heads, choices = {}, {}
        for mode, flag in (("smooth", 0), ("raw", 1)):
            _, train_features = b.infer(model, xt, device, flag)
            _, val_features = b.infer(model, xv, device, flag)
            head = b.select_ridge(train_features[1], ft.label.to_numpy(),
                                  val_features[1], fv.label.to_numpy(), cfg.ridge_alphas)
            choices[mode] = dict(alpha=head["alpha"], source_validation_ba=b.balanced_accuracy(
                fv.label, b.ridge_predict(head, val_features[1])))
            heads.update({f"{mode}_{key}": value for key, value in head.items()})
        np.savez_compressed(folder / "source_heads.npz", **heads)
        b.atomic_json(folder / "source_readout_selection.json", choices)
        assert all(torch.equal(before[k], v.cpu()) for k, v in model.state_dict().items())


def run(cfg, cache, output, device="cpu", workers=1, threads=2):
    cfg.validate()
    if workers < 1 or threads < 1:
        raise ValueError("workers and threads must be positive")
    configure_engine()
    device = torch.device(device)
    identity = protocol(cfg)
    name = ("SMOKE_" if cfg.synthetic else "DEPTHWISE_LIF_") + b.digest(identity)[:16]
    root = Path(output) / name
    root.mkdir(parents=True, exist_ok=True)
    b.atomic_json(root / "protocol.json", identity)
    (root / "ai4bci_study.py").write_bytes(Path(b.__file__).read_bytes())
    (root / "replicate_readout.py").write_bytes(Path(__file__).read_bytes())
    workers = min(workers, len(cfg.subjects))
    if device.type != "cpu" and workers != 1:
        raise ValueError("Use one source-training worker on GPU")
    if workers == 1:
        for subject in cfg.subjects:
            train_source(cfg, subject, cache, root, str(device), threads)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = [pool.submit(train_source, cfg, subject, cache, root, str(device), threads)
                    for subject in cfg.subjects]
            for job in jobs:
                job.result()
    locked = {str(p.relative_to(root)): b.sha_file(p) for p in sorted(root.rglob("*"))
              if p.name in {"best.pt", "selection.json", "source_heads.npz", "source_readout_selection.json"}}
    b.atomic_json(root / "source_locked.json", locked)
    rows = []
    for subject in cfg.subjects:
        data = b.prepare_subject(cfg, subject, cache, root, target=True)
        xe, fe = data["test"]
        for seed in cfg.seeds:
            folder = root / f"subject_{subject:02d}" / "ordinary" / f"seed_{seed}"
            model = make_model(cfg, "ordinary", subject, seed, device)
            model.load_state_dict(torch.load(folder / "best.pt", map_location=device, weights_only=True))
            saved = np.load(folder / "source_heads.npz", allow_pickle=False)
            predictions = fe.copy()
            for mode, flag in (("smooth", 0), ("raw", 1)):
                logits, features = b.infer(model, xe, device, flag)
                head = {k: saved[f"{mode}_{k}"] for k in ("weights", "intercept", "mean", "scale", "alpha")}
                for kind, pred in (("original", logits.argmax(1)),
                                   ("ridge", b.ridge_predict(head, features[1]))):
                    key = f"{mode}_{kind}"
                    predictions[key] = pred
                    clean = fe.artifact.to_numpy() == 0
                    rows.append(dict(subject=subject, seed=seed, mode=mode, readout=kind,
                                     ba=b.balanced_accuracy(fe.label, pred),
                                     artifact_free_ba=b.balanced_accuracy(fe.label[clean], pred[clean])))
            predictions.to_csv(folder / "test_predictions.csv", index=False)
    for path, sha in locked.items():
        assert b.sha_file(root / path) == sha, f"Source fit changed during target evaluation: {path}"
    scores = pd.DataFrame(rows)
    scores.to_csv(root / "per_seed.csv", index=False)
    participants = scores.groupby(["subject", "mode", "readout"]).ba.mean().unstack(["mode", "readout"])
    estimates = []
    def estimate(name, values):
        mean, low, high = b.interval(values, cfg)
        return dict(comparison=name, mean=mean, low=low, high=high)
    for key in participants:
        estimates.append(estimate("_".join(key), participants[key]))
    contrasts = {
        "raw_ridge_minus_raw_original": participants["raw", "ridge"] - participants["raw", "original"],
        "smooth_ridge_minus_smooth_original": participants["smooth", "ridge"] - participants["smooth", "original"],
        "raw_ridge_minus_smooth_ridge": participants["raw", "ridge"] - participants["smooth", "ridge"],
        "excess_raw_recovery_over_smooth_recovery":
            (participants["raw", "ridge"] - participants["raw", "original"])
            - (participants["smooth", "ridge"] - participants["smooth", "original"])}
    for name, values in contrasts.items():
        estimates.append(estimate(name, values))
    pd.DataFrame(estimates).to_csv(root / "summary.csv", index=False)
    participants.columns = ["_".join(c) for c in participants.columns]
    participants.to_csv(root / "participant_scores.csv")
    b.atomic_json(root / "complete.json", dict(participants=len(cfg.subjects), seeds=len(cfg.seeds),
                                               synthetic=cfg.synthetic, source_lock_verified=True,
                                               results_rows=len(scores)))
    print(root, flush=True)
    print(pd.DataFrame(estimates).to_string(index=False), flush=True)
    return root


def self_check():
    cfg = replace(b.BCIConfig(), samples=64, synthetic=True)
    model = DepthwiseLIF(cfg)
    x = torch.randn(8, 22, 64)
    mixed = F.conv1d(x, model.spatial.weight.reshape(16, 22, 1))
    fast = F.conv1d(mixed, model.temporal.weight.squeeze(2).repeat_interleave(2, dim=0),
                    padding=31, groups=16) + model.spatial.bias[None, :, None]
    reference = model.spatial(model.temporal(x.unsqueeze(1))).squeeze(2)
    torch.testing.assert_close(fast, reference, atol=2e-6, rtol=2e-5)
    filters = (model.temporal.weight, model.spatial.weight, model.spatial.bias)
    fast_grads = torch.autograd.grad(fast.square().mean(), filters)
    reference_grads = torch.autograd.grad(reference.square().mean(), filters)
    for actual, expected in zip(fast_grads, reference_grads):
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    logits, features = model(x, features=True)
    assert logits.shape == (8, 4) and features[1].shape == (8, 64)
    logits.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        raw, fs = model(x, mode=1, features=True)
        assert torch.all((fs[1] >= 0) & (fs[1] <= 1))
        assert torch.allclose(raw, model.head(fs[1]))
        assert torch.allclose(raw, torch.cat([model(t[None], mode=1) for t in x]), atol=2e-6)
    print("Second-architecture shape, gradient, binary-feature and rebatching checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="replication/raw_data")
    parser.add_argument("--output", default="replication/runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    self_check()
    cfg = replace(b.smoke_config(), subjects=(1,), widths=(16, 32)) if args.smoke else b.BCIConfig()
    workers = args.workers if args.workers is not None else (3 if args.device == "cpu" else 1)
    run(cfg, args.cache, args.output, args.device, workers=workers, threads=args.threads)
