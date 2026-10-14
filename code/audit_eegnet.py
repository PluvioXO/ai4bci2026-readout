"""Audit the archived EEGNet source histories and selection; never tune on test scores.

Usage: python code/audit_eegnet.py ORIGINAL_RUN_DIRECTORY
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def audit(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    study = json.loads((root / "study.json").read_text())
    expected = "42ba021e26724e3e0d18ec77d1df765cc438092cf487c2eb1b94b8d5a56aacde"
    assert study["engine"] == expected
    assert hashlib.sha256((root / "ai4bci_study.py").read_bytes()).hexdigest() == expected
    assert study["config"]["subjects"] == list(range(1, 10))
    assert study["config"]["seeds"] == list(range(5))
    rows, histories, hashes = [], [], {}
    for subject in range(1, 10):
        folder = root / f"subject_{subject:02d}" / "eegnet"
        selection = json.loads((folder / "selection.json").read_text())
        assert len(selection["candidates"]) == 4
        winner = min(selection["candidates"], key=lambda c: (
            -c["validation_ba"], c["validation_loss"], c["candidate"]))
        assert selection["winner"] == winner
        for name in [f"pilot_{i}" for i in range(4)] + [f"seed_{i}" for i in range(5)]:
            run = folder / name
            h = pd.read_csv(run / "history.csv", float_precision="round_trip")
            meta = json.loads((run / "training.json").read_text())
            assert h.epoch.tolist() == list(range(1, 101))
            assert np.isfinite(h.train_loss).all() and (h.train_loss > 0).all()
            assert h.gradient_evaluations.tolist() == list(range(8, 801, 8))
            val = h.dropna(subset=["validation_ba", "validation_loss"])
            assert val.epoch.tolist() == list(range(5, 101, 5))
            assert np.isfinite(val.validation_loss).all()
            chosen = val.sort_values(["validation_ba", "validation_loss", "epoch"],
                                     ascending=[False, True, True]).iloc[0]
            assert int(chosen.epoch) == meta["best_epoch"]
            assert abs(chosen.validation_ba - meta["validation_ba"]) < 1e-10
            assert abs(chosen.validation_loss - meta["validation_loss"]) < 1e-10
            assert meta["gradient_evaluations"] == 800 and meta["method"] == "eegnet"
            if name.startswith("seed_"):
                assert meta["settings"] == winner["settings"]
                assert meta["seed"] == int(name.split("_")[1])
            else:
                candidate = selection["candidates"][int(name.split("_")[1])]
                assert meta["settings"] == candidate["settings"]
                assert abs(meta["validation_ba"] - candidate["validation_ba"]) < 1e-10
                assert abs(meta["validation_loss"] - candidate["validation_loss"]) < 1e-10
                assert meta["seed"] == 1000
            weights = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
            assert all(torch.isfinite(v).all() for v in weights.values())
            for key, bound in (("spatial.weight", 1.), ("head.weight", .25)):
                assert weights[key].flatten(1).norm(dim=1).max() <= bound + 1e-6
            assert all(int(weights[f"bn{i}.num_batches_tracked"]) == meta["best_epoch"] * 8
                       for i in (1, 2, 3))
            row = dict(subject=subject, run=name, main=name.startswith("seed_"),
                       best_epoch=meta["best_epoch"], validation_ba=meta["validation_ba"],
                       validation_loss=meta["validation_loss"],
                       final_validation_ba=float(val.iloc[-1].validation_ba),
                       final_validation_loss=float(val.iloc[-1].validation_loss),
                       minimum_loss_epoch=int(val.loc[val.validation_loss.idxmin()].epoch),
                       first_10_train_loss=float(h.train_loss.iloc[:10].mean()),
                       last_10_train_loss=float(h.train_loss.iloc[-10:].mean()),
                       late_validation_ba_gain=float(val.validation_ba.iloc[-5:].max()
                                                    - val.validation_ba.iloc[:-5].max()),
                       learning_rate=meta["settings"]["lr"], weight_decay=meta["settings"]["wd"])
            rows.append(row)
            histories.append(h.assign(subject=subject, run=name, main=row["main"]))
            for p in [folder / "selection.json", run / "history.csv", run / "training.json", run / "best.pt"]:
                hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    scores = pd.DataFrame(rows)
    scores.to_csv(output / "run_diagnostics.csv", index=False)
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(output / "source_histories.csv", index=False)
    main = scores[scores.main]
    summary = dict(
        run_id="BNCI2014_001_ccebe0c6749374ab", engine_sha256=expected,
        main_fits=len(main), pilot_fits=len(scores) - len(main), epochs_per_fit=100,
        updates_per_fit=800, validation_checkpoints=20,
        selection_recomputed=True, finite_weights_and_max_norm_verified=True,
        batchnorm_updates_verified=True, target_labels_used_for_audit=False,
        main_best_epoch_median=float(main.best_epoch.median()),
        main_best_epoch_at_least_90=int((main.best_epoch >= 90).sum()),
        main_best_epoch_100=int((main.best_epoch == 100).sum()),
        main_minimum_validation_loss_at_100=int((main.minimum_loss_epoch == 100).sum()),
        main_late_validation_ba_improved=int((main.late_validation_ba_gain > 1e-10).sum()),
        main_mean_selected_validation_ba=float(main.validation_ba.mean()),
        main_mean_final_validation_ba=float(main.final_validation_ba.mean()),
        main_median_training_loss_decrease=float((main.first_10_train_loss-main.last_10_train_loss).median()),
        interpretation="Selection and constraint checks passed. Late improvements describe the "
                       "bounded training budget; these histories do not establish convergence or "
                       "that a longer budget would improve target-session performance.",
        limitations="No hyperparameter changes, retraining, or target-dependent selection. "
                    "This independently written EEGNet port uses the common study optimizer and "
                    "four source-validation candidates; it is not a tuned benchmark ceiling.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "input_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    main_history = history[history.main]
    for _, h in main_history.groupby(["subject", "run"]):
        axes[0].plot(h.epoch, h.train_loss, alpha=.12, color="black")
        v = h.dropna(subset=["validation_ba"])
        axes[1].plot(v.epoch, v.validation_ba, alpha=.12, color="black")
    axes[0].plot(main_history.groupby("epoch").train_loss.mean(), color="tab:blue", linewidth=2)
    val = main_history.dropna(subset=["validation_ba"]).groupby("epoch").validation_ba.mean()
    axes[1].plot(val, color="tab:blue", linewidth=2)
    axes[0].set(xlabel="Epoch", ylabel="Training cross-entropy")
    axes[1].set(xlabel="Epoch", ylabel="Source-validation balanced accuracy (%)")
    fig.savefig(output / "source_learning_curves.png", dpi=180)
    plt.close(fig)
    text = ("EEGNet source-validation audit\n\n"
            f"All {len(scores)} fits (45 main, 36 pilot) completed 800 updates and 20 source-validation checks. "
            "Recomputed checkpoint choices, all nine optimizer-grid winners, finite weights, "
            "max-norm constraints, and BatchNorm update counts agree with the archive.\n\n"
            f"Main best epoch: median {summary['main_best_epoch_median']:g}; "
            f"{summary['main_best_epoch_at_least_90']}/45 at epoch 90 or later; "
            f"{summary['main_best_epoch_100']}/45 at epoch 100. "
            f"{summary['main_minimum_validation_loss_at_100']}/45 have their minimum validation loss at epoch 100. "
            f"{summary['main_late_validation_ba_improved']}/45 improve their best validation accuracy in the final 25 epochs.\n\n"
            + summary["interpretation"] + "\n\n" + summary["limitations"] + "\n")
    (output / "audit_report.txt").write_text(text)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original_run")
    p.add_argument("--output", default="results/eegnet_audit")
    args = p.parse_args()
    audit(args.original_run, args.output)
