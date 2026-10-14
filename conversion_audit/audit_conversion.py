"""Read-only conversion audit of downloaded AI4BCI outputs; never retrains a model.

Usage: .venv/bin/python SAM/ai4bci2026/conversion_audit/audit_conversion.py INPUT_DIR
INPUT_DIR contains the archived engine, aggregate CSVs, participant-1 epochs/scaler,
and sast/, matched/, direct/ folders with seed-0 weights and analysis exports.
Source-moment alignment below is an exploratory diagnostic, not a selected method.
"""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


inputs = Path(sys.argv[1])
out = Path(__file__).resolve().parent
project = out.parent
engine_path = project / "code/ai4bci_study.py"
if not engine_path.exists():
    engine_path = out.parents[2] / "src/ai4bci_study.py"
archived = inputs / "ai4bci_study.py"
expected_hash = "42ba021e26724e3e0d18ec77d1df765cc438092cf487c2eb1b94b8d5a56aacde"
assert hashlib.sha256(archived.read_bytes()).hexdigest() == expected_hash
assert engine_path.read_bytes() == archived.read_bytes()
notebook = json.loads((out.parent / "ai4bci_all_experiments.ipynb").read_text())
embedded = None
for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "ENGINE_SOURCE" for t in node.targets
            ):
                embedded = ast.literal_eval(node.value)
assert embedded == archived.read_text()
spec = importlib.util.spec_from_file_location("audited_bci", archived)
b = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = b
spec.loader.exec_module(b)
torch.set_num_threads(2)
cfg = b.BCIConfig()

e1 = pd.read_csv(inputs / "e1_per_seed.csv")
e2 = pd.read_csv(inputs / "e2_per_seed.csv")
conf = pd.read_csv(inputs / "confusion_counts.csv").query("experiment == 'E1'")
keys = ["subject", "method", "seed", "mode"]
assert len(e1) == 675 and len(e2) == 1350
assert not e1.duplicated(keys).any()
counts = []
for key, group in conf.groupby(keys):
    cm = group.pivot(index="true", columns="predicted", values="count").to_numpy()
    assert cm.shape == (4, 4) and (cm.sum(1) == 72).all()
    counts.append(dict(zip(keys, key), recomputed_ba=100 * np.mean(cm.diagonal() / cm.sum(1)),
                       dominant_fraction=cm.sum(0).max() / cm.sum(),
                       recomputed_classes=int((cm.sum(0) > 0).sum())))
verified = e1.merge(pd.DataFrame(counts), on=keys, validate="one_to_one")
np.testing.assert_allclose(verified.ba, verified.recomputed_ba, atol=1e-12, rtol=0)
assert (verified.prediction_classes == verified.recomputed_classes).all()
verified.to_csv(out / "all_run_checks.csv", index=False)
aggregates = verified.groupby(["method", "mode"]).agg(
    accuracy=("ba", "mean"), runs=("ba", "size"),
    single_class_runs=("prediction_classes", lambda s: int((s == 1).sum())),
    dominant_fraction=("dominant_fraction", "mean"))
aggregates.to_csv(out / "collapse_summary.csv")
e2.query("layer == 2").groupby(["method", "mode"])[
    ["probe_ba", "recovery_pp", "cka"]].mean().to_csv(out / "probe_summary.csv")

# Check the recurrence against an independent scalar implementation, including
# exact threshold ties, silent input, and negative input.
currents = np.array([0., 1., 0., -1., 2., .5, 0., 0.])
for mode in (0, 1, 2):
    for threshold in (.5, 1., 3.):
        voltage, previous, reference = 0., 0., []
        for current in currents:
            voltage = .9 * voltage + current - threshold * previous
            previous = (.5 + np.arctan(voltage - threshold) / np.pi) if mode == 0 else float(voltage >= threshold)
            reference.append(previous)
        actual = b.lif_sequence(torch.tensor(currents).reshape(1, 1, -1), .9, threshold, mode)
        np.testing.assert_allclose(actual.numpy().ravel(), reference, rtol=0, atol=1e-12)

train_frame = pd.read_csv(inputs / "trials_T.csv")
test_frame = pd.read_csv(inputs / "trials_E.csv")
assert train_frame.trial_id.is_unique and test_frame.trial_id.is_unique
assert not set(train_frame.trial_id) & set(test_frame.trial_id)
for frame in (train_frame, test_frame):
    assert len(frame) == 288 and (frame.groupby(["run", "label"]).size() == 12).all()
    assert (frame.cue_sample - frame.trial_start_sample == 500).all()
train_mask = train_frame.run.to_numpy() <= 5
yt = train_frame.label.to_numpy()[train_mask]
yv = train_frame.label.to_numpy()[~train_mask]
ye = test_frame.label.to_numpy()
scaler = np.load(inputs / "source_scaler.npz", allow_pickle=False)
raw_t = np.load(inputs / "epochs_T.npz", allow_pickle=False)["x"]
raw_e = np.load(inputs / "epochs_E.npz", allow_pickle=False)["x"]
np.testing.assert_array_equal(scaler["mean"], raw_t[train_mask].mean((0, 2), keepdims=True, dtype=np.float64).astype(np.float32))
np.testing.assert_array_equal(scaler["scale"], np.maximum(raw_t[train_mask].std((0, 2), keepdims=True, dtype=np.float64).astype(np.float32), 1e-6))
xe = ((raw_e - scaler["mean"]) / scaler["scale"]).astype(np.float32)
xv = ((raw_t[~train_mask] - scaler["mean"]) / scaler["scale"]).astype(np.float32)
assert xe.shape == (288, 22, 500) and np.isfinite(xe).all()

replay, diagnostics, probe_checks, grid_checks = [], [], [], []
for method in ("sast", "matched", "direct"):
    folder = inputs / method
    model = b.make_model(cfg, method, 1, 0, torch.device("cpu"))
    model.load_state_dict(torch.load(folder / "best.pt", map_location="cpu", weights_only=True))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    saved = np.load(folder / "features.npz", allow_pickle=False)
    predictions = pd.read_csv(folder / "predictions.csv.gz")
    cal = json.loads((folder / "calibration.json").read_text())
    modes = {"raw": (1, 1.), "calibrated": (1, cal["threshold"])}
    if method != "direct":
        modes["continuous"] = (0, 1.)
    val_predictions = {}
    for threshold in cfg.thresholds:
        logits, _ = b.infer(model, xv, torch.device("cpu"), 1, threshold)
        val_predictions[threshold] = logits.argmax(1)
        accuracy = b.balanced_accuracy(yv, logits.argmax(1))
        expected = cal["validation_scores"][str(threshold)]
        assert abs(accuracy - expected) < 1e-10
        grid_checks.append(dict(method=method, threshold=threshold, validation_ba=accuracy))
    assert b.select_threshold(val_predictions, yv, cfg.thresholds) == cal["threshold"]
    for mode, (flag, threshold) in modes.items():
        fs = saved[f"{mode}_test_L2"]
        with torch.no_grad():
            head_logits = model.head(torch.from_numpy(fs)).numpy()
        observed = predictions.query("experiment == 'E1' and mode == @mode").set_index("trial_id").loc[test_frame.trial_id]
        np.testing.assert_array_equal(observed.label, ye)
        np.testing.assert_array_equal(head_logits.argmax(1), observed.prediction)
        logits, features = b.infer(model, xe, torch.device("cpu"), flag, threshold)
        mismatches = int(np.sum(logits.argmax(1) != observed.prediction.to_numpy()))
        assert mismatches == 0
        max_diff = max(float(np.max(np.abs(features[j] - saved[f"{mode}_test_L{j+1}"]))) for j in range(2))
        replay.append(dict(method=method, mode=mode, subject=1, seed=0,
                           ba=b.balanced_accuracy(ye, logits.argmax(1)), prediction_mismatches=mismatches,
                           feature_max_abs_difference=max_diff, L2_mean=float(fs.mean()),
                           L2_mean_trial_std=float(fs.std(0).mean()), L2_zero_fraction=float((fs == 0).mean())))
        # Repeat/rebatch invariance catches retained state or batch-dependent inference.
        small, _ = b.infer(model, xe[:8], torch.device("cpu"), flag, threshold, batch=1)
        np.testing.assert_allclose(small, logits[:8], atol=2e-5, rtol=2e-5)
        for layer in (1, 2):
            ft, fv, fe = [saved[f"{mode}_{split}_L{layer}"] for split in ("train", "val", "test")]
            head = b.select_ridge(ft, yt, fv, yv, cfg.ridge_alphas)
            ba = b.balanced_accuracy(ye, b.ridge_predict(head, fe))
            reference = e2.query("subject == 1 and seed == 0 and method == @method and mode == @mode and layer == @layer").iloc[0]
            assert abs(ba - reference.probe_ba) < 1e-10 and head["alpha"] == reference.alpha
            probe_checks.append(dict(method=method, mode=mode, layer=layer, ba=ba, alpha=head["alpha"]))
    for k, value in model.state_dict().items():
        assert torch.equal(value, before[k])
    if method != "direct":
        st, rt = saved["continuous_train_L2"], saved["raw_train_L2"]
        raw = saved["raw_test_L2"]
        shifted = raw - rt.mean(0) + st.mean(0)
        affine = (raw - rt.mean(0)) * (st.std(0) / np.maximum(rt.std(0), 1e-6)) + st.mean(0)
        for name, representation in [("raw", raw), ("source_mean_alignment", shifted), ("source_mean_and_scale_alignment", affine)]:
            with torch.no_grad():
                logits = model.head(torch.from_numpy(representation)).numpy()
            diagnostics.append(dict(method=method, subject=1, seed=0, diagnostic=name,
                                    ba=b.balanced_accuracy(ye, logits.argmax(1)),
                                    prediction_classes=int(len(np.unique(logits.argmax(1))))))

pd.DataFrame(replay).to_csv(out / "checkpoint_replay.csv", index=False)
pd.DataFrame(diagnostics).to_csv(out / "exploratory_alignment.csv", index=False)
pd.DataFrame(probe_checks).to_csv(out / "probe_replay.csv", index=False)
pd.DataFrame(grid_checks).to_csv(out / "threshold_replay.csv", index=False)
report = dict(engine_sha256=expected_hash, source_and_notebook_match=True,
              all_E1_cells_recomputed=len(verified), recurrence_reference_passed=True,
              source_only_scaler_verified=True, weights_unchanged=True,
              replay_scope="Participant 1, seed 0; SAST, compute-matched ordinary, direct-hard; CPU",
              replay_environment=b.environment(),
              archived_environment={"torch": "2.11.0+cu130", "python": "3.13.15"},
              all_sampled_predictions_match=True, replays=replay,
              exploratory_alignment=diagnostics,
              limitation="No retraining; full forward replay is a three-checkpoint sample. Source-moment alignment is post-hoc and not a confirmatory result.")
(out / "audit_results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps(report, indent=2, allow_nan=False))
