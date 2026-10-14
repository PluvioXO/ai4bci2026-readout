"""Validate a complete second-architecture export against saved predictions and source locks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def interval(values):
    values = np.asarray(values, dtype=float)
    assert values.shape == (9,) and np.isfinite(values).all()
    rng = np.random.default_rng(20261012)
    means = values[rng.integers(9, size=(10000, 9))].mean(1)
    return np.array([values.mean(), *np.quantile(means, [.025, .975])])


def audit(root, baseline):
    root, baseline = Path(root), Path(baseline)
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["config"]["subjects"] == list(range(1, 10))
    assert protocol["config"]["seeds"] == list(range(5))
    assert protocol["config"]["synthetic"] is False
    assert len(list(root.glob("subject_*/ordinary/*/training.json"))) == 81
    assert protocol["replication_sha"] == hashlib.sha256((root / "replicate_readout.py").read_bytes()).hexdigest()
    assert protocol["original_engine_sha"] == hashlib.sha256((root / "ai4bci_study.py").read_bytes()).hexdigest()
    locked = json.loads((root / "source_locked.json").read_text())
    assert len(locked) == 81 + 9 + 45 + 45
    for name, digest in locked.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    scores = pd.read_csv(root / "per_seed.csv")
    assert len(scores) == 180 and not scores.duplicated(["subject", "seed", "mode", "readout"]).any()
    computed = []
    checksum_path = root / "data_checksums.json"
    checksums = json.loads(checksum_path.read_text()) if checksum_path.exists() else {}
    for subject in range(1, 10):
        source = pd.read_csv(root / f"subject_{subject:02d}" / "trials_T.csv")
        assert source.trial_id.is_unique and len(source) == 288
        assert source.groupby(["run", "label"]).size().eq(12).all()
        source_reference = pd.read_csv(baseline / f"subject_{subject:02d}" / "source_trials.csv")
        for column in ("trial_id", "label", "artifact", "run", "cue_sample"):
            np.testing.assert_array_equal(source[column], source_reference[column])
        source_meta = json.loads((baseline / f"subject_{subject:02d}" / "selection.json").read_text())
        target_meta = json.loads((baseline / f"subject_{subject:02d}" / "target_data.json").read_text())
        for letter, expected in (("T", source_meta["source_sha256"]), ("E", target_meta["target_sha256"])):
            key = f"A{subject:02d}{letter}.mat"
            prepared = root / f"subject_{subject:02d}" / f"epochs_{letter}.npz"
            if prepared.exists():
                with np.load(prepared, allow_pickle=False) as arrays:
                    checksums[key] = str(arrays["raw_sha"])
            assert checksums[key] == expected
        reference = pd.read_csv(baseline / f"subject_{subject:02d}" / "test_predictions.csv")
        for seed in range(5):
            folder = root / f"subject_{subject:02d}" / "ordinary" / f"seed_{seed}"
            predictions = pd.read_csv(folder / "test_predictions.csv")
            assert len(predictions) == 288 and predictions.trial_id.is_unique
            assert not set(predictions.trial_id) & set(source.trial_id)
            assert predictions.groupby(["run", "label"]).size().eq(12).all()
            for column in ("trial_id", "label", "artifact", "run", "cue_sample"):
                np.testing.assert_array_equal(predictions[column], reference[column])
            heads = np.load(folder / "source_heads.npz", allow_pickle=False)
            choices = json.loads((folder / "source_readout_selection.json").read_text())
            for mode in ("smooth", "raw"):
                assert float(heads[f"{mode}_alpha"]) == choices[mode]["alpha"]
                assert choices[mode]["alpha"] in protocol["config"]["ridge_alphas"]
                assert heads[f"{mode}_weights"].shape == (64, 4)
                assert np.isfinite(heads[f"{mode}_weights"]).all()
                for readout in ("original", "ridge"):
                    pred = predictions[f"{mode}_{readout}"]
                    assert pred.isin(range(4)).all()
                    def ba(frame):
                        return 100 * np.mean([np.mean(pred[frame.index][frame.label == c] == c) for c in range(4)])
                    computed.append(dict(subject=subject, seed=seed, mode=mode, readout=readout,
                                         ba=ba(predictions), artifact_free_ba=ba(predictions[predictions.artifact == 0])))
    recomputed = pd.DataFrame(computed)
    keys = ["subject", "seed", "mode", "readout"]
    joined = scores.merge(recomputed, on=keys, suffixes=("_saved", "_recomputed"), validate="one_to_one")
    for column in ("ba", "artifact_free_ba"):
        np.testing.assert_allclose(joined[f"{column}_saved"], joined[f"{column}_recomputed"], atol=1e-10, rtol=0)
    participants = recomputed.groupby(["subject", "mode", "readout"]).ba.mean().unstack(["mode", "readout"])
    values = {"_".join(key): participants[key].to_numpy() for key in participants}
    values.update(
        raw_ridge_minus_raw_original=values["raw_ridge"]-values["raw_original"],
        smooth_ridge_minus_smooth_original=values["smooth_ridge"]-values["smooth_original"],
        raw_ridge_minus_smooth_ridge=values["raw_ridge"]-values["smooth_ridge"],
        excess_raw_recovery_over_smooth_recovery=(values["raw_ridge"]-values["raw_original"])
                                               -(values["smooth_ridge"]-values["smooth_original"]))
    summary = pd.read_csv(root / "summary.csv").set_index("comparison")
    assert set(summary.index) == set(values)
    for key, value in values.items():
        np.testing.assert_allclose(interval(value), summary.loc[key, ["mean", "low", "high"]], atol=1e-10, rtol=0)
    report = dict(status="passed", participants=9, seeds=5, main_fits=45, tuning_fits=36,
                  prediction_records=45*288, outcome_predictions=45*288*4,
                  source_files_hash_checked=len(locked), score_cells_recomputed=180,
                  bootstrap_intervals_recomputed=len(values),
                  target_trial_ids_labels_and_artifacts_match_classical_reference=True,
                  source_and_target_recording_checksums_match_classical_reference=True,
                  original_engine_sha256=protocol["original_engine_sha"],
                  replication_engine_sha256=protocol["replication_sha"],
                  limitations="Export and source-lock audit; not independent retraining or full forward replay. "
                              "Same dataset and participants; ordinary training only.")
    checksum_path.write_text(json.dumps(checksums, indent=2) + "\n")
    (root / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--baseline", default="results/fbcsp_baseline")
    args = p.parse_args()
    audit(args.run, args.baseline)
