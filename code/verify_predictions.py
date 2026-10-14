"""Recompute E1/E2/E3 table cells from the original archive's saved predictions.

This checks exported predictions and splits, not independent retraining or forward replay.
Usage: python code/verify_predictions.py path/to/extracted/original_run
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def balanced_accuracy(labels, predictions):
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    if not set(labels) == set(range(4)):
        raise ValueError("Expected all four classes")
    return 100 * np.mean([np.mean(predictions[labels == c] == c) for c in range(4)])


def verify(root, destination):
    root = Path(root)
    exports = sorted(root.glob("subject_*/**/seed_*/analysis/predictions.csv.gz"))
    assert len(exports) == 270
    records = {name: [] for name in ("E1", "E2", "E3")}
    rows = 0
    for path in exports:
        frame = pd.read_csv(path)
        rows += len(frame)
        assert frame.session.eq("E").all()
        assert frame.label.isin(range(4)).all() and frame.prediction.isin(range(4)).all()
        assert frame.correct.eq(frame.label == frame.prediction).all()
        assert (frame.cue_sample - frame.trial_start_sample).eq(500).all()
        keys = ["subject", "seed", "method", "mode", "experiment", "intervention", "budget"]
        for key, group in frame.groupby(keys):
            row = dict(zip(keys, key))
            assert group.trial_id.is_unique
            target_runs = set(range(2, 7)) if row["experiment"] == "E3" else set(range(1, 7))
            assert set(group.run) == target_runs
            assert group.groupby(["run", "label"]).size().eq(12).all()
            row["ba"] = balanced_accuracy(group.label, group.prediction)
            clean = group[group.artifact == 0]
            row["artifact_free_ba"] = balanced_accuracy(clean.label, clean.prediction)
            if row["experiment"] == "E2":
                row["layer"] = int(row["intervention"].split("L")[1])
            records[row["experiment"]].append(row)
    expected_counts = dict(E1=675, E2=1350, E3=2700)
    for experiment, values in records.items():
        actual = pd.DataFrame(values).rename(columns={"ba": "computed_ba", "artifact_free_ba": "computed_artifact_free_ba"})
        reference = pd.read_csv(root / "reports" / f"{experiment.lower()}_per_seed.csv")
        keys = ["subject", "seed", "method", "mode"]
        if experiment == "E2":
            keys += ["layer"]
        elif experiment == "E3":
            keys += ["intervention", "budget"]
        assert len(actual) == len(reference) == expected_counts[experiment]
        joined = actual.merge(reference, on=keys, suffixes=("_computed", "_saved"), validate="one_to_one")
        metric = "probe_ba" if experiment == "E2" else "ba"
        clean_metric = "artifact_free_probe_ba" if experiment == "E2" else "artifact_free_ba"
        np.testing.assert_allclose(joined.computed_ba, joined[metric], atol=1e-10, rtol=0)
        np.testing.assert_allclose(joined.computed_artifact_free_ba, joined[clean_metric], atol=1e-10, rtol=0)
    report = dict(status="passed", original_run="BNCI2014_001_ccebe0c6749374ab",
                  prediction_exports=len(exports), saved_prediction_rows=rows,
                  verified_cells=expected_counts, trial_ids_unique_within_cells=True,
                  fixed_E3_test_runs_verified=True, cue_offset_verified=True,
                  artifact_free_cells_recomputed=True,
                  limitation="Export consistency only; no retraining or full forward replay")
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original_run")
    parser.add_argument("--output", default="results/prediction_audit.json")
    args = parser.parse_args()
    assert balanced_accuracy(np.arange(4), np.arange(4)) == 100
    assert balanced_accuracy(np.arange(4), np.zeros(4)) == 25
    verify(args.original_run, args.output)
