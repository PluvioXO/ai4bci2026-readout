"""Prepare the three-checkpoint forward replay from the original archive and provider data.

Usage: python code/prepare_conversion_replay.py ORIGINAL_RUN OUTPUT --cache RAW_DATA_CACHE
Prepared EEG epochs are local audit inputs; do not redistribute them with the release.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import ai4bci_study as b


def prepare(original, output, cache):
    original, output = Path(original), Path(output)
    if output.resolve() == original.resolve():
        raise ValueError("Use a separate output directory")
    assert b.sha_file(original / "ai4bci_study.py") == b.sha_file(b.__file__)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original / "ai4bci_study.py", output / "ai4bci_study.py")
    for name in ("e1_per_seed.csv", "e2_per_seed.csv", "confusion_counts.csv"):
        shutil.copy2(original / "reports" / name, output / name)
    subject = original / "subject_01"
    expected = json.loads((subject / "sast/seed_0/analysis/complete.json").read_text())
    for letter, field in (("T", "source_sha"), ("E", "target_sha")):
        path = b.download_session(1, letter, cache)
        assert b.sha_file(path) == expected[field]
        x, frame = b.read_session(path, 1, letter)
        saved = b.pd.read_csv(subject / f"trials_{letter}.csv")
        b.pd.testing.assert_frame_equal(frame, saved)
        frame.to_csv(output / f"trials_{letter}.csv", index=False)
        np.savez_compressed(output / f"epochs_{letter}.npz", x=x)
    shutil.copy2(subject / "source_scaler.npz", output / "source_scaler.npz")
    for method in ("sast", "matched", "direct"):
        destination = output / method
        destination.mkdir(exist_ok=True)
        run = subject / method / "seed_0"
        shutil.copy2(run / "best.pt", destination / "best.pt")
        for name in ("features.npz", "predictions.csv.gz", "calibration.json"):
            shutil.copy2(run / "analysis" / name, destination / name)
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("original_run")
    p.add_argument("output")
    p.add_argument("--cache", default="replication/raw_data")
    args = p.parse_args()
    prepare(args.original_run, args.output, args.cache)
