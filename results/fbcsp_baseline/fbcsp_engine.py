


from pathlib import Path
from functools import partial
from copy import deepcopy
import hashlib
import importlib.metadata
import io
import json
import os
import pickle
import platform
import shutil
import time
import urllib.request
import zipfile

import joblib
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.io import loadmat
from scipy.signal import butter, resample_poly, sosfiltfilt
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

mne.set_log_level("ERROR")
for module, distribution in ((np, "numpy"), (scipy, "scipy"), (pd, "pandas"),
                             (mne, "mne"), (sklearn, "scikit-learn")):
    if module.__version__ != importlib.metadata.version(distribution):
        raise RuntimeError("A loaded package changed during installation. Restart the kernel, then run from section 2.")


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    os.replace(tmp, path)

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

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

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24),
         (24, 28), (28, 32), (32, 36), (34, 38)]
FEATURE_COUNTS = (8, 16, 32, 64)


def band_epochs(x, band):
    if x.ndim != 3 or x.shape[1:] != (22, 500) or not np.isfinite(x).all():
        raise ValueError("Expected finite trial × 22 EEG channels × 500 samples")
    sos = butter(4, band, btype="bandpass", fs=125, output="sos")
    return sosfiltfilt(sos, x.astype(np.float64), axis=-1, padtype="odd")


def fit_spatial(x, y, bands=BANDS):
    if not np.array_equal(np.unique(y), np.arange(4)) or len(x) != len(y):
        raise ValueError("Training data must contain all four classes")
    models, features = [], []
    for band in bands:
        xb = band_epochs(x, band)
        band_models = []
        for label in range(4):
            csp = CSP(n_components=4, reg=0.1, log=True, norm_trace=False,
                      cov_est="concat", component_order="alternate", rank="full")
            features.append(csp.fit_transform(xb, (y == label).astype(int)))
            band_models.append(csp)
        models.append(band_models)
    return np.concatenate(features, axis=1), models


def transform_spatial(x, models, bands=BANDS):
    if len(models) != len(bands):
        raise ValueError("Filter-bank model does not match the prescribed bands")
    features = []
    for band, band_models in zip(bands, models):
        xb = band_epochs(x, band)
        features.extend(csp.transform(xb) for csp in band_models)
    result = np.concatenate(features, axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite CSP features")
    return result


def choose_head(train_features, train_y, val_features, val_y, counts=FEATURE_COUNTS):
    # These source-only fits are identical for every candidate feature count.
    scaler = StandardScaler().fit(train_features)
    train_scaled = scaler.transform(train_features)
    ranking = SelectKBest(
        partial(mutual_info_classif, random_state=20261012), k="all"
    ).fit(train_scaled, train_y)
    candidates, fitted = [], []
    for order, count in enumerate(counts):
        if not 0 < count <= train_features.shape[1]:
            raise ValueError("Selected-feature count exceeds the feature bank")
        selector = deepcopy(ranking).set_params(k=count)
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        classifier.fit(selector.transform(train_scaled), train_y)
        model = make_pipeline(scaler, selector, classifier)
        probabilities = model.predict_proba(val_features)
        ba = 100 * balanced_accuracy_score(val_y, model.predict(val_features))
        loss = log_loss(val_y, probabilities, labels=np.arange(4))
        candidates.append(dict(order=order, selected_features=count,
                               validation_ba=float(ba), validation_log_loss=float(loss)))
        fitted.append(model)
    best = min(range(len(candidates)),
               key=lambda i: (-candidates[i]["validation_ba"],
                              candidates[i]["validation_log_loss"], i))
    return fitted[best], candidates, best


def create_study(base, subjects, engine_source, reference_csv):
    subjects = tuple(subjects)
    if not subjects or len(set(subjects)) != len(subjects) or not set(subjects) <= set(range(1, 10)):
        raise ValueError("SUBJECTS must be unique integers from 1 through 9")
    config = dict(subjects=subjects, bands=BANDS, feature_counts=FEATURE_COUNTS,
                  csp_components=4, csp_shrinkage=0.1, feature_selection_seed=20261012,
                  bootstrap_seed=20261012, bootstrap_samples=10000,
                  classifier="multiclass LDA, lsqr solver, automatic shrinkage",
                  front_end="Original run-level 4–38 Hz / cue 0–4 s / 125 Hz; source scaling",
                  filter_bank="Fourth-order Butterworth, forward/backward within each epoch",
                  split="Session 1 runs 1–5 fit; run 6 select; all session 2 test",
                  tuning="Four feature counts; no refit on validation; no target fitting",
                  status="Post-study exploratory external baseline, not a new primary contrast")
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "pandas", "mne", "scikit-learn")}
    identity = dict(config=config, software=versions, python=platform.python_version(),
                    engine_sha256=hashlib.sha256(engine_source.encode()).hexdigest(),
                    reference_sha256=hashlib.sha256(reference_csv.encode()).hexdigest())
    study_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    root = Path(base) / f"FBCSP_LDA_{study_id}"
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "protocol.json", identity)
    (root / "fbcsp_engine.py").write_text(engine_source)
    (root / "original_eegnet_per_seed.csv").write_text(reference_csv)
    return root


def select_all(subjects, cache, root):
    selections = []
    for subject in subjects:
        directory = root / f"subject_{subject:02d}"
        directory.mkdir(exist_ok=True)
        path = download_session(subject, "T", cache)
        source_sha = sha_file(path)
        model_path, selection_path = directory / "model.joblib", directory / "selection.json"
        if model_path.exists() and selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection["source_sha256"] != source_sha or selection["model_sha256"] != sha_file(model_path):
                raise RuntimeError("Cached model/data hash mismatch; inspect the study directory")
            print(f"Participant {subject}: using completed source selection", flush=True)
        else:
            x, frame = read_session(path, subject, "T")
            train = frame.run.to_numpy() <= 5
            if train.sum() != 240 or (~train).sum() != 48:
                raise ValueError("Unexpected source train/validation split")
            mean = x[train].mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
            scale = x[train].std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
            scale = np.maximum(scale, 1e-6)
            x = ((x - mean) / scale).astype(np.float32)
            y = frame.label.to_numpy()
            print(f"Participant {subject}: fitting source-only CSP and four validation candidates", flush=True)
            fit_started = time.perf_counter()
            with threadpool_limits(limits=2):
                train_features, spatial = fit_spatial(x[train], y[train])
                val_features = transform_spatial(x[~train], spatial)
                head, candidates, best = choose_head(
                    train_features, y[train], val_features, y[~train])
            bundle = dict(mean=mean, scale=scale, spatial=spatial, head=head)
            temporary = model_path.with_suffix(".tmp")
            joblib.dump(bundle, temporary, compress=3)
            os.replace(temporary, model_path)
            selection = dict(subject=int(subject), source_sha256=source_sha,
                             model_sha256=sha_file(model_path),
                             selected_features=candidates[best]["selected_features"],
                             validation_ba=candidates[best]["validation_ba"],
                             fit_seconds=time.perf_counter() - fit_started,
                             candidates=candidates)
            atomic_json(selection_path, selection)
            print(f"Participant {subject}: fit/search finished in {selection['fit_seconds']:.2f} s (CPU; excludes download/preprocessing)", flush=True)
            frame.assign(split=np.where(train, "train", "validation")).to_csv(
                directory / "source_trials.csv", index=False)
        selections.append(selection)
    # No evaluation-session file is opened until every participant's choice is locked.
    atomic_json(root / "locked_selection.json", dict(subjects=list(subjects), selections=selections))
    print("All source choices locked. Session 2 may now be evaluated.", flush=True)


def evaluate_all(subjects, cache, root):
    lock_path = root / "locked_selection.json"
    lock_sha = sha_file(lock_path)
    lock = json.loads(lock_path.read_text())
    if lock["subjects"] != list(subjects):
        raise ValueError("Locked selections do not cover the configured participants")
    rows = []
    for selection in lock["selections"]:
        subject = selection["subject"]
        directory = root / f"subject_{subject:02d}"
        model_path = directory / "model.joblib"
        if sha_file(model_path) != selection["model_sha256"]:
            raise RuntimeError("Source-fitted model changed after selection")
        model = joblib.load(model_path)  # Only this run's locally created model is loaded.
        path = download_session(subject, "E", cache)
        x, frame = read_session(path, subject, "E")
        x = ((x - model["mean"]) / model["scale"]).astype(np.float32)
        with threadpool_limits(limits=2):
            features = transform_spatial(x, model["spatial"])
            predictions = model["head"].predict(features)
        y = frame.label.to_numpy()
        ba = 100 * balanced_accuracy_score(y, predictions)
        clean = frame.artifact.to_numpy() == 0
        clean_ba = (100 * balanced_accuracy_score(y[clean], predictions[clean])
                    if np.array_equal(np.unique(y[clean]), np.arange(4)) else np.nan)
        frame.assign(predicted_label=predictions, correct=predictions == y).to_csv(
            directory / "test_predictions.csv", index=False)
        atomic_json(directory / "target_data.json",
                    dict(target_sha256=sha_file(path), model_sha256=selection["model_sha256"],
                         locked_selection_sha256=lock_sha, n_trials=len(frame)))
        rows.append(dict(subject=subject, ba=float(ba), artifact_free_ba=float(clean_ba),
                         validation_ba=selection["validation_ba"],
                         selected_features=selection["selected_features"]))
        print(f"Participant {subject}: session-2 BA {ba:.2f}%", flush=True)
    if sha_file(lock_path) != lock_sha:
        raise RuntimeError("Selections changed during target evaluation")
    scores = pd.DataFrame(rows)
    scores.to_csv(root / "participant_scores.csv", index=False)
    return scores


def participant_interval(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Intervals require finite participant-level values")
    if len(values) < 2:
        return float(values.mean()), np.nan, np.nan
    rng = np.random.default_rng(20261012)
    samples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return float(values.mean()), *np.quantile(samples, [0.025, 0.975]).tolist()


def report_results(scores, reference_csv, root):
    reference = pd.read_csv(io.StringIO(reference_csv))
    if (len(reference) != 45 or reference.duplicated(["subject", "seed"]).any()
            or set(reference.subject) != set(range(1, 10))
            or set(reference.seed) != set(range(5))
            or not np.isfinite(reference.ba).all()
            or not reference.ba.between(0, 100).all()):
        raise ValueError("Bundled EEGNet reference must contain nine participants × five seeds")
    eegnet = reference.groupby("subject", as_index=False).ba.mean().rename(columns={"ba": "eegnet_ba"})
    paired = scores.merge(eegnet, on="subject", validate="one_to_one").sort_values("subject").reset_index(drop=True)
    paired["fbcsp_minus_eegnet_pp"] = paired.ba - paired.eegnet_ba
    paired.to_csv(root / "paired_eegnet_comparison.csv", index=False)
    summary = []
    for label, values in (
        ("FBCSP-style / shrinkage LDA", paired.ba),
        ("EEGNet (original five-seed mean)", paired.eegnet_ba),
        ("FBCSP minus EEGNet (percentage points)", paired.fbcsp_minus_eegnet_pp),
    ):
        mean, low, high = participant_interval(values)
        summary.append(dict(comparison=label, mean=mean, low=low, high=high,
                            n_subjects=len(paired), complete_nine_subjects=len(paired) == 9))
    summary = pd.DataFrame(summary)
    summary.to_csv(root / "summary.csv", index=False)
    if len(paired) == 9:
        mean, low, high = participant_interval(paired.ba)
        (root / "paper_table_row.txt").write_text(
            f"FBCSP-style (MI + LDA) & {mean:.1f} [{low:.1f}, {high:.1f}] & -- & -- \\\\\n")
    else:
        print("PROVISIONAL: fewer than nine participants; do not use this subset as the paper result.")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(paired.subject, paired.ba, "o-", label="FBCSP-style / shrinkage LDA")
    ax.plot(paired.subject, paired.eegnet_ba, "s--", label="EEGNet (original seed mean)")
    ax.axhline(25, color="grey", linestyle=":", label="Chance (25%)")
    ax.set(xlabel="Participant", ylabel="Session-2 balanced accuracy (%)",
           xticks=paired.subject, ylim=(0, 100))
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "participant_comparison.png", dpi=180)
    plt.show()
    plt.close(fig)
    archive = root.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(root.parent))
    return paired, summary, archive


def self_check():
    rng = np.random.default_rng(42)
    def synthetic(n_each):
        y = np.repeat(np.arange(4), n_each)
        rng.shuffle(y)
        x = rng.normal(0, 0.5, (len(y), 22, 500))
        wave = np.sin(2 * np.pi * 10 * np.arange(500) / 125)
        for i, label in enumerate(y):
            x[i, label * 4:label * 4 + 3] += 3 * wave
        return x.astype(np.float32), y
    train, train_y = synthetic(16)
    val, val_y = synthetic(8)
    test, test_y = synthetic(8)
    train_copy = train.copy()
    small_bank = [(8, 12)]
    ft, spatial = fit_spatial(train, train_y, small_bank)
    assert np.array_equal(train, train_copy), "CSP altered the input epochs"
    fv = transform_spatial(val, spatial, small_bank)
    head, candidates, best = choose_head(ft, train_y, fv, val_y, counts=(4, 8))
    # Regression: cached ranking must match independently fitted sklearn pipelines.
    for candidate in candidates:
        reference = make_pipeline(
            StandardScaler(),
            SelectKBest(partial(mutual_info_classif, random_state=20261012),
                        k=candidate["selected_features"]),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ).fit(ft, train_y)
        assert candidate["validation_ba"] == 100 * balanced_accuracy_score(val_y, reference.predict(fv))
        assert candidate["validation_log_loss"] == log_loss(val_y, reference.predict_proba(fv), labels=np.arange(4))
        if candidate == candidates[best]:
            np.testing.assert_array_equal(reference.predict_proba(fv), head.predict_proba(fv))
    before = hashlib.sha256(pickle.dumps((spatial, head))).hexdigest()
    predictions = head.predict(transform_spatial(test, spatial, small_bank))
    assert 100 * balanced_accuracy_score(test_y, predictions) >= 75
    assert before == hashlib.sha256(pickle.dumps((spatial, head))).hexdigest(), "Inference refitted a model"
    assert len(candidates) == 2 and candidates[best]["selected_features"] in (4, 8)
    assert participant_interval([1, 2, 3])[0] == 2
    print("Synthetic fitting, held-out prediction, input immutability and interval checks passed.")



