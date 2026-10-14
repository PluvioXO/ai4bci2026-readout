"""Build a standalone Colab/local notebook with both experiment sources embedded."""
import hashlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def build():
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))
    md("""
# AI4BCI: second-architecture readout replication

Standalone notebook: upload to Colab, choose a GPU runtime, and Run all.
Local Jupyter also works. Google Drive persists checkpoints in Colab.

This post-study exploratory experiment tests whether source-only ridge refitting
recovers performance after smooth-to-spiking conversion in a structurally different
encoder. It uses the same BCI IV-2a dataset and nine participants, not independent data.
It does not repeat the SAST comparison.

**Default:** nine participants × (four tuning fits + five main fits) = 81 fits.
All source training and readout selection finish before target sessions are loaded.
Rerun unchanged after a disconnect to resume from completed epochs. Keep one notebook
runtime per output directory. Numerical results can differ across software/devices.
""")
    code('''
import importlib.metadata, importlib.util, os, subprocess, sys, tempfile, hashlib, json, shutil
from pathlib import Path

required = ["numpy", "scipy", "pandas", "matplotlib", "torch"]
missing = [p for p in required if importlib.util.find_spec(p) is None]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)

RUN_FULL_REPLICATION = True
USE_GOOGLE_DRIVE = True
try:
    from google.colab import drive
except ImportError:
    drive = None
if drive is not None and USE_GOOGLE_DRIVE:
    drive.mount("/content/drive")
    PROJECT = Path("/content/drive/MyDrive/AI4BCI_readout_replication")
else:
    PROJECT = Path(os.environ.get("AI4BCI_REPLICATION_DIR", "ai4bci_replication")).resolve()
PROJECT.mkdir(parents=True, exist_ok=True)
print("Persistent outputs:", PROJECT)
''')
    md("""
## Fixed experiment

Original preprocessing: run-wise 4–38 Hz zero-phase filter, 0–4 s after the cue,
250→125 Hz resampling, source-training channel standardization. Session 1 runs 1–5
train (240 trials), run 6 validates (48), all session 2 tests (288). Artifact-marked
trials are retained, with artifact-free scores also exported. This is offline decoding.

Second encoder: 8 shared temporal filters (63 samples), depthwise electrode filters
(two per temporal filter), average pooling by 8, GroupNorm(1), one LIF block
(16 channels, leak 0.9, subtractive previous-activity reset, threshold 1), four
temporal-bin averages, linear 64→4 head. Linear spatial filtering is evaluated
before temporal filtering to reduce intermediates; a reference-value/gradient
check verifies that the filter ordering represents the same operation.

Ordinary smooth AdamW training only: 800 updates, batch 32, cosine LR schedule;
LR {0.0003, 0.001} × weight decay {0.0001, 0.001}. Pilot seed 1000 selects settings;
main seeds 0–4 use the selected source settings. Best checkpoint at 20 validation
points uses accuracy, then loss, then earlier epoch. No target-dependent selection.

Smooth and raw-spike final features receive separately fitted ridge heads with
the original seven-value source-validation penalty grid. Encoders stay frozen.
Four outcomes: smooth/original, smooth/ridge, raw/original, raw/ridge. Seed means
within participants precede paired 10,000-resample participant bootstrap intervals.
The smooth-refit control distinguishes general refitting gains from extra raw-spike
recovery. All results, including weak or contrary replication, must be retained.
""")
    sources = {name: (ROOT / "code" / name).read_text()
               for name in ("ai4bci_study.py", "replicate_readout.py")}
    hashes = {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()}
    code(f"SOURCES = {sources!r}\nEXPECTED_SHA256 = {hashes!r}\n" + '''
ENGINE_DIR = Path(tempfile.mkdtemp(prefix="ai4bci_replication_engine_"))
for name, source in SOURCES.items():
    assert hashlib.sha256(source.encode()).hexdigest() == EXPECTED_SHA256[name]
    (ENGINE_DIR / name).write_text(source)
print("Embedded source hashes:", EXPECTED_SHA256)
''')
    code('''
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
workers = 1 if device == "cuda" else min(3, os.cpu_count() or 1)
command = [sys.executable, "-u", str(ENGINE_DIR / "replicate_readout.py"),
           "--cache", str(PROJECT / "raw_data"), "--output", str(PROJECT / "runs"),
           "--device", device, "--workers", str(workers)]
subprocess.run(command + ["--smoke"], check=True)
print("Synthetic check passed; these are not paper results.")
''')
    code('''
if RUN_FULL_REPLICATION:
    subprocess.run(command, check=True)
else:
    print("Full real-data replication disabled; no real replication result is available.")
''')
    code('''
import zipfile
from IPython.display import FileLink, display
roots = sorted((PROJECT / "runs").glob("DEPTHWISE_LIF_*/complete.json"),
               key=lambda p: p.stat().st_mtime)
if RUN_FULL_REPLICATION and roots:
    ROOT = roots[-1].parent
    import pandas as pd
    display(pd.read_csv(ROOT / "summary.csv"))
    ARCHIVE = PROJECT / (ROOT.name + "_results.zip")
    with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(ROOT.rglob("*")):
            if p.is_file() and p.name != "resume.pt" and not p.name.startswith("epochs_"):
                z.write(p, p.relative_to(ROOT))
    display(FileLink(str(ARCHIVE)))
    print("Results:", ROOT, "Download:", ARCHIVE)
''')
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "ai4bci_source_sha256": hashes})
    nbf.validate(nb)
    destination = ROOT / "ai4bci_readout_replication.ipynb"
    nbf.write(nb, destination)
    print(destination)


if __name__ == "__main__":
    build()
