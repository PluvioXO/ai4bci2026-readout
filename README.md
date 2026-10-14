# Recoverable Readout Mismatch in Smooth-to-Spiking EEG Decoding

Research code and results for the AI4BCI 2026 extended abstract by Maximilian Nicholson.
The paper studies whether poor converted EEG predictions reflect a failed readout
or a loss of linearly decodable information in the frozen encoder.

## Rebuild the paper's numbers without training

Use Python 3.10 or later from this directory:

```sh
python -m pip install -r requirements-analysis.txt
python build_paper_results.py
pdflatex -interaction=nonstopmode -halt-on-error ai4bci_abstract.tex
pdflatex -interaction=nonstopmode -halt-on-error ai4bci_abstract.tex
pdflatex -interaction=nonstopmode -halt-on-error ai4bci_appendix.tex
pdflatex -interaction=nonstopmode -halt-on-error ai4bci_appendix.tex
```

The Python command recomputes participant-level means, paired contrasts and bootstrap
intervals from the included CSVs and regenerates the marked table/figure blocks.
LaTeX additionally requires PGFPlots and the packages listed in the manuscript source.
The main submission is limited to two pages including references. The companion
appendix provides supporting detail and is not assumed to be an accepted supplement.

## Run the experiments

The notebooks are standalone: upload one to Google Colab and Run all. Use a GPU
runtime for the neural experiments; the classical pipeline uses CPU libraries.
No GitHub checkout or previous checkpoints are needed by the notebooks.

- `ai4bci_all_experiments.ipynb`: original study, 270 main fits plus 216 tuning fits.
- `ai4bci_fbcsp_baseline.ipynb`: post-study FBCSP-style mutual-information/shrinkage-LDA
  reference for all nine participants, with the optimized archived implementation.
- `ai4bci_readout_replication.ipynb`: post-study second-architecture readout replication,
  nine participants, five seeds, four source-validation candidates per participant.

The neural notebooks support checkpoint resumption. Keep the scientific configuration
and software unchanged when resuming and use one runtime per output directory.
The second-architecture runner also works directly:

```sh
python -m pip install -r requirements-experiments.txt
python code/replicate_readout.py --smoke
python code/replicate_readout.py
```

On CPU it trains three participants in separate worker processes; on CUDA it uses
one worker. All source training and ridge selection finish before target sessions
are loaded. Smoke outputs are synthetic and must not be used as paper evidence.

## Saved evidence and audits

`results/` contains the original per-seed E1/E2/E3 tables, the classical reference's
saved predictions and source selections, and the EEGNet source-validation audit.
The original 72 MB result archive is distributed as a GitHub release asset rather
than a Git object. It contains all 270 neural prediction exports, fitted checkpoints,
features, histories, tuning choices, protocol and software metadata. It excludes
raw recordings, prepared EEG epochs and optimizer resume states.
Download all saved run archives from the
[v1.0.0 release](https://github.com/PluvioXO/ai4bci2026-readout/releases/tag/v1.0.0).

After downloading and extracting the original archive, the EEGNet audit can be rerun:

```sh
python code/audit_eegnet.py path/to/extracted/original_run
python code/verify_predictions.py path/to/extracted/original_run
python code/audit_replication.py path/to/extracted/replication_run
```

The audit checks all 45 main and 36 pilot histories, checkpoint selection, all nine
optimizer selections, finite weights, max-norm constraints and BatchNorm update
counts. It uses source histories only. It does not establish full convergence or
retune using target-session performance.
The prediction audit independently recomputes all 675 E1, 1,350 E2 and 2,700 E3
cells, including artifact-free scores, from 1,231,200 saved prediction rows.
This is export consistency, not independent model retraining or full forward replay.
For the separate participant-1/seed-0 forward replay (SAST, matched ordinary and
direct-hard), prepare local inputs from the archive and original provider recordings:

```sh
python code/prepare_conversion_replay.py path/to/extracted/original_run replay_inputs
python conversion_audit/audit_conversion.py replay_inputs
```

This downloads only participant 1's recordings, verifies their archived checksums,
and prepares local EEG epochs. Do not add `replay_inputs/` to the public release.

Local implementation checks:

```sh
python -m unittest discover -s tests -v
```

## Interpretation and limits

The primary calibrated SAST-minus-compute-matched ordinary contrast is −0.42 pp,
95% CI [−2.88, 2.67]. This does not demonstrate superiority or establish equivalence.
Source-only ridge refitting improves raw-spike decoding; smooth-feature refitting
also improves performance, so the full raw-spike gain is not uniquely conversion repair.
Both converted models and the EEGNet reference remain below the post-study classical
reference. EEGNet uses the common 800-update study budget and is not a benchmark ceiling.

The second architecture is a post-study exploratory check on the same dataset and
participants. It does not add independent participants or replicate the SAST comparison.
Its raw-spike balanced accuracy rises from 27.44% to 37.55% after source-only refitting:
+10.11 pp, paired 95% CI [5.86, 14.29]. Smooth refitting gains +3.02 pp [0.70, 5.42];
excess raw recovery is +7.09 pp [2.01, 12.21]. Matched raw probes remain below smooth
probes by 1.91 pp (raw minus smooth: [−3.14, −0.61]).
Filtering and normalization are offline; no online-control, energy, hardware or
clinical benefit is established. Seed averaging occurs within participants before
participant bootstrap inference; seeds are not independent human observations.

Data originate from BCI Competition IV dataset 2a / BNCI2014-001:
[description](https://www.bbci.de/competition/iv/desc_2a.pdf),
[distribution](https://bnci-horizon-2020.eu/database/data-sets).
Download recordings from the original provider and follow its terms. This repository
does not redistribute EEG recordings or prepared EEG epochs. No new participants
were recruited. No institutional secondary-analysis approval or exemption is asserted
without a confirmed determination.
The applicable institutional secondary-analysis determination remains unconfirmed
in this release.

SAST is prior work, separately disclosed in the manuscript:
[first preprint](https://arxiv.org/abs/2603.18039),
[on-sensor preprint](https://arxiv.org/abs/2604.09696).
The EEG study and frozen-feature readout diagnosis are the contribution here.
