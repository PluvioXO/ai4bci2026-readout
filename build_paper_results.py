"""Regenerate paper tables, plot data, and paired estimates from completed outputs.

Run: python build_paper_results.py (requires numpy and pandas).
The TeX files remain self-contained after generation; this never retrains models.
"""
from pathlib import Path
import hashlib
import itertools
import json
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
METHODS = ("ordinary", "matched", "sam", "sast", "direct", "eegnet")
NAMES = ("Ordinary", r"Ordinary, $2\times$ updates", "Same-batch SAM",
         "Independent SAST", "Direct hard-forward", "EEGNet reference")
SHORT = ("Ordinary", r"Ordinary $2\times$", "SAM", "SAST", "Direct hard", "EEGNet")
e1 = pd.read_csv(RESULTS / "e1_per_seed.csv")
e2 = pd.read_csv(RESULTS / "e2_per_seed.csv")
e3 = pd.read_csv(RESULTS / "e3_per_seed.csv")
cost = pd.read_csv(RESULTS / "training_cost.csv")
fbcsp = pd.read_csv(RESULTS / "fbcsp_baseline/participant_scores.csv").sort_values("subject")


def interval(values):
    """Same participant percentile bootstrap and seed as the executed engine."""
    values = np.asarray(values, dtype=float)
    if len(values) != 9 or not np.isfinite(values).all():
        raise ValueError("Expected nine finite participant estimates")
    rng = np.random.default_rng(20261012)
    samples = values[rng.integers(9, size=(10000, 9))].mean(axis=1)
    return np.array([values.mean(), *np.quantile(samples, [.025, .975])])


def estimate(frame, metric):
    return interval(frame.groupby("subject")[metric].mean())


def row(*cells):
    return " & ".join(map(str, cells)) + r" \\"


def ci(values, digits=2):
    mean, low, high = values
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def replace_block(path, name, text):
    source = path.read_text()
    begin, end = f"% BEGIN GENERATED {name}", f"% END GENERATED {name}"
    pattern = re.escape(begin) + r".*?" + re.escape(end)
    updated, n = re.subn(pattern, lambda _: begin + "\n" + text + "\n" + end,
                         source, flags=re.S)
    if n != 1:
        raise ValueError(f"Expected exactly one {name} block in {path}")
    if updated != source:
        path.write_text(updated)


for frame, count, extra in (
    (e1, 675, []), (e2, 1350, ["layer"]),
    (e3, 2700, ["intervention", "budget"])
):
    keys = ["subject", "seed", "method", "mode"] + extra
    assert len(frame) == count and not frame.duplicated(keys).any()
    assert set(frame.subject) == set(range(1, 10)) and set(frame.seed) == set(range(5))
    assert (frame.groupby(["method", "mode"] + extra).size() == 45).all()
assert len(cost) == 270
assert set(e3.budget) == {0, 16, 32, 48}
assert fbcsp.subject.tolist() == list(range(1, 10))
assert fbcsp.ba.between(0, 100).all()
assert set(fbcsp.selected_features) <= {8, 16, 32, 64}
fbcsp_participants = fbcsp.set_index("subject").ba
eegnet_participants = e1.query("method == 'eegnet' and mode == 'continuous'").groupby("subject").ba.mean()
fbcsp_estimate = interval(fbcsp_participants)
fbcsp_difference = interval(fbcsp_participants - eegnet_participants)
exported_fbcsp = pd.read_csv(RESULTS / "fbcsp_baseline/summary.csv")
for values, index in ((fbcsp_estimate, 0), (interval(eegnet_participants), 1), (fbcsp_difference, 2)):
    np.testing.assert_allclose(values, exported_fbcsp.iloc[index][["mean", "low", "high"]].astype(float),
                               atol=1e-10, rtol=0)

stats1 = {(m, mode): estimate(g, "ba") for (m, mode), g in e1.groupby(["method", "mode"])}
stats2 = {(m, mode, layer): estimate(g, "probe_ba")
          for (m, mode, layer), g in e2.groupby(["method", "mode", "layer"])}
stats3 = {(m, mode, branch, budget): estimate(g, "ba")
          for (m, mode, branch, budget), g in e3.groupby(["method", "mode", "intervention", "budget"])}
# Verify the final analysis against the original exported E3 intervals.
original3 = pd.read_csv(RESULTS / "e3_summary.csv").query("metric == 'ba'")
for record in original3.itertuples():
    np.testing.assert_allclose(
        stats3[(record.method, record.mode, record.intervention, record.budget)],
        [record.mean, record.low, record.high], atol=1e-10, rtol=0)

wide = e1.query("mode == 'calibrated'").pivot(
    index=["subject", "seed"], columns="method", values="ba")
primary_participants = (wide.sast - wide.matched).groupby(level="subject").mean()
primary = interval(primary_participants)
clean = e1.query("mode == 'calibrated'").pivot(
    index=["subject", "seed"], columns="method", values="artifact_free_ba")
clean_primary = interval((clean.sast - clean.matched).groupby(level="subject").mean())
signs = np.array(list(itertools.product([-1, 1], repeat=9)))
p = np.mean(np.abs((signs * primary_participants.to_numpy()).mean(1)) >= abs(primary[0]) - 1e-12)
primary_participants.rename("sast_minus_matched_pp").to_csv(RESULTS / "primary_participant_effects.csv")

contrasts = []
for method in METHODS[:4]:
    g = e2.query("method == @method and layer == 2")
    probes = g.pivot(index=["subject", "seed"], columns="mode", values="probe_ba")
    residual = interval((probes.raw - probes.continuous).groupby(level="subject").mean())
    recovery = estimate(g.query("mode == 'raw'"), "recovery_pp")
    smooth_recovery = estimate(g.query("mode == 'continuous'"), "recovery_pp")
    for name, values in [("raw_probe_minus_smooth_probe", residual),
                         ("raw_probe_minus_original_head", recovery),
                         ("smooth_probe_minus_original_head", smooth_recovery)]:
        contrasts.append(dict(method=method, contrast=name, mean=values[0], low=values[1], high=values[2]))
contrasts = pd.DataFrame(contrasts)
contrasts.to_csv(RESULTS / "paired_readout_contrasts.csv", index=False)
pd.DataFrame([dict(method=m, mode=mode, mean=v[0], low=v[1], high=v[2])
              for (m, mode), v in stats1.items()]).to_csv(RESULTS / "e1_summary.csv", index=False)
pd.DataFrame([dict(method=m, mode=mode, layer=layer, mean=v[0], low=v[1], high=v[2])
              for (m, mode, layer), v in stats2.items()]).to_csv(RESULTS / "probe_accuracy_summary.csv", index=False)
(RESULTS / "paper_estimates.json").write_text(json.dumps({
    "primary_sast_minus_matched_pp": primary.tolist(),
    "artifact_free_primary_pp": clean_primary.tolist(),
    "sign_flip_p": float(p), "participants": 9, "seeds_per_participant": 5,
    "bootstrap_resamples": 10000, "bootstrap_seed": 20261012,
    "readout_contrasts": contrasts.to_dict("records"),
    "source_run": "BNCI2014_001_ccebe0c6749374ab",
    "post_study_fbcsp_style_baseline": {
        "source_run": "FBCSP_LDA_fb436529d768932f",
        "ba": fbcsp_estimate.tolist(),
        "minus_eegnet_pp": fbcsp_difference.tolist(),
        "participants": 9, "fits_per_participant": 1,
        "source_validation_candidates": 4,
    },
    "source_csv_sha256": {
        name: hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()
        for name in ("e1_per_seed.csv", "e2_per_seed.csv", "e3_per_seed.csv", "training_cost.csv",
                     "fbcsp_baseline/participant_scores.csv", "fbcsp_baseline/summary.csv")
    }
}, indent=2) + "\n")

main = ROOT / "ai4bci_abstract.tex"
appendix = ROOT / "ai4bci_appendix.tex"
replace_block(main, "E1 ROWS", "\n".join(
    row(name, *(ci(stats1[(m, mode)], 1) if (m, mode) in stats1 else "--"
                for mode in ("continuous", "raw", "calibrated")))
    for m, name in zip(METHODS, NAMES)))
replace_block(main, "FBCSP ROW", row("FBCSP-style reference", ci(fbcsp_estimate, 1), "--", "--"))
replace_block(appendix, "FBCSP PARTICIPANT ROWS", "\n".join(
    row(int(record.subject), int(record.selected_features), f"{record.ba:.2f}",
        f"{eegnet_participants.loc[record.subject]:.2f}",
        f"{record.ba - eegnet_participants.loc[record.subject]:+.2f}")
    for record in fbcsp.itertuples()))

plot = [
    r"\begin{tikzpicture}",
    r"\begin{axis}[",
    r"width=0.91\linewidth,height=4.2cm,",
    r"xmin=-0.5,xmax=3.5,ymin=23,ymax=53,",
    r"xtick={0,1,2,3},xticklabels={Ordinary,Ordinary $2\times$,SAM,SAST},",
    r"ytick={25,30,35,40,45,50},ylabel={Balanced accuracy (\%)},",
    r"tick label style={font=\small},label style={font=\small},",
    r"axis lines=left,ymajorgrids,grid style={black!12},",
    r"legend style={font=\small,draw=none,at={(0.5,1.03)},anchor=south,column sep=5pt},",
    r"legend columns=3,legend cell align=left]",
    r"\addplot[black!45,dashed,forget plot] coordinates {(-0.5,25) (3.5,25)};"
]
for mode, probe, offset, mark, color, label in (
    ("raw", False, -.18, "o", "black", "Raw: original head"),
    ("raw", True, 0, "*", "black", "Raw: ridge head"),
    ("continuous", True, .18, "square*", "black!55", "Smooth: ridge head"),
):
    plot += [rf"\addplot[only marks,{color},mark={mark},error bars/.cd,y dir=both,y explicit]",
             "table[x=x,y=mean,y error minus=minus,y error plus=plus]{",
             "x mean minus plus"]
    for x, method in enumerate(METHODS[:4]):
        mean, low, high = stats2[(method, mode, 2)] if probe else stats1[(method, mode)]
        plot.append(f"{x+offset:.2f} {mean:.9f} {mean-low:.9f} {high-mean:.9f}")
    plot += ["};", rf"\addlegendentry{{{label}}}"]
plot += [r"\end{axis}", r"\end{tikzpicture}"]
replace_block(main, "READOUT PLOT", "\n".join(plot))

replace_block(appendix, "COST ROWS", "\n".join(
    row(name, 1600 if m == "matched" else 800,
        int(cost.query("method == @m").gradient_evaluations.iloc[0]),
        f"{cost.query('method == @m').seconds.mean():.1f}")
    for m, name in zip(METHODS, SHORT)))
replace_block(appendix, "PRIMARY ROW", row(r"Difference (pp)",
    *("$" + f"{v:+.2f}" + "$" for v in primary_participants)))

conversion_rows, probe_rows, cka_rows = [], [], []
for method, name in zip(METHODS[:4], SHORT):
    g = e1.query("method == @method")
    w = g.pivot(index=["subject", "seed"], columns="mode", values="ba")
    conversion_rows.append(row(name, *(
        ci(interval((w.continuous - w[mode]).groupby(level="subject").mean()))
        for mode in ("raw", "calibrated")), *(
        int((g.query("mode == @mode").prediction_classes == 1).sum())
        for mode in ("raw", "calibrated"))))
    c = contrasts.query("method == @method").set_index("contrast")
    probe_rows.append(row(name, f"{stats2[(method, 'continuous', 2)][0]:.2f}",
        f"{stats2[(method, 'raw', 2)][0]:.2f}",
        ci(c.loc["raw_probe_minus_original_head", ["mean", "low", "high"]]),
        ci(c.loc["raw_probe_minus_smooth_probe", ["mean", "low", "high"]])))
    cka_rows.append(row(name, *(
        f"{e2.query('method == @method and mode == @mode and layer == @layer').cka.mean():.3f}"
        for layer in (1, 2) for mode in ("raw", "calibrated"))))
replace_block(appendix, "CONVERSION ROWS", "\n".join(conversion_rows))
replace_block(appendix, "PROBE ROWS", "\n".join(probe_rows))
replace_block(appendix, "CKA ROWS", "\n".join(cka_rows))

def adaptation_row(method, name, mode, branch):
    g = e3.query("method == @method and mode == @mode and intervention == @branch")
    w = g.pivot(index=["subject", "seed"], columns="budget", values="ba")
    gain = interval((w[48] - w[0]).groupby(level="subject").mean())
    exported = pd.read_csv(RESULTS / "adaptation_gains_summary.csv").query(
        "method == @method and mode == @mode and intervention == @branch and budget == 48")
    np.testing.assert_allclose(gain, exported[["mean", "low", "high"]].iloc[0], atol=1e-10, rtol=0)
    cells = [name]
    if branch == "readout":
        cells.append("Spikes" if mode == "calibrated" else ("Native" if method == "eegnet" else "Smooth"))
    cells.extend(f"{stats3[(method, mode, branch, b)][0]:.2f}" for b in (0, 16, 32, 48))
    cells.append(ci(gain))
    return row(*cells)

adaptation_rows, threshold_rows = [], []
for method, name in zip(METHODS, SHORT):
    modes = ("continuous", "calibrated") if method in METHODS[:4] else (
        ("calibrated",) if method == "direct" else ("continuous",))
    for mode in modes:
        adaptation_rows.append(adaptation_row(method, name, mode, "readout"))
    if method != "eegnet":
        threshold_rows.append(adaptation_row(method, name, "calibrated", "threshold"))
replace_block(appendix, "ADAPTATION ROWS", "\n".join(adaptation_rows))
replace_block(appendix, "THRESHOLD ROWS", "\n".join(threshold_rows))
print("Verified 675 E1, 1,350 E2, and 2,700 E3 cells; regenerated paper plot and tables.")
print(f"Primary: {ci(primary)} pp; exact sign-flip p={p:.8f}.")
print(f"Post-study FBCSP-style BA: {ci(fbcsp_estimate)}%; paired minus EEGNet: {ci(fbcsp_difference)} pp.")

replication = pd.read_csv(RESULTS / "architecture_replication/per_seed.csv")
assert len(replication) == 180
assert not replication.duplicated(["subject", "seed", "mode", "readout"]).any()
assert set(replication.subject) == set(range(1, 10)) and set(replication.seed) == set(range(5))
assert replication.groupby(["mode", "readout"]).size().eq(45).all()
rep_participants = replication.groupby(["subject", "mode", "readout"]).ba.mean().unstack(["mode", "readout"])
rep_values = {"_".join(key): rep_participants[key].to_numpy() for key in rep_participants}
rep_values.update(
    raw_ridge_minus_raw_original=rep_values["raw_ridge"] - rep_values["raw_original"],
    smooth_ridge_minus_smooth_original=rep_values["smooth_ridge"] - rep_values["smooth_original"],
    raw_ridge_minus_smooth_ridge=rep_values["raw_ridge"] - rep_values["smooth_ridge"],
    excess_raw_recovery_over_smooth_recovery=(rep_values["raw_ridge"] - rep_values["raw_original"])
                                           - (rep_values["smooth_ridge"] - rep_values["smooth_original"]))
rep_stats = {name: interval(values) for name, values in rep_values.items()}
rep_export = pd.read_csv(RESULTS / "architecture_replication/summary.csv").set_index("comparison")
assert set(rep_export.index) == set(rep_stats)
for name, value in rep_stats.items():
    np.testing.assert_allclose(value, rep_export.loc[name, ["mean", "low", "high"]], atol=1e-10, rtol=0)
raw, ridge = rep_stats["raw_original"][0], rep_stats["raw_ridge"][0]
gain = rep_stats["raw_ridge_minus_raw_original"]
smooth_gain = rep_stats["smooth_ridge_minus_smooth_original"]
excess_gain = rep_stats["excess_raw_recovery_over_smooth_recovery"]
replace_block(main, "ARCHITECTURE CHECK", (
    r"\textbf{Post-study architecture check.} "
    f"A second encoder yields {raw:.2f}\\% raw BA and {ridge:.2f}\\% after source-only refitting "
    f"(gain ${gain[0]:+.2f}$ pp $[{gain[1]:.2f},{gain[2]:.2f}]$); "
    f"recovery exceeds matched smooth refitting by ${excess_gain[0]:+.2f}$ pp "
    f"$[{excess_gain[1]:.2f},{excess_gain[2]:.2f}]$."))
replace_block(appendix, "ARCHITECTURE SUMMARY", (
    f"Raw original-head BA is {ci(rep_stats['raw_original'])}\\%; source-only raw-spike "
    f"ridge BA is {ci(rep_stats['raw_ridge'])}\\%, giving a paired gain of "
    f"{ci(gain)} pp. Smooth original-head BA is {ci(rep_stats['smooth_original'])}\\% "
    f"and smooth-ridge BA is {ci(rep_stats['smooth_ridge'])}\\%, with gain {ci(smooth_gain)} pp. "
    f"The raw-minus-smooth matched-ridge contrast is {ci(rep_stats['raw_ridge_minus_smooth_ridge'])} pp; "
    f"excess raw recovery over smooth recovery is {ci(rep_stats['excess_raw_recovery_over_smooth_recovery'])} pp. "
    "Intervals are paired participant-bootstrap intervals. These estimates describe this "
    "architecture check; they do not add independent participants or establish generality across datasets."))
replace_block(appendix, "ARCHITECTURE ROWS", "\n".join(
    row(subject, *(f"{rep_participants.loc[subject, key]:.2f}" for key in (
        ("smooth", "original"), ("smooth", "ridge"), ("raw", "original"), ("raw", "ridge"))))
    for subject in range(1, 10)))
estimates_path = RESULTS / "paper_estimates.json"
estimates = json.loads(estimates_path.read_text())
estimates["post_study_architecture_replication"] = {name: value.tolist() for name, value in rep_stats.items()}
estimates["source_csv_sha256"].update({
    "architecture_replication/" + name: hashlib.sha256((RESULTS / "architecture_replication" / name).read_bytes()).hexdigest()
    for name in ("per_seed.csv", "summary.csv")})
estimates_path.write_text(json.dumps(estimates, indent=2) + "\n")
print(f"Second architecture: raw recovery {ci(gain)} pp; smooth recovery {ci(smooth_gain)} pp.")
