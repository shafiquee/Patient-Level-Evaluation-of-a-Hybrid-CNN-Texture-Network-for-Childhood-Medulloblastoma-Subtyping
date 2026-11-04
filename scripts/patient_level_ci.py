#!/usr/bin/env python3
"""95% CIs for the patient-level LOPO results reported in the paper (n = 10 tumour patients).

Inputs are the per-patient majority-vote calls copied from medulloblastoma_dsaf_net_final.ipynb and
mb_frozen_embedding_lopo.ipynb (same dictionaries as make_figures.py).
  * patient accuracy:  exact Clopper-Pearson interval
  * balanced accuracy: percentile interval from a bootstrap that resamples patients within class (B = 10000)
  * per-class recall (subgroup analysis): Clopper-Pearson interval
Run:  python scripts/patient_level_ci.py > outputs/patient_level_ci.txt
"""
import numpy as np
from scipy.stats import beta

PATIENTS = [("1766", "C"), ("188", "C"), ("277", "C"), ("279", "C"), ("321", "C"),
            ("141", "D"), ("203", "D"), ("411", "D"), ("17126", "L"), ("395", "L")]
VOTES = {
    "Majority-patient baseline (always Classic)": ["C"] * 10,
    "Frozen EffNetV2-S + LR (stain+CLAHE, both)": ["C", "C", "C", "C", "C", "C", "C", "C", "D", "C"],
    "DSAF-Net, no texture":              ["C", "C", "D", "C", "D", "C", "C", "C", "D", "C"],
    "DSAF-Net, texture":                 ["C", "C", "C", "C", "D", "C", "D", "C", "D", "C"],
    "DSAF-Net, texture, fine-tuned":     ["C", "C", "C", "C", "C", "C", "D", "C", "D", "C"],
    "Path Foundation head, texture":     ["C", "C", "D", "C", "D", "C", "C", "C", "D", "C"],
    # Rerun on 2026-11-02 with per-tile probabilities saved (patient_calls.csv from evaluate_predictions.py).
    # Differs from the row above at patient 1766 (GPU nondeterminism between runs).
    "DSAF-Net, texture, fine-tuned (rerun)": ["D", "C", "C", "C", "C", "C", "D", "C", "D", "C"],
}
CLASSES = {"C": "Classic", "D": "Desmoplastic/Nodular", "L": "Large cell"}


def cp(k, n, a=0.05):
    lo = 0.0 if k == 0 else beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def bal(truth, pred):
    return np.mean([(pred[truth == c] == c).mean() for c in np.unique(truth)])


def macro_spec(truth, pred):
    return np.mean([((pred != c) & (truth != c)).sum() / (truth != c).sum() for c in np.unique(truth)])


def macro_f1(truth, pred):
    f = []
    for c in np.unique(truth):
        tp = ((pred == c) & (truth == c)).sum(); fp = ((pred == c) & (truth != c)).sum(); fn = ((pred != c) & (truth == c)).sum()
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return np.mean(f)


truth = np.array([t for _, t in PATIENTS])
rng = np.random.default_rng(0)
print(f"{'method':46s} correct  acc [95% CI]          bal.acc [95% CI]")
for name, v in VOTES.items():
    pred = np.array(v)
    k = int((pred == truth).sum())
    lo, hi = cp(k, 10)
    b = bal(truth, pred)
    draws = []
    for _ in range(10000):
        idx = np.concatenate([rng.choice(np.flatnonzero(truth == c), (truth == c).sum()) for c in "CDL"])
        draws.append(bal(truth[idx], pred[idx]))
    bl, bh = np.percentile(draws, [2.5, 97.5])
    print(f"{name:46s} {k:2d}/10    {k/10:.2f} [{lo:.2f}, {hi:.2f}]    {b:.3f} [{bl:.3f}, {bh:.3f}]")

print("\nPatient-level macro metrics (one-vs-rest over the 3 tumour classes), bootstrap 95% CI, B = 10000")
print(f"{'method':46s} sens(=bal.acc)            spec                   F1")
rng = np.random.default_rng(1)
for name, v in VOTES.items():
    pred = np.array(v)
    est = [bal(truth, pred), macro_spec(truth, pred), macro_f1(truth, pred)]
    D = []
    for _ in range(10000):
        idx = np.concatenate([rng.choice(np.flatnonzero(truth == c), (truth == c).sum()) for c in "CDL"])
        D.append([bal(truth[idx], pred[idx]), macro_spec(truth[idx], pred[idx]), macro_f1(truth[idx], pred[idx])])
    D = np.array(D)
    ci = np.percentile(D, [2.5, 97.5], axis=0)
    print(f"{name:46s} " + "   ".join(f"{e:.3f} [{ci[0, j]:.3f}, {ci[1, j]:.3f}]" for j, e in enumerate(est)))

print("\nPer-class patient recall (subgroup analysis), Clopper-Pearson 95% CI")
for name, v in VOTES.items():
    pred = np.array(v)
    parts = []
    for c, cn in CLASSES.items():
        m = truth == c
        k = int((pred[m] == c).sum())
        lo, hi = cp(k, int(m.sum()))
        parts.append(f"{cn} {k}/{m.sum()} [{lo:.2f}, {hi:.2f}]")
    print(f"{name:46s} " + "; ".join(parts))
