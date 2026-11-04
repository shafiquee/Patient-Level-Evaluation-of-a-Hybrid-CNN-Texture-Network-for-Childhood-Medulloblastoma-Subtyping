#!/usr/bin/env python3
"""Metrics with patient-cluster bootstrap 95% CIs, plus ROC / PR / calibration plots.

Input: a CSV of held-out (leave-one-patient-out) tile predictions, one row per tile:
    patient_id, y_true, p_<class1>, p_<class2>, ...      (probabilities, rows sum to ~1)
Produced by save_lopo_predictions() in scripts/notebook_snippets.py.

Usage:
    python scripts/evaluate_predictions.py outputs/lopo_predictions.csv --out outputs/eval
    python scripts/evaluate_predictions.py --selftest          # synthetic data, no real data needed

What it reports (one-vs-rest, macro-averaged over classes that have both positives and negatives):
    AUROC, AUPRC, sensitivity, specificity, F1, Brier score, expected calibration error (ECE),
    plus tile-level and patient-level (majority vote) accuracy and balanced accuracy.
Confidence intervals: percentile 95% CI from B patient-cluster bootstrap resamples. Patients are
resampled with replacement *within their class*, and all tiles of a patient move together, so
tiles from one slide are never treated as independent. A class with a single patient (here: Normal)
cannot vary between resamples, so its contribution is fixed. With 10 tumour patients the CIs are wide;
that is the honest answer, not a bug.

Regression metrics (MAE, RMSE) do not apply: this is a classification task.
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, roc_auc_score, roc_curve)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = [BLUE, ORANGE, AQUA, "#7a5cc9"]


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial CI for k successes out of n."""
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def ece_score(conf, correct, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return e


def tile_metrics(y, P):
    """y: int labels (n,), P: probabilities (n, K). Macro one-vs-rest over evaluable classes."""
    K = P.shape[1]
    pred = P.argmax(1)
    ks = [k for k in range(K) if (y == k).any() and (y != k).any()]
    if not ks:
        return {}
    sens, spec = [], []
    for k in ks:
        pos, neg = y == k, y != k
        sens.append(((pred == k) & pos).sum() / pos.sum())
        spec.append(((pred != k) & neg).sum() / neg.sum())
    return {
        "AUROC": np.mean([roc_auc_score(y == k, P[:, k]) for k in ks]),
        "AUPRC": np.mean([average_precision_score(y == k, P[:, k]) for k in ks]),
        "Sensitivity": np.mean(sens),
        "Specificity": np.mean(spec),
        "F1": f1_score(y, pred, labels=ks, average="macro", zero_division=0),
        "Brier": np.mean([brier_score_loss(y == k, P[:, k]) for k in ks]),
        "ECE": ece_score(P.max(1), (pred == y).astype(float)),
        "Tile accuracy": (pred == y).mean(),
    }


def patient_level(df, pcols, classes):
    """Majority vote over tiles per patient, then accuracy and balanced accuracy over patients."""
    pred_tile = df[pcols].values.argmax(1)
    d = pd.DataFrame({"pid": df["patient_id"].values, "y": df["y_idx"].values, "pred": pred_tile})
    rows = []
    for pid, g in d.groupby("pid"):
        votes = np.bincount(g["pred"], minlength=len(classes))
        rows.append((pid, g["y"].iloc[0], votes.argmax()))
    r = pd.DataFrame(rows, columns=["pid", "y", "pred"])
    acc = (r.y == r.pred).mean()
    bal = np.mean([(r.pred[r.y == c] == c).mean() for c in r.y.unique()])
    return {"Patient accuracy": acc, "Patient balanced accuracy": bal}, r


def bootstrap_patient_level(df, pcols, classes, B=1000, seed=0):
    """Stratified patient bootstrap for the patient-level (majority vote) metrics."""
    rng = np.random.default_rng(seed)
    _, r = patient_level(df, pcols, classes)
    out = []
    by_class = {c: r[r.y == c] for c in r.y.unique()}
    for _ in range(B):
        parts = [g.iloc[rng.integers(0, len(g), len(g))] for g in by_class.values()]
        s = pd.concat(parts)
        acc = (s.y == s.pred).mean()
        bal = np.mean([(s.pred[s.y == c] == c).mean() for c in s.y.unique()])
        out.append({"Patient accuracy": acc, "Patient balanced accuracy": bal})
    return pd.DataFrame(out)


def summarise(df, pcols, classes, B=1000, seed=0):
    y = df["y_idx"].values
    P = df[pcols].values
    point = tile_metrics(y, P)
    pl, r = patient_level(df, pcols, classes)
    point.update(pl)
    # tile-level metrics: cluster bootstrap with duplicate patients kept distinct
    rng = np.random.default_rng(seed)
    pid_class = df.groupby("patient_id")["y_idx"].first()
    groups = {c: pid_class.index[pid_class == c].to_numpy() for c in pid_class.unique()}
    rows_of = {pid: np.flatnonzero(df["patient_id"].values == pid) for pid in pid_class.index}
    draws = []
    for _ in range(B):
        idx = np.concatenate([rows_of[p] for c, pids in groups.items()
                              for p in rng.choice(pids, size=len(pids), replace=True)])
        draws.append(tile_metrics(y[idx], P[idx]))
    tdraw = pd.DataFrame(draws)
    pdraw = bootstrap_patient_level(df, pcols, classes, B, seed + 1)
    allraw = pd.concat([tdraw, pdraw], axis=1)
    rows = []
    for k, v in point.items():
        lo, hi = np.nanpercentile(allraw[k], [2.5, 97.5])
        rows.append({"metric": k, "estimate": v, "ci_low": lo, "ci_high": hi})
    out = pd.DataFrame(rows)
    # exact binomial CI for patient accuracy (does not need resampling)
    k = int((r.y == r.pred).sum())
    lo, hi = clopper_pearson(k, len(r))
    out.loc[len(out)] = {"metric": f"Patient accuracy ({k}/{len(r)}), Clopper-Pearson",
                         "estimate": k / len(r), "ci_low": lo, "ci_high": hi}
    return out, r


def plots(df, pcols, classes, outdir):
    y = df["y_idx"].values
    P = df[pcols].values
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    for k, c in enumerate(classes):
        if not ((y == k).any() and (y != k).any()):
            continue
        col = COLORS[k % len(COLORS)]
        fpr, tpr, _ = roc_curve(y == k, P[:, k])
        axs[0].plot(fpr, tpr, color=col, lw=1.4, label=f"{c} ({roc_auc_score(y == k, P[:, k]):.2f})")
        pr, rc, _ = precision_recall_curve(y == k, P[:, k])
        axs[1].plot(rc, pr, color=col, lw=1.4, label=f"{c} ({average_precision_score(y == k, P[:, k]):.2f})")
        frac, mean = calibration_curve(y == k, P[:, k], n_bins=8, strategy="quantile")
        axs[2].plot(mean, frac, marker="o", ms=3, color=col, lw=1.2, label=c)
    axs[0].plot([0, 1], [0, 1], color="#52514e", lw=0.7, ls="--")
    axs[2].plot([0, 1], [0, 1], color="#52514e", lw=0.7, ls="--")
    axs[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC (AUROC)")
    axs[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (AUPRC)")
    axs[2].set(xlabel="Predicted probability", ylabel="Observed fraction", title="Calibration")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, [t.replace("_", " ") for t in l], loc="lower center", ncol=len(l), frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig_roc_pr_calibration.{ext}"), dpi=300)


def synthetic(n_tiles=30, seed=0):
    """Fake LOPO predictions with a weak real signal. For testing the code only."""
    rng = np.random.default_rng(seed)
    classes = ["Classic", "Desmoplastic", "Large cell"]
    pats = [("P1", 0), ("P2", 0), ("P3", 0), ("P4", 0), ("P5", 0),
            ("P6", 1), ("P7", 1), ("P8", 1), ("P9", 2), ("P10", 2)]
    rows = []
    for pid, c in pats:
        for _ in range(n_tiles):
            logit = rng.normal(0, 1, 3)
            logit[c] += 0.6
            p = np.exp(logit) / np.exp(logit).sum()
            rows.append([pid, classes[c], *p])
    return pd.DataFrame(rows, columns=["patient_id", "y_true"] + [f"p_{c}" for c in classes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    df = synthetic() if a.selftest else pd.read_csv(a.csv)
    pcols = [c for c in df.columns if c.startswith("p_")]
    classes = [c[2:] for c in pcols]
    df["y_idx"] = df["y_true"].map({c: i for i, c in enumerate(classes)})
    assert df["y_idx"].notna().all(), "y_true values must match the p_<class> column names"
    df["y_idx"] = df["y_idx"].astype(int)
    assert np.allclose(df[pcols].sum(axis=1), 1, atol=1e-3), "probabilities must sum to 1"
    table, per_patient = summarise(df, pcols, classes, B=a.B)
    os.makedirs(a.out, exist_ok=True)
    table.to_csv(os.path.join(a.out, "metrics_with_ci.csv"), index=False)
    per_patient.to_csv(os.path.join(a.out, "patient_calls.csv"), index=False)
    plots(df, pcols, classes, a.out)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(table.to_string(index=False))
    if a.selftest:
        print("\n[SELFTEST] synthetic data: these numbers mean nothing about medulloblastoma.")


if __name__ == "__main__":
    main()
