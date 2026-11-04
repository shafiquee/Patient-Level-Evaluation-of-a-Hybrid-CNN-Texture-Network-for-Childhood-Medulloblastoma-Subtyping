#!/usr/bin/env python3
"""SHAP attribution for the 34-element GLCM + LBP texture vector under leave-one-patient-out.

Input CSV (one row per tile): patient_id, y_true, f_0 ... f_33   (feature columns start with 'f_';
use the real names, e.g. f_glcm_contrast_0, if you have them). Export it from the notebook with
export_texture_features() in scripts/notebook_snippets.py.

For each held-out patient the scaler and logistic regression are fit on the other patients only, then
SHAP values (LinearExplainer, scaled space) are computed for the held-out tiles. Mean |SHAP| is pooled
over folds. SHAP explains the model, not the disease: with 10 tumour patients and no model skill
above baseline, the ranking is a description of what the classifier used, not evidence of biology.

Usage:
    python scripts/shap_texture.py outputs/texture_features.csv --out outputs/shap
    python scripts/shap_texture.py --selftest
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def lopo_shap(df, fcols):
    classes = sorted(df["y_true"].unique())
    imp = np.zeros((len(classes), len(fcols)))
    n = 0
    for pid in df["patient_id"].unique():
        te, tr = df["patient_id"] == pid, df["patient_id"] != pid
        if df.loc[te, "y_true"].iloc[0] not in set(df.loc[tr, "y_true"]):
            continue                                       # class unseen in training: fold not evaluable
        sc = StandardScaler().fit(df.loc[tr, fcols])      # scaler fit on training patients only
        clf = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(df.loc[tr, fcols]), df.loc[tr, "y_true"])
        Xte = sc.transform(df.loc[te, fcols])
        expl = shap.LinearExplainer(clf, sc.transform(df.loc[tr, fcols]))
        sv = np.array(expl.shap_values(Xte))               # (n, F, K) or (K, n, F) depending on version
        if sv.ndim == 2:                                   # binary case
            sv = np.stack([-sv, sv])
        elif sv.shape[0] == len(Xte):
            sv = np.moveaxis(sv, 2, 0)
        for k, c in enumerate(clf.classes_):
            imp[classes.index(c)] += np.abs(sv[k]).sum(axis=0)
        n += len(Xte)
    return pd.DataFrame(imp / n, index=classes, columns=fcols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--out", default="outputs/shap")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        rng = np.random.default_rng(0)
        pats = [("P1", "C"), ("P2", "C"), ("P3", "C"), ("P4", "D"), ("P5", "D"), ("P6", "L"), ("P7", "L")]
        rows = []
        for pid, c in pats:
            X = rng.normal(0, 1, (25, 6))
            X[:, 0] += {"C": 0, "D": 1, "L": -1}[c]
            rows += [[pid, c, *x] for x in X]
        df = pd.DataFrame(rows, columns=["patient_id", "y_true"] + [f"f_{i}" for i in range(6)])
    else:
        df = pd.read_csv(a.csv)
    fcols = [c for c in df.columns if c.startswith("f_")]
    imp = lopo_shap(df, fcols)
    os.makedirs(a.out, exist_ok=True)
    imp.to_csv(os.path.join(a.out, "shap_mean_abs.csv"))
    tot = imp.sum(axis=0).sort_values(ascending=False).head(a.top)[::-1]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.barh(tot.index, tot.values, color="#2a78d6")
    ax.set_xlabel("Mean |SHAP| summed over classes")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(a.out, f"fig_shap_texture.{ext}"), dpi=300)
    print(imp.sum(axis=0).sort_values(ascending=False).head(a.top).round(3).to_string())
    if a.selftest:
        print("[SELFTEST] feature f_0 was built to carry the signal; it should rank first.")


if __name__ == "__main__":
    main()
