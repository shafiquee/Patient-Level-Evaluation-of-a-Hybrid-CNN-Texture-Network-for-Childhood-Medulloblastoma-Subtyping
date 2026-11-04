#!/usr/bin/env python3
"""Patient-cluster bootstrap CIs for the leakage gap (image-level minus LOPO balanced tile accuracy).

Input: outputs/frozen_tile_predictions.csv, written by mb_frozen_embedding_lopo.ipynb (patched version):
    extractor, preproc, magnification, patient_id, y, lopo_pred, img_pred     (one row per tile and configuration)
For each extractor the point estimate is the mean over its preprocessing x magnification configurations,
as in the paper's Fig. 4. In each of B resamples, patients are drawn with replacement within class
(the same draw is applied to every configuration), tiles of a patient keep their weight, and the balanced tile
accuracy is recomputed for both splits. Percentile 95% CIs.
Usage:  python scripts/gap_ci.py outputs/frozen_tile_predictions.csv [--B 2000]
        python scripts/gap_ci.py --selftest
"""
import argparse
import numpy as np
import pandas as pd


def prepare(df):
    """Per config: arrays (n_patients_in_config, n_classes) of tile counts and correct counts."""
    pats = np.sort(df["patient_id"].astype(str).unique())
    classes = np.sort(df["y"].unique())
    pidx = {p: i for i, p in enumerate(pats)}
    pcls = df.groupby(df["patient_id"].astype(str))["y"].first()
    out = {}
    for key, g in df.groupby(["extractor", "preproc", "magnification"]):
        n = np.zeros((len(pats), len(classes))); cl = n.copy(); ci = n.copy()
        for p, y, a, b in zip(g["patient_id"].astype(str), g["y"], g["lopo_pred"], g["img_pred"]):
            r, c = pidx[p], int(np.searchsorted(classes, y))
            n[r, c] += 1; cl[r, c] += (a == y); ci[r, c] += (b == y)
        out[key] = (n, cl, ci)
    return pats, np.array([pcls[p] for p in pats]), out


def bal(w, n, hit):
    tot = w @ n; ok = w @ hit
    m = tot > 0
    return float(np.mean(ok[m] / tot[m]))


def run(df, B=2000, seed=0):
    pats, pc, cfg = prepare(df)
    rng = np.random.default_rng(seed)
    exts = sorted({k[0] for k in cfg})
    point = {e: [] for e in exts}
    for k, (n, cl, ci) in cfg.items():
        w = np.ones(len(pats)); point[k[0]].append((bal(w, n, cl), bal(w, n, ci)))
    draws = {e: [] for e in exts}
    for _ in range(B):
        w = np.zeros(len(pats))
        for c in np.unique(pc):
            idx = np.flatnonzero(pc == c)
            for j in rng.choice(idx, size=len(idx), replace=True):
                w[j] += 1
        per = {e: [] for e in exts}
        for k, (n, cl, ci) in cfg.items():
            per[k[0]].append((bal(w, n, cl), bal(w, n, ci)))
        for e in exts:
            draws[e].append(np.mean(per[e], axis=0))
    rows = []
    for e in exts:
        pt = np.mean(point[e], axis=0); d = np.array(draws[e])
        lo = np.percentile(d, [2.5, 97.5], axis=0); gd = np.percentile(d[:, 1] - d[:, 0], [2.5, 97.5])
        rows.append(dict(extractor=e, n_configs=len(point[e]),
                         lopo=pt[0], lopo_lo=lo[0, 0], lopo_hi=lo[1, 0],
                         image=pt[1], image_lo=lo[0, 1], image_hi=lo[1, 1],
                         gap=pt[1] - pt[0], gap_lo=gd[0], gap_hi=gd[1]))
    return pd.DataFrame(rows)


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    pats = [("1", 0), ("2", 0), ("3", 0), ("4", 0), ("5", 0), ("6", 1), ("7", 1), ("8", 1), ("9", 2), ("10", 2)]
    rows = []
    for ext in ("A", "B"):
        for mag in ("10x", "100x"):
            for p, c in pats:
                for _ in range(20):
                    rows.append(dict(extractor=ext, preproc="raw", magnification=mag, patient_id=p, y=c,
                                     lopo_pred=rng.integers(0, 3), img_pred=c if rng.random() < 0.8 else rng.integers(0, 3)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?"); ap.add_argument("--B", type=int, default=2000); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    df = synthetic() if a.selftest else pd.read_csv(a.csv)
    res = run(df, a.B)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    if a.selftest:
        print("[SELFTEST] synthetic data; image-level was built to leak, so its gap should be large.")
