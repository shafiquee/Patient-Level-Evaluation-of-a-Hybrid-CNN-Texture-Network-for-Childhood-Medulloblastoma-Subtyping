#!/usr/bin/env python3
"""Generate the result figures for the medulloblastoma report.

Usage (from the project root):
    python scripts/make_figures.py                 # writes figures/*.pdf and *.png
    python scripts/make_figures.py --null outputs/permutation_null.npy
        (optional: adds the permutation-null histogram if you saved the array)

Every number below is copied from notebook outputs:
  das2019_reproduction.ipynb          -> DAS_* tables
  mb_frozen_embedding_lopo.ipynb      -> LEAK, ABLATION, PERM
  medulloblastoma_dsaf_net_final.ipynb-> VOTES
If you rerun a notebook with a different seed or data, update the dictionaries here
(or point --csv at outputs/frozen_embedding_lopo_results.csv to rebuild LEAK/ABLATION).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

# ---- style ---------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1-3
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.bbox": "tight", "pdf.fonttype": 42,
})
OUT = "figures"
os.makedirs(OUT, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=300)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# Figure 1. Leakage gap (frozen features, tumour-subtype task)
# Mean over all preprocessing/magnification configurations per extractor.
# ---------------------------------------------------------------------------
LEAK = {  # extractor: (patient-level LOPO tile bal. acc., image-level 5-fold tile bal. acc.)
    "Path Foundation":    (0.189, 0.891),
    "EfficientNetV2-S":   (0.258, 0.797),
    "GLCM + LBP texture": (0.175, 0.641),
    "LAB colour stats":   (0.214, 0.599),
}


def fig_leakage():
    names = list(LEAK)
    pat = np.array([LEAK[n][0] for n in names])
    img = np.array([LEAK[n][1] for n in names])
    y = np.arange(len(names))[::-1]
    h = 0.34
    fig, ax = plt.subplots(figsize=(6.2, 2.7))
    ax.barh(y + h / 2 + 0.01, img, h, color=ORANGE, label="Image-level 5-fold CV (tiles from one slide on both sides)")
    ax.barh(y - h / 2 - 0.01, pat, h, color=BLUE, label="Patient-level leave-one-patient-out")
    for yi, a, b in zip(y, img, pat):
        ax.text(a + 0.012, yi + h / 2 + 0.01, f"{a:.2f}", va="center", fontsize=8, color=INK)
        ax.text(b + 0.012, yi - h / 2 - 0.01, f"{b:.2f}", va="center", fontsize=8, color=INK)
        ax.text(1.02, yi, f"gap {a - b:.2f}", va="center", fontsize=8, color=MUTED)
    ax.axvline(1 / 3, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(1 / 3, len(names) - 0.42, "chance (3 classes)", fontsize=7.5, color=MUTED, ha="center", va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Balanced tile accuracy, tumour subtype (mean over all settings)")
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), frameon=False, ncol=1, fontsize=7.8)
    save(fig, "fig_leakage_gap")


# ---------------------------------------------------------------------------
# Figure 2. Das et al. protocol reproduction (normal vs abnormal, all 5 texture families)
# ---------------------------------------------------------------------------
CLFS = ["Tree", "LD", "QD", "LR", "SVM", "KNN"]
DAS_PAPER = [100.0, 97.2, 99.1, 95.4, 100.0, 100.0]      # Table 1 of Das et al. 2019
DAS_IMG = [98.2, 97.5, 98.5, 98.5, 98.8, 96.8]           # image-level 5-fold CV, this work
DAS_PAT = [96.5, 98.0, 73.0, 97.5, 98.0, 97.0]           # tumour patients held out, balanced acc.


def fig_das():
    y = np.arange(len(CLFS))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for yi in y:
        ax.axhline(yi, color=GRID, lw=0.5, zorder=0)
    ax.scatter(DAS_PAPER, y, s=46, marker="D", facecolor="white", edgecolor=MUTED, lw=1.2, zorder=3,
               label="Reported by Das et al. (image-level)")
    ax.scatter(DAS_IMG, y, s=46, color=ORANGE, zorder=3, edgecolor="white", lw=0.8,
               label="This work, image-level 5-fold CV")
    ax.scatter(DAS_PAT, y, s=46, color=BLUE, zorder=3, edgecolor="white", lw=0.8,
               label="This work, tumour patients held out")
    ax.annotate("QD collapses on\nnew patients (73.0)", xy=(73.0, y[2]), xytext=(80.2, y[2] - 0.55),
                fontsize=7.5, color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7), va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(CLFS)
    ax.set_xlim(68, 101.5)
    ax.set_xlabel("Accuracy / balanced accuracy (%), normal vs abnormal, 10x images")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), frameon=False, ncol=1, fontsize=7.8)
    save(fig, "fig_das_reproduction")


# ---------------------------------------------------------------------------
# Figure 3. Preprocessing x magnification heat map (patient-level balanced accuracy)
# Colour is centred on the majority-patient baseline (0.333).
# ---------------------------------------------------------------------------
PRE = ["raw", "stain", "clahe", "stain+clahe"]
ABLATION = {  # (extractor, magnification): [raw, stain, clahe, stain+clahe]
    ("EfficientNetV2-S", "10x"):   [0.200, 0.244, 0.200, 0.200],
    ("EfficientNetV2-S", "100x"):  [0.200, 0.267, 0.133, 0.333],
    ("EfficientNetV2-S", "both"):  [0.333, 0.378, 0.378, 0.333],
    ("Path Foundation", "10x"):    [0.133, 0.133, 0.133, 0.067],
    ("Path Foundation", "100x"):   [0.267, 0.433, 0.267, 0.267],
    ("Path Foundation", "both"):   [0.200, 0.367, 0.200, 0.267],
    ("GLCM + LBP texture", "10x"): [0.067, 0.133, 0.133, 0.067],
    ("GLCM + LBP texture", "100x"):[0.133, 0.133, 0.133, 0.067],
    ("GLCM + LBP texture", "both"):[0.200, 0.422, 0.422, 0.422],
}
COLOR_STATS = {"10x": 0.133, "100x": 0.422, "both": 0.200}   # raw only, preprocessing grid does not apply
BASELINE = 1 / 3


def fig_ablation():
    rows = list(ABLATION) + [("LAB colour stats", m) for m in COLOR_STATS]
    data = np.full((len(rows), 4), np.nan)
    for i, r in enumerate(rows):
        if r in ABLATION:
            data[i] = ABLATION[r]
        else:
            data[i, 0] = COLOR_STATS[r[1]]
    cmap = LinearSegmentedColormap.from_list("div", [ORANGE, "#f1f0ec", BLUE])
    norm = TwoSlopeNorm(vmin=0.0, vcenter=BASELINE, vmax=0.667)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.imshow(np.ma.masked_invalid(data), cmap=cmap, norm=norm, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(4):
            if np.isnan(data[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5, color=MUTED)
            else:
                v = data[i, j]
                dark = abs(v - BASELINE) > 0.22
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if dark else INK)
    ax.set_xticks(range(4))
    ax.set_xticklabels(PRE)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{e}, {m}" for e, m in rows], fontsize=8)
    for k in (2.5, 5.5, 8.5):
        ax.axhline(k, color="white", lw=2.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.04, pad=0.03)
    cb.set_ticks([0, BASELINE, 0.667])
    cb.set_ticklabels(["0.00", "0.33\nbaseline", "0.67"])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5)
    ax.set_xlabel("Patient-level balanced accuracy, tumour subtype, 10 patients", labelpad=8)
    save(fig, "fig_preprocessing_heatmap")


# ---------------------------------------------------------------------------
# Figure 4. Per-patient calls (majority vote) for each method
# ---------------------------------------------------------------------------
PATIENTS = [("1766", "C"), ("188", "C"), ("277", "C"), ("279", "C"), ("321", "C"),
            ("141", "D"), ("203", "D"), ("411", "D"), ("17126", "L"), ("395", "L")]
CLASSNAME = {"C": "Classic", "D": "Desm./Nodular", "L": "Large cell"}
VOTES = {  # method: predicted class per patient, same order as PATIENTS
    "Always\nClassic":                  ["C"] * 10,
    "Frozen\nEffNetV2-S\n+ LR":          ["C", "C", "C", "C", "C", "C", "C", "C", "D", "C"],
    "DSAF-Net\nno texture":             ["C", "C", "D", "C", "D", "C", "C", "C", "D", "C"],
    "DSAF-Net\ntexture":                ["C", "C", "C", "C", "D", "C", "D", "C", "D", "C"],
    "DSAF-Net\ntexture\n+ fine-tune":    ["C", "C", "C", "C", "C", "C", "D", "C", "D", "C"],
    "Path Found.\nhead\n+ texture":      ["C", "C", "D", "C", "D", "C", "C", "C", "D", "C"],
}


def fig_patients():
    methods = list(VOTES)
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    for j, m in enumerate(methods):
        for i, (pid, truth) in enumerate(PATIENTS):
            pred = VOTES[m][i]
            ok = pred == truth
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92,
                                       facecolor=BLUE if ok else "#ecebe6", edgecolor="none"))
            ax.text(j, i, pred, ha="center", va="center", fontsize=9,
                    color="white" if ok else INK, fontweight="bold" if ok else "normal")
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_ylim(len(PATIENTS) - 0.5, -0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=7.6)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(PATIENTS)))
    ax.set_yticklabels([f"{p}  ({t})" for p, t in PATIENTS], fontsize=8)
    ax.tick_params(length=0)
    for k in (4.5, 7.5):
        ax.axhline(k, color="white", lw=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(handles=[Patch(facecolor=BLUE, label="Correct patient-level call"),
                       Patch(facecolor="#ecebe6", label="Wrong call")],
              loc="upper center", bbox_to_anchor=(0.45, -0.03), ncol=2, frameon=False, fontsize=7.8)
    save(fig, "fig_patient_calls")


# ---------------------------------------------------------------------------
# Optional: permutation null histogram, if the array was saved from the notebook
# ---------------------------------------------------------------------------
def fig_null(path):
    null = np.load(path)
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    ax.hist(null, bins=np.arange(0, 0.78, 0.0556), color="#c9c8c2", edgecolor="white")
    ax.axvline(BASELINE, color=BLUE, lw=1.6)
    ax.text(BASELINE + 0.01, ax.get_ylim()[1] * 0.92, "observed 0.333", color=INK, fontsize=8)
    ax.set_xlabel("Patient-level balanced accuracy under shuffled patient labels")
    ax.set_ylabel("Runs")
    save(fig, "fig_permutation_null")



# ---------------------------------------------------------------------------
# IEEE two-column versions (3.4 in wide, 6-7 pt text) used in main.tex
# ---------------------------------------------------------------------------
def fig_leakage_col():
    names = ["Path Foundation", "EfficientNetV2-S", "GLCM + LBP", "LAB colour stats"]
    keys = list(LEAK)
    pat = np.array([LEAK[k][0] for k in keys])
    img = np.array([LEAK[k][1] for k in keys])
    y = np.arange(len(names))[::-1]
    h = 0.36
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    ax.barh(y + h / 2, img, h, color=ORANGE, label="Image-level 5-fold CV")
    ax.barh(y - h / 2, pat, h, color=BLUE, label="Leave-one-patient-out")
    for yi, a, b in zip(y, img, pat):
        ax.text(a + 0.01, yi + h / 2, f"{a:.2f}", va="center", fontsize=6.5, color=INK)
        ax.text(b + 0.01, yi - h / 2, f"{b:.2f}", va="center", fontsize=6.5, color=INK)
    ax.axvline(1 / 3, color=MUTED, lw=0.7, ls=(0, (3, 3)))
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Balanced tile accuracy, tumour subtype", fontsize=7)
    ax.tick_params(axis="x", labelsize=6.5)
    ax.xaxis.grid(True, color=GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.5)
    save(fig, "fig_leakage_gap_col")


def fig_patients_col():
    labels = ["Always\nClassic", "Frozen\nEffNet", "DSAF\nno tex.", "DSAF\ntex.", "DSAF\ntex.+FT", "Path\nFound."]
    methods = list(VOTES)
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    for j, m in enumerate(methods):
        for i, (pid, truth) in enumerate(PATIENTS):
            pred = VOTES[m][i]
            ok = pred == truth
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.46), 0.92, 0.92,
                                       facecolor=BLUE if ok else "#ecebe6", edgecolor="none"))
            ax.text(j, i, pred, ha="center", va="center", fontsize=7,
                    color="white" if ok else INK, fontweight="bold" if ok else "normal")
    ax.set_xlim(-0.5, len(methods) - 0.5)
    ax.set_ylim(len(PATIENTS) - 0.5, -0.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(PATIENTS)))
    ax.set_yticklabels([f"{p} ({t})" for p, t in PATIENTS], fontsize=6.5)
    ax.tick_params(length=0)
    for k in (4.5, 7.5):
        ax.axhline(k, color="white", lw=2.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[Patch(facecolor=BLUE, label="Correct"), Patch(facecolor="#ecebe6", label="Wrong")],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=6.5)
    save(fig, "fig_patient_calls_col")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--null", default=None, help="optional .npy with the permutation-null accuracies")
    args = ap.parse_args()
    fig_leakage()
    fig_das()
    fig_ablation()
    fig_patients()
    fig_leakage_col()
    fig_patients_col()
    if args.null and os.path.exists(args.null):
        fig_null(args.null)
