# Childhood medulloblastoma histopathology: patient-level evaluation of DSAF-Net

Code accompanying the paper "Patient-Level Evaluation of a Hybrid CNN-Texture Network for Childhood
Medulloblastoma Subtyping on an 11-Patient H\&E Histopathology Dataset".

**Research prototype. Not a diagnostic tool.**

## What is here

|Path|Purpose|
|-|-|
|`notebooks/medulloblastoma\_dsaf\_net\_final.ipynb`|DSAF-Net training, LOPO evaluation, Grad-CAM++, export, Gradio app|
|`notebooks/mb\_frozen\_embedding\_lopo.ipynb`|78 frozen-feature configurations, leakage gap, permutation test|
|`notebooks/das2019\_reproduction.ipynb`|Reproduction of the normal-vs-abnormal protocol of Das et al. 2019|
|`notebooks/pathfoundation\_colab.ipynb`|Path Foundation embedding extraction (Colab)|
|`scripts/gap\_ci.py`|Bootstrap 95% CIs for the leakage gap (image-level minus LOPO balanced tile accuracy) per frozen-feature extractor|
|`scripts/make\_figures.py`|Paper figures from the recorded results|
|`scripts/evaluate\_predictions.py`|AUROC, AUPRC, sens., spec., F1, Brier, ECE with patient-cluster bootstrap CIs; ROC/PR/calibration plots|
|`scripts/patient\_level\_ci.py`|Clopper-Pearson and bootstrap CIs for the patient-level results|
|`scripts/shap\_texture.py`|SHAP for the texture stream under leave-one-patient-out|
|`scripts/notebook\_snippets.py`|Helpers to export per-tile predictions and texture features from the notebooks|

## Data (not included)

The images are NOT redistributed. Download them from IEEE DataPort (doi 10.21227/w0m0-mw21), check the licence
shown on that page, and set the data path at the top of each notebook (the notebooks contain Windows
paths from the author's machine; edit them).

## Reproduce

```
pip install -r requirements.txt
python scripts/make\_figures.py
python scripts/patient\_level\_ci.py
python scripts/evaluate\_predictions.py --selftest      # synthetic data, checks the code only
python scripts/evaluate\_predictions.py outputs/lopo\_predictions.csv --out outputs/eval
python scripts/shap\_texture.py outputs/texture\_features.csv --out outputs/shap
```

`lopo\_predictions.csv` comes from `save\_lopo\_predictions` and
`export\_texture\_features` in `scripts/notebook\_snippets.py`; the notebooks as delivered did not save them.

## Known limits

11 patients; Normal has no patient id; one seed per arm; see the paper's limitations section.

