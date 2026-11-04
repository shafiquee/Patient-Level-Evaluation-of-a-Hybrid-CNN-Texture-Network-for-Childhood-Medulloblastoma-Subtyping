"""Paste these into your notebooks to produce the three things the report still needs.
Nothing here runs on its own.

1) medulloblastoma_dsaf_net_final.ipynb, after the Gradio/Grad-CAM cells (final_model exists):
   writes figures/gradcam_example.png (original | preprocessed | Grad-CAM++ overlay).
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_gradcam_figure(filepath, out_path="figures/gradcam_example.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    proc = preprocess_image(img_rgb)
    tex = extract_texture_features(proc)
    batch = np.expand_dims(proc.astype(np.float32), 0)
    probs = final_model.predict([batch, np.expand_dims(tex, 0)], verbose=0)[0]
    k = int(np.argmax(probs))
    heat = make_gradcam_heatmap(batch, np.expand_dims(tex, 0), final_model, pred_index=k, plus_plus=True)
    overlay = overlay_heatmap(proc, heat)
    fig, ax = plt.subplots(1, 3, figsize=(7.5, 2.7))
    for a, im, t in zip(ax, [img_rgb, proc, overlay],
                        ["Original", "Preprocessed", f"Grad-CAM++: {CLASS_NAMES[k]} ({probs[k]:.0%})"]):
        a.imshow(im); a.set_title(t, fontsize=8); a.axis("off")
    plt.tight_layout(); plt.savefig(out_path, dpi=300); plt.close()

# save_gradcam_figure(file_df.iloc[0]["filepath"])      # pick a tile you want to show

"""
2) mb_frozen_embedding_lopo.ipynb, right after `null, p = permutation_test(...)` in section 6.
   Lets make_figures.py draw the permutation-null histogram (python scripts/make_figures.py --null outputs/permutation_null.npy):

    np.save(os.path.join(OUTPUT_DIR, "permutation_null.npy"), null)

3) Gradio screenshot: run `demo.launch()`, upload one tile, screenshot the page, and save it as
   figures/gradio_app.png (PNG, any width).
"""


# ---------------------------------------------------------------------------
# Snippets for the evaluation scripts added in the IEEE version
# ---------------------------------------------------------------------------
def save_lopo_predictions(path, patient_ids, y_true_names, prob_matrix, class_names):
    """Call once per LOPO fold (append) or once after collecting every fold.

    patient_ids   : list/array, one id per held-out tile
    y_true_names  : list/array of class names (strings) per tile
    prob_matrix   : (n_tiles, n_classes) softmax outputs of the fold's model on its held-out tiles
    class_names   : list of class names in the same column order as prob_matrix
    Writes the CSV that scripts/evaluate_predictions.py reads. Contains no images and no patient
    identifiers beyond the study's slide numbers; check before pushing it to a public repo.
    """
    import pandas as pd
    df = pd.DataFrame(prob_matrix, columns=[f"p_{c}" for c in class_names])
    df.insert(0, "y_true", list(y_true_names))
    df.insert(0, "patient_id", list(patient_ids))
    df.to_csv(path, index=False)


def export_texture_features(path, patient_ids, y_true_names, texture_matrix, feature_names=None):
    """Save the 34-element GLCM+LBP vectors per tile for scripts/shap_texture.py."""
    import pandas as pd
    n = texture_matrix.shape[1]
    names = feature_names or [f"f_{i}" for i in range(n)]
    assert all(c.startswith("f_") for c in names), "feature column names must start with 'f_'"
    df = pd.DataFrame(texture_matrix, columns=names)
    df.insert(0, "y_true", list(y_true_names))
    df.insert(0, "patient_id", list(patient_ids))
    df.to_csv(path, index=False)
