"""
Student Depression Screening - Model Lab
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
MODEL_DIR = BASE / "model"
SAMPLE_CSV = BASE / "test_data.csv"
CLASS_NAMES = ["No depression", "Depression"]

st.set_page_config(page_title="Student Depression Screening - Model Lab", layout="wide")


# ---------------------------------------------------------------- helpers ---
@st.cache_data
def load_meta():
    return json.loads((MODEL_DIR / "meta.json").read_text())


@st.cache_resource
def load_model(filename):
    return joblib.load(MODEL_DIR / filename)


def encode_target(s):
    """Yes/No (or 1/0, True/False) -> 1/0. Returns None if values are not recognised."""
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(float)
    else:
        mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
        out = s.astype(str).str.strip().str.lower().map(mapping)
    return out.astype("Int64") if not out.isna().any() and set(out.unique()) <= {0, 1} else None


def compute_metrics(y, pred, prob):
    try:
        auc = roc_auc_score(y, prob)
    except ValueError:                      # only one class present in the uploaded file
        auc = float("nan")
    return {"Accuracy": accuracy_score(y, pred), "AUC": auc,
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred)}


def confusion_figure(y, pred):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES, rotation=15)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    return fig, cm


# ---------------------------------------------------------------- sidebar ---
meta = load_meta()
st.title("Student Depression Screening - Model Lab")
st.caption("Compare five classifiers on survey data about student life. "
           "The labels are self-reported survey answers, **not a clinical diagnosis**, "
           "and this tool is for coursework, not for screening real people.")

with st.sidebar:
    st.header("1. Test data")
    upload = st.file_uploader("Upload a test CSV", type="csv")
    use_sample = st.checkbox("Use the bundled test_data.csv instead", value=upload is None)
    st.header("2. Model")
    model_name = st.selectbox("Classifier", list(meta["model_files"].keys()))
    st.header("3. Decision threshold")
    threshold = st.slider("Predict 'Depression' when probability >=", 0.05, 0.95, 0.50, 0.05,
                          help="AUC does not change with the threshold; the other metrics do.")
    st.markdown("**Required columns**")
    st.code("\n".join(meta["features"] + [meta["target"]]), language=None)

# ---------------------------------------------------------------- data in ---
if upload is not None:
    df = pd.read_csv(upload)
    source = f"uploaded file ({upload.name})"
elif use_sample and SAMPLE_CSV.exists():
    df = pd.read_csv(SAMPLE_CSV)
    source = "bundled test_data.csv"
else:
    st.info("Upload a test CSV in the sidebar (or tick the bundled sample) to begin.")
    st.stop()

df.columns = [c.strip() for c in df.columns]
needed = meta["features"] + [meta["target"]]
missing = [c for c in needed if c not in df.columns]
if missing:
    st.error(f"The CSV is missing required columns: {missing}")
    st.stop()

y = encode_target(df[meta["target"]])
if y is None:
    st.error(f"Column '{meta['target']}' must contain Yes/No (or 1/0) values only, with no blanks.")
    st.stop()
y = y.astype(int).to_numpy()
X = df[meta["features"]]
st.write(f"Evaluating on **{len(df)} rows** from the {source}. "
         f"Share of 'Depression' in this file: **{y.mean():.1%}**.")

# ------------------------------------------------------- selected model -----
pipe = load_model(meta["model_files"][model_name])
prob = pipe.predict_proba(X)[:, 1]
pred = (prob >= threshold).astype(int)
m = compute_metrics(y, pred, prob)

st.subheader(f"{model_name} - metrics at threshold {threshold:.2f}")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.3f}" if not np.isnan(v) else "n/a")

tab_cm, tab_roc, tab_all, tab_data = st.tabs(
    ["Confusion matrix & report", "ROC curve", "Compare all 5 models", "Data & predictions"])

with tab_cm:
    left, right = st.columns([1, 1.3])
    fig, cm = confusion_figure(y, pred)
    left.pyplot(fig)
    tn, fp, fn, tp = cm.ravel()
    left.caption(f"Missed cases (false negatives): **{fn}** | false alarms (false positives): **{fp}**")
    report = pd.DataFrame(classification_report(y, pred, labels=[0, 1], target_names=CLASS_NAMES,
                                                output_dict=True, zero_division=0)).T.round(3)
    right.dataframe(report)
    st.info("In a screening setting a missed student at risk costs more than a false alarm. "
            "Lower the threshold in the sidebar and watch recall rise while precision falls.")

with tab_roc:
    try:
        fpr, tpr, _ = roc_curve(y, prob)
        fig, ax = plt.subplots(figsize=(4.8, 4))
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {m['AUC']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
    except ValueError:
        st.warning("ROC needs both classes present in the uploaded file.")

with tab_all:
    rows = {}
    for name, fname in meta["model_files"].items():
        p = load_model(fname).predict_proba(X)[:, 1]
        rows[name] = compute_metrics(y, (p >= threshold).astype(int), p)
    table = pd.DataFrame(rows).T.round(4)
    table.index.name = "ML Model Name"
    st.dataframe(table.style.highlight_max(axis=0, color="#cde8cd"))
    st.caption(f"All five saved models scored on the same data at threshold {threshold:.2f}. "
               "Best value per metric is highlighted.")

with tab_data:
    out = df.copy()
    out["predicted_probability"] = prob.round(4)
    out["predicted_label"] = np.where(pred == 1, "Yes", "No")
    st.dataframe(out.head(200))
    st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv")
