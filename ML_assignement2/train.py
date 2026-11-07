"""
train.py - trains 5 classifiers on the Student Depression Detection dataset
(Kaggle, MKhoirul Mustaqim; originally Haque & Alom, Mendeley Data).
"""
import json
from pathlib import Path
 
import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
 
# ----------------------------------------------------------------- CONFIG ---
DATA_PATH = "Student_Depression_Dataset.csv"      # path to the Kaggle CSV
TARGET = "Depression"
FEATURES = [                               # the 12 independent variables
    "Gender", "Age", "Family_background", "Relationship_status",
    "Marital_status", "Living_style", "Current_semester", "CGPA",
    "friend_circle", "Health_condition", "Cost_without_study", "Home_division",
]
NUMERIC_COLS = None       
CATEGORICAL_COLS = None   
DROP_DUPLICATES = True     
TEST_SIZE = 0.20
RANDOM_STATE = 2026        
OUT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path(".")
# ----------------------------------------------------------------------------
 
MODEL_DIR = OUT_DIR / "model"
 
 
# ------------------------------------------------------------------ data ----
def load_data():
    """Read the CSV."""
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    print(f"Raw file: {df.shape[0]} rows x {df.shape[1]} columns")
 
    lookup = {c.lower(): c for c in df.columns}          # tolerant to case differences
    wanted = FEATURES + [TARGET]
    missing = [w for w in wanted if w.lower() not in lookup]
    if missing:
        raise SystemExit(f"Columns not found: {missing}\nColumns in file: {list(df.columns)}\n"
                         "Fix FEATURES / TARGET in the CONFIG block.")
    df = df.rename(columns={lookup[w.lower()]: w for w in wanted})
 
    dropped = [c for c in df.columns if c not in wanted]
    print(f"Ignored {len(dropped)} extra columns: {dropped}")
    df = df[wanted].dropna(subset=[TARGET]).reset_index(drop=True)
 
    n_dup = int(df.duplicated().sum())
    print(f"Exact duplicate rows (all {len(wanted)} columns): {n_dup}")
    if DROP_DUPLICATES and n_dup:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Dropped duplicates -> {len(df)} rows remain")
    return df

def encode_target(s):
    """Yes/No (or 1/0, True/False) -> 1/0."""
    if pd.api.types.is_numeric_dtype(s):
        out = s.astype(int)
        if not set(out.unique()) <= {0, 1}:
            raise SystemExit(f"Target must be binary, found values {sorted(out.unique())}")
        return out
    mapping = {"yes": 1, "y": 1, "true": 1, "1": 1, "no": 0, "n": 0, "false": 0, "0": 0}
    out = s.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any():
        raise SystemExit(f"Unrecognised target values: {sorted(s[out.isna()].astype(str).unique())}")
    return out.astype(int)
 
 
# ---------------------------------------------------------------- models ----
def make_preprocessor(num_cols, cat_cols, scale):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer([("num", Pipeline(num_steps), num_cols),
                              ("cat", Pipeline(cat_steps), cat_cols)])
 
 
def get_models():
    """name -> (file slug, classifier, needs feature scaling)."""
    return {
        "Logistic Regression": ("logistic_regression", LogisticRegression(max_iter=2000), True),
        # depth/leaf limits stop the tree memorising the training set (overfitting)
        "Decision Tree": ("decision_tree",
                          DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                                 random_state=RANDOM_STATE), False),
        "KNN": ("knn", KNeighborsClassifier(n_neighbors=7), True),
        "Naive Bayes": ("naive_bayes", GaussianNB(), True),
        "Support Vector Machine": ("svm", SVC(kernel="rbf", C=1.0, probability=True,
                                              random_state=RANDOM_STATE), True),
    }
 
 
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
 
 
def to_markdown_table(df):
    """Markdown table without needing the tabulate package."""
    cols = [df.index.name] + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, row in df.iterrows():
        lines.append("| " + " | ".join([str(name)] + [f"{v:.4f}" for v in row]) + " |")
    return "\n".join(lines) + "\n"
 
 
# ------------------------------------------------------------------ main ----
def main():
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_data()
 
    y_raw = df[TARGET]
    y = encode_target(y_raw)
    X = df[FEATURES]
 
    num_cols = NUMERIC_COLS or X.select_dtypes(include="number").columns.tolist()
    cat_cols = CATEGORICAL_COLS or [c for c in FEATURES if c not in num_cols]
    print(f"\nNumeric ({len(num_cols)}): {num_cols}\nCategorical ({len(cat_cols)}): {cat_cols}")
    miss = X.isna().sum()
    miss = miss[miss > 0]
    print("Missing values per feature:", "none" if miss.empty else "\n" + miss.to_string())
    print(f"Class balance (1 = depressed):\n{y.value_counts(normalize=True).round(3).to_string()}")
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    print(f"\nTrain: {len(X_train)} rows | Test: {len(X_test)} rows")
 
    rows, model_files = {}, {}
    for name, (slug, clf, scale) in get_models().items():
        pipe = Pipeline([("prep", make_preprocessor(num_cols, cat_cols, scale)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows[name] = compute_metrics(y_test, pred, prob)
        joblib.dump(pipe, MODEL_DIR / f"{slug}.joblib")
        model_files[name] = f"{slug}.joblib"
        print(f"  trained + saved: {name}")
 
    results = pd.DataFrame(rows).T.round(4)
    results.index.name = "ML Model Name"
    print("\n=== Test-set comparison ===")
    print(results.to_string())
    print(f"\nBest by MCC: {results['MCC'].idxmax()} | best by AUC: {results['AUC'].idxmax()}")
 
    # most influential features according to the tree (useful for README observations)
    tree = joblib.load(MODEL_DIR / "decision_tree.joblib")
    imp = pd.Series(tree.named_steps["clf"].feature_importances_,
                    index=tree.named_steps["prep"].get_feature_names_out())
    print("\nDecision-tree top 8 features:\n" + imp.sort_values(ascending=False).head(8).round(3).to_string())
 
    # ---- artefacts for the app / README / deployment ----
    test_df = X_test.copy()
    test_df[TARGET] = y_raw.loc[X_test.index]          # keep the original Yes/No labels
    test_df.to_csv(OUT_DIR / "test_data.csv", index=False)
    results.to_csv(OUT_DIR / "metrics.csv")
    (OUT_DIR / "results_table.md").write_text(to_markdown_table(results))
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "features": FEATURES, "target": TARGET, "numeric": num_cols, "categorical": cat_cols,
        "model_files": model_files, "sklearn_version": sklearn.__version__,
        "random_state": RANDOM_STATE, "test_size": TEST_SIZE,
        "n_train": len(X_train), "n_test": len(X_test)}, indent=2))
    (OUT_DIR / "requirements.txt").write_text(
        f"streamlit\nscikit-learn=={sklearn.__version__}\npandas\nnumpy\nmatplotlib\njoblib\n")
    print("\nSaved: model/*.joblib, model/meta.json, test_data.csv, metrics.csv, "
          "results_table.md, requirements.txt")
 if __name__ == "__main__":
    main()

