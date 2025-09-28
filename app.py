# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CreditPathAI - Quick UI", layout="wide")

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

PREPACKAGED = {
    "Loan_default.csv": DATA_DIR / "Loan_default.csv",
    "microsoft_loan.csv": DATA_DIR / "microsoft_loan.csv",
    "Loan.txt": DATA_DIR / "Loan.txt",
}

def load_prepackaged(name):
    p = PREPACKAGED.get(name)
    if p and p.exists():
        try:
            if p.suffix == ".txt":
                return pd.read_csv(p, sep=None, engine="python")  # attempt autodetect
            return pd.read_csv(p)
        except Exception as e:
            st.error(f"Failed to load {p.name}: {e}")
            return None
    else:
        st.warning(f"{name} not found in data/ folder.")
        return None

# Sidebar controls
st.sidebar.title("Load data / model")
dataset_choice = st.sidebar.selectbox(
    "Choose dataset",
    ["Loan_default.csv", "microsoft_loan.csv", "Loan.txt", "Upload my own CSV"],
)

if dataset_choice == "Upload my own CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error("Can't read uploaded file: " + str(e))
            df = None
    else:
        df = None
else:
    df = load_prepackaged(dataset_choice)

if df is None:
    st.title("CreditPathAI - Quick UI")
    st.write("Please select a dataset or upload a CSV on the left.")
    st.stop()

# Main app
st.title("CreditPathAI - Quick UI")
st.subheader(f"Dataset: {dataset_choice} — shape: {df.shape}")

# Show sample
with st.expander("Show data (first 100 rows)"):
    st.dataframe(df.head(100))

# Basic summary
st.markdown("### Basic summary")
col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    st.write("Rows, Columns")
    st.write(df.shape)
with col2:
    st.write("Missing values (%) per column")
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    st.dataframe(miss.to_frame("missing_pct"))
with col3:
    st.write("Column types")
    st.dataframe(df.dtypes.astype(str).to_frame("dtype"))

# Simple EDA
st.markdown("### Quick plots")
plot_type = st.selectbox("Plot type", ["Histogram (numeric)", "Bar (categorical)", "Correlation heatmap", "Boxplot (numeric)"])
if plot_type == "Histogram (numeric)":
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        col = st.selectbox("Numeric column", num_cols)
        bins = st.slider("Bins", 5, 100, 30)
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_xlabel(col)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found.")
elif plot_type == "Bar (categorical)":
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if cat_cols:
        col = st.selectbox("Categorical column", cat_cols)
        top_n = st.slider("Top categories to show", 1, 30, 10)
        vc = df[col].fillna("MISSING").value_counts().head(top_n)
        fig, ax = plt.subplots()
        vc.plot(kind="bar", ax=ax)
        ax.set_ylabel("count")
        st.pyplot(fig)
    else:
        st.info("No categorical columns found.")
elif plot_type == "Correlation heatmap":
    num = df.select_dtypes(include=["number"])
    if not num.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(num.corr(), annot=True, fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No numeric columns to correlate.")
elif plot_type == "Boxplot (numeric)":
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        col = st.selectbox("Numeric column for boxplot", num_cols, key="box")
        fig, ax = plt.subplots()
        ax.boxplot(df[col].dropna())
        ax.set_title(col)
        st.pyplot(fig)
    else:
        st.info("No numeric columns found.")

# Modeling panel
st.markdown("## Quick modeling")
target = st.selectbox("Select target column (what you want to predict)", df.columns.tolist())
all_features = [c for c in df.columns if c != target]
features = st.multiselect("Features (choose columns to use as X)", options=all_features, default=all_features)

if not features:
    st.warning("Select at least one feature to train a model.")
    st.stop()

test_size = st.slider("Test set fraction", 0.05, 0.5, 0.2)
model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])
random_state = st.number_input("Random state (seed)", value=42, step=1)
if model_choice == "Random Forest":
    n_estimators = st.slider("n_estimators (trees)", 10, 1000, 100)
else:
    C = st.number_input("Logistic Regression C (inverse regularization)", value=1.0, format="%.4f")

train_btn = st.button("Train model")

if train_btn:
    with st.spinner("Training..."):
        sub = df[features + [target]].copy()

        # Handle target encoding if non-numeric
        y = sub[target]
        if y.dtype == "object" or y.dtype.name == "category":
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            label_map = dict(enumerate(le.classes_))
        else:
            label_map = None
            y = y.values

        X = sub[features].copy()

        # train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state),
            stratify=(y if len(np.unique(y)) > 1 else None)
        )

        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))

        ])

        transformers = []
        if numeric_cols:
            transformers.append(("num", numeric_transformer, numeric_cols))
        if categorical_cols:
            transformers.append(("cat", categorical_transformer, categorical_cols))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.3
        )


        # model
        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state))
        else:
            model = LogisticRegression(max_iter=1000, C=C, random_state=int(random_state))

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

        # Fit
        pipeline.fit(X_train, y_train)

        # Predict & evaluate
        y_pred = pipeline.predict(X_test)
        try:
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps["model"], "predict_proba") else None
        except Exception:
            y_proba = None

        acc = metrics.accuracy_score(y_test, y_pred)
        st.success(f"Accuracy: {acc:.4f}")
        st.text("Classification report:")
        st.text(metrics.classification_report(y_test, y_pred, zero_division=0))

        # confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # ROC if binary and proba available
        if len(np.unique(y_test)) == 2 and y_proba is not None:
            fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
            auc = metrics.auc(fpr, tpr)
            st.write(f"ROC AUC: {auc:.4f}")
            fig2, ax2 = plt.subplots()
            ax2.plot(fpr, tpr)
            ax2.plot([0, 1], [0, 1], "--")
            ax2.set_xlabel("FPR")
            ax2.set_ylabel("TPR")
            ax2.set_title("ROC Curve")
            st.pyplot(fig2)

        # Feature importance for RandomForest
        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            st.write("Feature importances (approximate)")
            try:
                ohe = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
                cat_feature_names = list(ohe.get_feature_names_out(categorical_cols)) if categorical_cols else []
            except Exception:
                cat_feature_names = []
            feature_names = list(numeric_cols) + cat_feature_names
            importances = pipeline.named_steps["model"].feature_importances_
            fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(30)
            st.dataframe(fi.to_frame("importance"))
        st.balloons()
