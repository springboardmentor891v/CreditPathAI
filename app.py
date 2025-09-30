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

# Load Loan.txt as main dataset
DATA_FILE = DATA_DIR / "Loan.txt"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE, sep=None, engine="python")
    return df

df = load_data()

# Clean data (drop IDs, dates not useful for modeling)
drop_cols = ["loanId", "memberId", "date"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Target
target = "loanStatus"
all_features = [c for c in df.columns if c != target]

# Encode target
le = LabelEncoder()
df[target] = le.fit_transform(df[target].astype(str))  # Current=0, Default=1

# Train/Test Split
X = df[all_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessor
numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []
if numeric_cols:
    transformers.append(("num", numeric_transformer, numeric_cols))
if categorical_cols:
    transformers.append(("cat", categorical_transformer, categorical_cols))

preprocessor = ColumnTransformer(transformers=transformers)

# Sidebar Loan Form
st.sidebar.title("Loan Application Form")

# Model choice
model_choice = st.sidebar.selectbox("Choose Model", ["Random Forest", "Logistic Regression"])

# Build model based on choice
if model_choice == "Random Forest":
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
else:
    model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
        class_weight="balanced"
    )

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train model
pipeline.fit(X_train, y_train)

# Inputs for user (exclude loanStatus!)
purpose = st.sidebar.selectbox("Purpose", df["purpose"].unique())
is_joint = st.sidebar.selectbox("Joint Application", [0, 1])
loan_amount = st.sidebar.number_input("Loan Amount", min_value=1000, max_value=50000, step=500)
term = st.sidebar.selectbox("Term", df["term"].unique())
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=30.0, step=0.1)
monthly_payment = st.sidebar.number_input("Monthly Payment", min_value=50, max_value=2000, step=10)
grade = st.sidebar.selectbox("Grade", df["grade"].unique())

if st.sidebar.button("Predict Loan Default"):
    input_df = pd.DataFrame([{
        "purpose": purpose,
        "isJointApplication": is_joint,
        "loanAmount": loan_amount,
        "term": term,
        "interestRate": interest_rate,
        "monthlyPayment": monthly_payment,
        "grade": grade
    }])

    pred = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]  # probability of default

    result = "Defaulter" if pred == 1 else "Current"
    st.sidebar.success(f"Prediction: {result}")
    st.sidebar.write(f"Probability of Default: {prob*100:.2f}%")

# Main Panel
st.title("CreditPathAI - Quick UI")
st.subheader("Dataset Summary")

st.write("Shape:", df.shape)

# Show sample
with st.expander("Show data (first 100 rows)"):
    st.dataframe(df.head(100))

# Model Evaluation
st.markdown("## Model Performance on Test Data")
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

acc = metrics.accuracy_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)

#st.write(f"Accuracy: {acc:.4f}")
#st.write(f"F1 Score: {f1:.4f}")
#st.write(f"Precision: {precision:.4f}")
#st.write(f"Recall: {recall:.4f}")

cm = metrics.confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
auc = metrics.auc(fpr, tpr)
st.write(f"ROC AUC: {auc:.4f}")
fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
ax2.plot([0, 1], [0, 1], "--", color="gray")
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve")
ax2.legend(loc="lower right")
st.pyplot(fig2)

# Feature Importance (only for Random Forest)
if model_choice == "Random Forest":
    st.markdown("## Feature Importance (Random Forest)")
    importances = pipeline.named_steps["model"].feature_importances_

    feature_names = []
    for name, trans, cols in pipeline.named_steps["preprocessor"].transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            try:
                ohe = trans.named_steps["onehot"]
                feature_names.extend(ohe.get_feature_names_out(cols))
            except Exception:
                feature_names.extend(cols)

    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    fig3, ax3 = plt.subplots()
    feat_imp.head(20).plot(kind="barh", ax=ax3)
    ax3.invert_yaxis()
    st.pyplot(fig3)
