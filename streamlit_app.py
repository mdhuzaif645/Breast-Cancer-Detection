from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

BASE = Path(__file__).parent
MODEL_PATH = BASE / "easy_model.joblib"
CSV_PATH = BASE / "Breast_cancer_data.csv"

st.set_page_config(
    page_title="Breast Cancer Prediction Studio",
    page_icon="🎀",
    layout="wide",
)

st.title("🎀 Breast Cancer Prediction Studio")
st.caption(
    "An educational machine-learning demo using Logistic Regression. "
    "It is not a medical device and must not be used to diagnose or make treatment decisions."
)

COLS = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
]
PRETTY = {
    "mean_radius": "Mean radius",
    "mean_texture": "Mean texture",
    "mean_perimeter": "Mean perimeter",
    "mean_area": "Mean area",
    "mean_smoothness": "Mean smoothness",
}
RANGES = {
    "mean_radius": (6.981, 28.11, 13.37),
    "mean_texture": (9.71, 39.28, 18.84),
    "mean_perimeter": (43.79, 188.5, 86.24),
    "mean_area": (143.5, 2501.0, 551.1),
    "mean_smoothness": (0.0526, 0.1634, 0.0959),
}

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "easy_model.joblib was not found. Place it in the same folder as streamlit_app.py."
        )
    scaler, model = joblib.load(MODEL_PATH)
    return scaler, model

@st.cache_data
def load_dataset():
    if not CSV_PATH.exists():
        return None
    try:
        data = pd.read_csv(CSV_PATH)
        if not all(c in data.columns for c in COLS) or "diagnosis" not in data.columns:
            return None
        return data
    except Exception:
        return None

def malignant_probability(X, scaler, model):
    # Matches the class convention used in the original Gradio app:
    # class 1 is benign, so 1 - P(class 1) is the malignant probability.
    return float(1.0 - model.predict_proba(scaler.transform(X[COLS]))[0, 1])

def make_input(values):
    return pd.DataFrame([values], columns=COLS)

def risk_band(probability):
    if probability < 0.25:
        return "Low"
    if probability < 0.50:
        return "Borderline"
    if probability < 0.75:
        return "Elevated"
    return "High"

try:
    scaler, model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

data = load_dataset()

with st.sidebar:
    st.header("Tumour measurements")
    st.write("Adjust the values, then click **Analyze**.")
    values = {}
    for col in COLS:
        low, high, median = RANGES[col]
        if col == "mean_smoothness":
            values[col] = st.slider(
                PRETTY[col], float(low), float(high), float(median), step=0.0001,
                format="%.4f",
            )
        elif col == "mean_area":
            values[col] = st.slider(
                PRETTY[col], float(low), float(high), float(median), step=0.1,
            )
        else:
            values[col] = st.slider(
                PRETTY[col], float(low), float(high), float(median), step=0.01,
            )
    threshold = st.slider(
        "Decision threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="The app flags a malignant pattern when the estimated probability is at least this threshold.",
    )
    analyze = st.button("Analyze", type="primary", use_container_width=True)

if "analyzed" not in st.session_state:
    st.session_state.analyzed = True
if analyze:
    st.session_state.analyzed = True

if st.session_state.analyzed:
    X = make_input([values[c] for c in COLS])
    probability = malignant_probability(X, scaler, model)
    prediction = "Malignant pattern" if probability >= threshold else "Benign pattern"
    confidence = probability if probability >= threshold else 1 - probability
    band = risk_band(probability)

    st.subheader("Prediction")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model verdict", prediction)
    c2.metric("Estimated P(malignant)", f"{probability:.1%}")
    c3.metric("Estimated P(benign)", f"{1 - probability:.1%}")
    c4.metric("Risk band", band)
    st.progress(float(np.clip(probability, 0.0, 1.0)), text="Estimated probability of malignant class")

    outside = [
        PRETTY[c] for c in COLS
        if not RANGES[c][0] <= values[c] <= RANGES[c][1]
    ]
    if outside:
        st.warning(
            "One or more measurements are outside the reference ranges: "
            + ", ".join(outside)
            + ". Treat the output with extra caution."
        )

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("Input measurements")
        st.dataframe(
            pd.DataFrame(
                {"Feature": [PRETTY[c] for c in COLS],
                 "Value": [values[c] for c in COLS]}
            ),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.subheader("Feature contributions")
        try:
            z = scaler.transform(X[COLS])[0]
            # Same interpretation as the original app: negative model coefficients
            # are the direction that pushes toward malignant for this class convention.
            contributions = -model.coef_[0] * z
            contrib_df = pd.DataFrame({
                "Feature": [PRETTY[c] for c in COLS],
                "Contribution": contributions,
            }).sort_values("Contribution")
            fig, ax = plt.subplots(figsize=(7, 3.6))
            bar_colors = [
                "#d6336c" if x > 0 else "#2f80c0"
                for x in contrib_df["Contribution"]
            ]
            ax.barh(contrib_df["Feature"], contrib_df["Contribution"], color=bar_colors)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Relative contribution (model-specific)")
            ax.set_title("Scaled feature contributions")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("These are model-specific contributions, not clinical explanations.")
        except Exception as exc:
            st.info(f"Feature contributions could not be displayed: {exc}")

    st.divider()
    st.subheader("What-if analysis")
    st.write("Each chart varies one measurement while holding the other four at their current values.")
    whatif_cols = st.columns(2)
    for idx, col in enumerate(COLS):
        low, high, _ = RANGES[col]
        grid = np.linspace(low, high, 80)
        rows = pd.DataFrame([X.iloc[0].values] * len(grid), columns=COLS)
        rows[col] = grid
        probs = 1.0 - model.predict_proba(scaler.transform(rows[COLS]))[:, 1]
        fig, ax = plt.subplots(figsize=(6, 2.7))
        ax.plot(grid, probs, linewidth=2, color="#d6336c")
        ax.axhline(threshold, linestyle="--", color="#555", label="Decision threshold")
        ax.scatter([values[col]], [probability], color="#222", zorder=3, label="Current input")
        ax.set_xlabel(PRETTY[col])
        ax.set_ylabel("P(malignant)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        with whatif_cols[idx % 2]:
            st.pyplot(fig)
        plt.close(fig)

    if data is not None:
        st.divider()
        st.subheader("Training-data distributions")
        st.caption("The dashed line marks the current input. These charts require Breast_cancer_data.csv.")
        chart_cols = st.columns(2)
        for idx, col in enumerate(COLS):
            fig, ax = plt.subplots(figsize=(6, 3))
            # Original app's dataset convention: diagnosis 0 = malignant, 1 = benign.
            for cls, label, color in [
                (0, "Malignant (dataset label 0)", "#d6336c"),
                (1, "Benign (dataset label 1)", "#2f80c0"),
            ]:
                vals_cls = data.loc[data["diagnosis"] == cls, col].dropna()
                if not vals_cls.empty:
                    ax.hist(vals_cls, bins=25, density=True, alpha=0.35, color=color, label=label)
            ax.axvline(values[col], color="black", linestyle="--", label="Current input")
            ax.set_xlabel(PRETTY[col])
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)
            fig.tight_layout()
            with chart_cols[idx % 2]:
                st.pyplot(fig)
            plt.close(fig)
    else:
        st.info(
            "Optional: add Breast_cancer_data.csv beside streamlit_app.py to enable "
            "the training-data distribution charts."
        )

with st.expander("How the model works and limitations"):
    st.markdown(
        """
        1. The five measurements are transformed using the saved scaler.
        2. The trained Logistic Regression model estimates class probabilities.
        3. The selected threshold determines the displayed model verdict.

        **Important:** This is an educational software demonstration, not a medical device.
        Model probabilities are not a patient's actual clinical risk. Real diagnosis requires
        qualified medical professionals and appropriate clinical evaluation.
        """
    )
