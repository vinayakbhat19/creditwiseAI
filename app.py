from pathlib import Path
import time

import pandas as pd
import streamlit as st

from src.predict import load_artifacts, predict_applicant
from src.explain import explain_prediction

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "artifacts" / "creditwise_xgboost_pipeline.joblib"
METADATA_PATH = ROOT / "artifacts" / "creditwise_metadata.json"

st.set_page_config(
    page_title="CreditWise AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp {
        background:
          radial-gradient(circle at 12% 8%, rgba(88,101,242,.13), transparent 25%),
          radial-gradient(circle at 88% 15%, rgba(0,200,170,.10), transparent 24%),
          #090d16;
        color: #f4f7fb;
    }
    [data-testid="stSidebar"] {
        background: rgba(13,18,30,.94);
        border-right: 1px solid rgba(255,255,255,.08);
    }
    .hero {
        padding: 1.5rem 1.7rem;
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(30,38,65,.92), rgba(13,20,35,.78));
        box-shadow: 0 18px 55px rgba(0,0,0,.28);
        margin-bottom: 1.2rem;
    }
    .eyebrow {
        letter-spacing: .16em;
        text-transform: uppercase;
        font-size: .75rem;
        opacity: .65;
    }
    .hero h1 { margin: .2rem 0 .35rem 0; font-size: 2.55rem; }
    .hero p { opacity: .72; margin: 0; }
    .status-card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,.09);
        background: rgba(255,255,255,.035);
        min-height: 122px;
    }
    .status-label { font-size:.78rem; opacity:.62; text-transform:uppercase; letter-spacing:.08em; }
    .status-value { font-size:1.65rem; font-weight:700; margin-top:.35rem; }
    .status-sub { font-size:.82rem; opacity:.55; margin-top:.25rem; }
    .empty-state {
        padding: 2.4rem 1rem;
        text-align:center;
        border:1px dashed rgba(255,255,255,.14);
        border-radius:18px;
        color:rgba(255,255,255,.55);
        background:rgba(255,255,255,.02);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08);
        padding: 14px;
        border-radius: 16px;
    }
    .stButton > button, .stFormSubmitButton > button {
        border-radius: 12px;
        font-weight: 700;
        min-height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_artifacts():
    return load_artifacts(MODEL_PATH, METADATA_PATH)

model, metadata = get_artifacts()
preprocessor = model.named_steps["preprocessor"]
expected_numeric = list(preprocessor.transformers_[0][2])
expected_categorical = list(preprocessor.transformers_[1][2])
engineered = set(metadata.get("engineered_features", []))
raw_numeric = [c for c in expected_numeric if c not in engineered]
raw_categorical = [c for c in expected_categorical if c not in engineered]

# Keep form state empty until the user intentionally starts an analysis.
if "analysis" not in st.session_state:
    st.session_state.analysis = None

st.markdown("""
<div class="hero">
  <div class="eyebrow">Explainable Credit Intelligence</div>
  <h1>CreditWise AI</h1>
  <p>Real-time applicant scoring, explainable risk signals, and threshold-aware credit decisions.</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Decision Engine")
policy = st.sidebar.radio(
    "Decision policy",
    ["Standard", "Cost-aware"],
    captions=["0.50 probability threshold", "Notebook-derived illustrative threshold"],
)
threshold = (
    float(metadata["default_threshold"])
    if policy == "Standard"
    else float(metadata["business_threshold"])
)
st.sidebar.metric("Active threshold", f"{threshold:.2f}")
st.sidebar.caption("Cost-aware mode is a portfolio experiment, not a real lending policy.")

tab_score, tab_explain, tab_model = st.tabs(
    ["Live Scoring", "Risk Explanation", "Model Intelligence"]
)

with tab_score:
    left, right = st.columns([1.08, .92], gap="large")

    with left:
        st.subheader("Applicant profile")
        st.caption(
            "No sample applicant is pre-filled. Enter the values you want to analyze. "
            "DTI is entered as a percentage (for example, 18 means 18%)."
        )

        with st.form("applicant_form", clear_on_submit=False):
            values = {}
            cols = st.columns(2)

            for i, feature in enumerate(raw_numeric):
                with cols[i % 2]:
                    if feature == "DTI_Ratio":
                        # Use a percentage UI to avoid decimal-entry/state confusion.
                        # Example: 18 means 18%, and the model receives 0.18.
                        dti_percent = st.number_input(
                            "DTI Ratio (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=None,
                            step=1.0,
                            placeholder="e.g. 18",
                            help="Enter DTI as a percentage. Example: 18 = 18% = 0.18 for the model.",
                            key="input_DTI_Ratio_percent",
                        )
                        values[feature] = (
                            None if dti_percent is None else dti_percent / 100.0
                        )
                    else:
                        values[feature] = st.number_input(
                            feature.replace("_", " "),
                            value=None,
                            placeholder="Enter value",
                            format="%.2f",
                            key=f"input_{feature}",
                        )

            # Read the exact categories learned by the fitted OneHotEncoder.
            categorical_transformer = preprocessor.named_transformers_["categorical"]
            onehot = categorical_transformer.named_steps["onehot"]
            learned_categories = {
                feature: [str(v) for v in categories]
                for feature, categories in zip(raw_categorical, onehot.categories_)
            }

            for i, feature in enumerate(raw_categorical):
                with cols[i % 2]:
                    options = learned_categories.get(feature, [])
                    values[feature] = st.selectbox(
                        feature.replace("_", " "),
                        options=options,
                        index=None,
                        placeholder="Select category",
                        key=f"input_{feature}",
                    )

            submitted = st.form_submit_button(
                "Analyze credit profile",
                type="primary",
                use_container_width=True,
            )

        if submitted:
            missing = [
                k for k, v in values.items()
                if v is None or (isinstance(v, str) and not v.strip())
            ]
            if missing:
                st.warning("Complete all required fields before running the analysis.")
                st.caption("Missing: " + ", ".join(x.replace("_", " ") for x in missing))
            else:
                progress = st.progress(0, text="Validating applicant data…")
                stages = [
                    (18, "Validating applicant data…"),
                    (38, "Engineering credit features…"),
                    (61, "Running XGBoost risk scoring…"),
                    (82, "Applying decision threshold…"),
                    (100, "Analysis complete"),
                ]
                for pct, label in stages:
                    progress.progress(pct, text=label)
                    time.sleep(0.16)

                result = predict_applicant(
                    model=model,
                    metadata=metadata,
                    raw_input=values,
                    threshold=threshold,
                )
                st.session_state.analysis = result
                progress.empty()
                st.success("Applicant analysis completed.")

    with right:
        st.subheader("Real-time score")
        result = st.session_state.analysis

        if result is None:
            st.markdown("""
            <div class="empty-state">
                <div style="font-size:2rem">◌</div>
                <strong>No applicant analyzed yet</strong><br>
                Complete the applicant profile and run the analysis to generate a live score.
            </div>
            """, unsafe_allow_html=True)
        else:
            probability = result["probability"]
            score = round(probability * 100)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div class="status-card">
                    <div class="status-label">CreditWise Score</div>
                    <div class="status-value">{score}/100</div>
                    <div class="status-sub">Model approval confidence</div>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div class="status-card">
                    <div class="status-label">Decision</div>
                    <div class="status-value">{result["decision"]}</div>
                    <div class="status-sub">Threshold {result["threshold"]:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            st.progress(probability, text=f"Approval probability · {probability:.1%}")

            c3, c4 = st.columns(2)
            c3.metric("Risk level", result["risk_level"])
            c4.metric("Approval probability", f"{probability:.1%}")

            st.caption(
                "The score updates whenever a new applicant is analyzed. "
                "Decision = probability compared with the active policy threshold."
            )

with tab_explain:
    result = st.session_state.analysis

    if result is None:
        st.markdown("""
        <div class="empty-state">
            <strong>No explanation available</strong><br>
            Analyze an applicant first. SHAP factors will appear here only after scoring.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.subheader("What influenced this score?")
        with st.spinner("Analyzing feature contributions…"):
            explanation = explain_prediction(
                model,
                result["prepared_input"],
                top_n=8,
            )

        chart_df = explanation[["Feature", "SHAP Value"]].set_index("Feature")
        st.bar_chart(chart_df)

        st.caption(
            "Positive SHAP values push the prediction toward approval; "
            "negative values push it toward rejection."
        )

        st.subheader("Decision factors")
        table = explanation[["Feature", "SHAP Value", "Impact"]].copy()
        table["SHAP Value"] = table["SHAP Value"].round(4)
        st.dataframe(table, use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("Model intelligence")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model", "XGBoost")
    m2.metric("Training CV ROC-AUC", f'{metadata.get("training_cv_roc_auc", 0):.3f}')
    m3.metric("Saved test ROC-AUC", f'{metadata.get("test_roc_auc", 0):.3f}')

    st.write(
        "CreditWise uses leakage-safe preprocessing, engineered financial ratios, "
        "XGBoost classification, SHAP explanations, and configurable decision thresholds."
    )
    st.info(
        "Educational portfolio system only. It is not designed or validated for real-world "
        "credit underwriting or financial decisions."
    )
