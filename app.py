"""
app.py
EduGuard AI – Student Academic Risk Prediction System
Streamlit application with 6 pages:
  Home | Data Analysis | Model Performance | Student Risk Prediction
  Feature Insights | About / SDG 4
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Allow src/ imports regardless of cwd ────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_dataset, create_target, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS
from evaluate_model import (
    build_comparison_dataframe,
    confusion_matrix_to_df,
    summarise_metrics,
)
from prediction import load_model_bundle, predict_risk, format_probability

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(__file__)
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "academic_risk_model.pkl")
DATA_PATH    = os.path.join(PROJECT_ROOT, "data", "student-mat.csv")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduGuard AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Root colour tokens ─────────────────────────────────────────────────── */
:root {
    --bg-app:      #f0f4f8;
    --bg-card:     #ffffff;
    --bg-sidebar0: #1a2a4a;
    --bg-sidebar1: #0d1b35;
    --text-main:   #0f1923;
    --text-muted:  #3d4f63;
    --text-light:  #5a6f87;
    --accent:      #2563eb;
    --border:      #d1d9e3;
}

/* ─── App shell ──────────────────────────────────────────────────────────── */
.stApp,
.stApp > div,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
section[data-testid="stMain"] > div {
    background-color: var(--bg-app) !important;
}

/* ─── Force ALL main-area text to dark ───────────────────────────────────── */
.stApp p,
.stApp li,
.stApp span,
.stApp label,
.stApp div,
.stApp small,
.stApp caption,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: var(--text-main) !important;
}

/* ─── Headings ───────────────────────────────────────────────────────────── */
.stApp h1,
[data-testid="stMarkdownContainer"] h1 {
    color: #0f1923 !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.25rem !important;
}
.stApp h2,
[data-testid="stMarkdownContainer"] h2 {
    color: #162236 !important;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--accent);
    padding-bottom: 5px;
    margin-top: 1.2rem !important;
}
.stApp h3,
[data-testid="stMarkdownContainer"] h3 {
    color: #1e3252 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
.stApp h4,
[data-testid="stMarkdownContainer"] h4 {
    color: #253d5e !important;
    font-weight: 600 !important;
}

/* ─── Metric cards ───────────────────────────────────────────────────────── */
div[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 14px 20px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.07) !important;
}
div[data-testid="metric-container"] label,
div[data-testid="metric-container"] [data-testid="stMetricLabel"],
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: var(--text-muted) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"],
div[data-testid="metric-container"] [data-testid="stMetricValue"] div {
    color: #0f1923 !important;
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}

/* ─── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--bg-sidebar0) 0%, var(--bg-sidebar1) 100%) !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color: #dce6f5 !important;
}
[data-testid="stSidebar"] .stRadio label {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] h2 {
    color: #ffffff !important;
    border-bottom: 1px solid rgba(255,255,255,0.2) !important;
    font-size: 1.15rem !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* ─── Input widgets – keep labels readable ───────────────────────────────── */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: var(--text-main) !important;
    font-weight: 500 !important;
}
.stSelectbox label,
.stSlider label,
.stNumberInput label,
.stTextInput label,
.stRadio label {
    color: var(--text-main) !important;
}
/* Selectbox dropdown text */
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] div {
    color: #0f1923 !important;
}

/* ─── Tabs ───────────────────────────────────────────────────────────────── */
button[data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    font-weight: 700 !important;
    border-bottom-color: var(--accent) !important;
}

/* ─── DataFrames / tables ────────────────────────────────────────────────── */
[data-testid="stDataFrame"] th,
[data-testid="stDataFrame"] td {
    color: #0f1923 !important;
}
thead tr th {
    background-color: #1a2a4a !important;
    color: #ffffff !important;
}

/* ─── Code blocks ────────────────────────────────────────────────────────── */
.stCode pre, .stCode code, pre, code {
    color: #0f1923 !important;
    background: #eef2f7 !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}

/* ─── Alert boxes ────────────────────────────────────────────────────────── */
.stAlert { border-radius: 8px !important; }
[data-testid="stAlert"] p { color: #0f1923 !important; }

/* ─── Expander ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] summary span {
    color: var(--text-main) !important;
    font-weight: 500 !important;
}

/* ─── Form submit button ─────────────────────────────────────────────────── */
.stFormSubmitButton button {
    background-color: var(--accent) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 0.55rem 1.4rem !important;
    font-size: 1rem !important;
}
.stFormSubmitButton button:hover {
    background-color: #1d4ed8 !important;
}

/* ─── Prediction result cards ────────────────────────────────────────────── */
.result-card {
    border-radius: 12px;
    padding: 24px 28px;
    margin-top: 12px;
    font-size: 1rem;
}
.result-card h2 { border-bottom: none !important; margin-top: 0 !important; }
.high-risk  {
    background: #fff1f0 !important;
    border: 2px solid #ef4444 !important;
    color: #7f1d1d !important;
}
.high-risk  h2 { color: #991b1b !important; }
.high-risk  p  { color: #7f1d1d !important; }
.lower-risk {
    background: #f0fdf4 !important;
    border: 2px solid #22c55e !important;
    color: #14532d !important;
}
.lower-risk h2 { color: #166534 !important; }
.lower-risk p  { color: #14532d !important; }

/* ─── SDG info boxes ─────────────────────────────────────────────────────── */
.sdg-card strong { color: #1e3a8a !important; }
.sdg-card small  { color: #374151 !important; }

/* ─── Captions ───────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p,
.stCaption {
    color: var(--text-light) !important;
    font-size: 0.8rem !important;
}

/* ─── Horizontal rules ───────────────────────────────────────────────────── */
hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)


# ── Plotly chart defaults ────────────────────────────────────────────────────
_CHART_FONT   = dict(family="'Segoe UI', Arial, sans-serif", color="#0f1923", size=13)
_CHART_LAYOUT = dict(
    font=_CHART_FONT,
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    title_font=dict(color="#0f1923", size=15, family="'Segoe UI', Arial, sans-serif"),
    xaxis=dict(
        tickfont=dict(color="#0f1923"),
        title_font=dict(color="#0f1923"),
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
    ),
    yaxis=dict(
        tickfont=dict(color="#0f1923"),
        title_font=dict(color="#0f1923"),
        gridcolor="#e2e8f0",
        linecolor="#cbd5e1",
    ),
    legend=dict(
        font=dict(color="#0f1923"),
        bgcolor="#ffffff",
        bordercolor="#d1d9e3",
        borderwidth=1,
    ),
    coloraxis_colorbar=dict(
        tickfont=dict(color="#0f1923"),
        title_font=dict(color="#0f1923"),
    ),
    margin=dict(t=50, b=40, l=40, r=20),
)

def _fig(fig):
    """Apply consistent dark-text styling to any Plotly figure."""
    fig.update_layout(**_CHART_LAYOUT)
    # Force axis text on any subplot axes that exist
    for ax in ("xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3"):
        fig.update_layout(**{ax: dict(
            tickfont=dict(color="#0f1923"),
            title_font=dict(color="#0f1923"),
        )})
    return fig


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    """Load and return df_full (with G3 + academic_risk column)."""
    df_raw = load_dataset(DATA_PATH)
    df = df_raw.drop_duplicates().copy()
    df = create_target(df)
    return df


@st.cache_resource(show_spinner=False)
def load_bundle():
    return load_model_bundle(MODEL_PATH)


# ── Sidebar navigation ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎓 EduGuard AI")
    st.markdown("*Early Academic Risk Detection*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "📊  Data Analysis",
            "🤖  Model Performance",
            "🔮  Student Risk Prediction",
            "💡  Feature Insights",
            "🌍  About / SDG 4",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("Dataset: UCI Student Performance\nBuilt with Streamlit & scikit-learn")

# ── Helper: model availability banner ────────────────────────────────────────

def model_not_trained():
    st.error(
        "⚠️ **Model not found.**\n\n"
        "Please run the training script first:\n\n"
        "```\npython src/train_model.py\n```"
    )
    st.stop()


def data_not_found():
    st.error(
        "⚠️ **Dataset not found.**\n\n"
        "Download `student-mat.csv` from:\n"
        "https://archive.ics.uci.edu/dataset/320/student+performance\n\n"
        "Place it at: `data/student-mat.csv`"
    )
    st.stop()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 – HOME
# ════════════════════════════════════════════════════════════════════════════════

if page == "🏠  Home":
    st.markdown("<h1>🎓 EduGuard AI</h1>", unsafe_allow_html=True)
    st.markdown("### Early Academic Risk Detection System")
    st.markdown("""
    EduGuard AI helps educators identify students at academic risk **before final
    grades are issued**, enabling timely interventions that support student success.
    """)

    col1, col2, col3, col4 = st.columns(4)

    # Dataset stats
    if os.path.exists(DATA_PATH):
        try:
            df = load_data()
            high_risk  = int(df["academic_risk"].sum())
            lower_risk = len(df) - high_risk
            col1.metric("📋 Total Students",  f"{len(df):,}")
            col2.metric("🔴 High Risk",        f"{high_risk:,}")
            col3.metric("🟢 Lower Risk",        f"{lower_risk:,}")
            col4.metric("📐 Features Used",    "30")
        except Exception:
            pass

    st.markdown("---")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("""
        #### 📊 Data Analysis
        Explore distributions, correlations, and patterns in the student
        performance dataset.
        """)
    with col_b:
        st.markdown("""
        #### 🤖 Model Performance
        Compare Logistic Regression, Random Forest, and XGBoost on real
        accuracy, precision, recall, F1, and ROC-AUC metrics.
        """)
    with col_c:
        st.markdown("""
        #### 🔮 Risk Prediction
        Enter individual student data and receive an instant academic risk
        assessment with confidence score.
        """)

    st.markdown("---")
    st.info(
        "**How to get started:**\n"
        "1. Download `student-mat.csv` from UCI and place it in `data/`\n"
        "2. Run `python src/train_model.py`\n"
        "3. Explore the pages in the sidebar"
    )


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 – DATA ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

elif page == "📊  Data Analysis":
    st.markdown("<h1>📊 Data Analysis</h1>", unsafe_allow_html=True)

    if not os.path.exists(DATA_PATH):
        data_not_found()

    df = load_data()

    # ── Overview ──────────────────────────────────────────────────────────────
    st.markdown("## Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(df):,}")
    col2.metric("Columns", f"{df.shape[1]}")
    col3.metric("High Risk Students", f"{int(df['academic_risk'].sum())}")
    col4.metric("Missing Values", "0")

    with st.expander("📋 Raw Data Sample (first 10 rows)", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # ── Risk distribution ─────────────────────────────────────────────────────
    st.markdown("## Risk Distribution")
    col1, col2 = st.columns([1, 2])

    risk_counts = df["academic_risk"].value_counts().reset_index()
    risk_counts.columns = ["Risk Class", "Count"]
    risk_counts["Label"] = risk_counts["Risk Class"].map({0: "Lower Risk", 1: "High Risk"})

    with col1:
        fig_pie = px.pie(
            risk_counts,
            values="Count",
            names="Label",
            color="Label",
            color_discrete_map={"High Risk": "#ef4444", "Lower Risk": "#22c55e"},
            hole=0.45,
            title="Academic Risk Distribution",
        )
        fig_pie.update_traces(textinfo="percent+value")
        fig_pie.update_layout(showlegend=True, height=360, margin=dict(t=40, b=20))
        st.plotly_chart(_fig(fig_pie), use_container_width=True)

    with col2:
        fig_g3 = px.histogram(
            df, x="G3",
            color=df["academic_risk"].map({0: "Lower Risk", 1: "High Risk"}),
            nbins=21,
            title="Final Grade (G3) Distribution by Risk",
            labels={"color": "Risk"},
            color_discrete_map={"High Risk": "#ef4444", "Lower Risk": "#22c55e"},
            barmode="overlay",
            opacity=0.75,
        )
        fig_g3.add_vline(x=10, line_dash="dash", line_color="#1a2a4a",
                         annotation_text="Threshold (G3=10)")
        fig_g3.update_layout(height=360, xaxis_title="Final Grade (G3)", margin=dict(t=40))
        st.plotly_chart(_fig(fig_g3), use_container_width=True)

    st.markdown("---")

    # ── Study time vs risk ────────────────────────────────────────────────────
    st.markdown("## Study Time vs Academic Risk")
    df_plot = df.copy()
    df_plot["Risk"] = df_plot["academic_risk"].map({0: "Lower Risk", 1: "High Risk"})

    study_map = {1: "< 2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: "> 10 hrs"}
    df_plot["Study Time Label"] = df_plot["studytime"].map(study_map).fillna(df_plot["studytime"].astype(str))

    study_risk = (
        df_plot.groupby(["Study Time Label", "Risk"])
        .size()
        .reset_index(name="Count")
    )
    fig_study = px.bar(
        study_risk,
        x="Study Time Label",
        y="Count",
        color="Risk",
        barmode="group",
        title="Study Time vs Academic Risk",
        color_discrete_map={"High Risk": "#ef4444", "Lower Risk": "#22c55e"},
        category_orders={"Study Time Label": ["< 2 hrs", "2–5 hrs", "5–10 hrs", "> 10 hrs"]},
    )
    fig_study.update_layout(height=380, margin=dict(t=40))
    st.plotly_chart(_fig(fig_study), use_container_width=True)

    st.markdown("---")

    # ── Absences vs risk ──────────────────────────────────────────────────────
    st.markdown("## Absences vs Academic Risk")
    fig_abs = px.box(
        df_plot,
        x="Risk",
        y="absences",
        color="Risk",
        color_discrete_map={"High Risk": "#ef4444", "Lower Risk": "#22c55e"},
        points="outliers",
        title="Distribution of Absences by Risk Group",
    )
    fig_abs.update_layout(height=380, margin=dict(t=40), showlegend=False)
    st.plotly_chart(_fig(fig_abs), use_container_width=True)

    st.markdown("---")

    # ── Previous failures vs risk ─────────────────────────────────────────────
    st.markdown("## Previous Failures vs Academic Risk")
    fail_risk = (
        df_plot.groupby(["failures", "Risk"])
        .size()
        .reset_index(name="Count")
    )
    fig_fail = px.bar(
        fail_risk,
        x="failures",
        y="Count",
        color="Risk",
        barmode="group",
        title="Number of Past Failures vs Academic Risk",
        labels={"failures": "Previous Class Failures"},
        color_discrete_map={"High Risk": "#ef4444", "Lower Risk": "#22c55e"},
    )
    fig_fail.update_layout(height=380, margin=dict(t=40))
    st.plotly_chart(_fig(fig_fail), use_container_width=True)

    st.markdown("---")

    # ── Parental education vs risk ────────────────────────────────────────────
    st.markdown("## Parental Education vs Academic Risk")
    edu_map = {0: "None", 1: "Primary", 2: "5th–9th Grade", 3: "Secondary", 4: "Higher Ed."}
    df_plot["Mother Edu"] = df_plot["Medu"].map(edu_map).fillna(df_plot["Medu"].astype(str))

    medu_risk = (
        df_plot.groupby(["Mother Edu", "Risk"])
        .size()
        .reset_index(name="Count")
    )
    fig_edu = px.bar(
        medu_risk,
        x="Mother Edu",
        y="Count",
        color="Risk",
        barmode="group",
        title="Mother's Education Level vs Academic Risk",
        color_discrete_map={"High Risk": "#ef4444", "Lower Risk": "#22c55e"},
        category_orders={"Mother Edu": ["None", "Primary", "5th–9th Grade", "Secondary", "Higher Ed."]},
    )
    fig_edu.update_layout(height=380, margin=dict(t=40))
    st.plotly_chart(_fig(fig_edu), use_container_width=True)

    st.markdown("---")

    # ── School / family support vs risk ───────────────────────────────────────
    st.markdown("## Support Systems vs Academic Risk")
    col1, col2 = st.columns(2)

    def support_bar(df_in, col_name, title, color_map):
        grp = (
            df_in.groupby([col_name, "Risk"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            grp, x=col_name, y="Count", color="Risk",
            barmode="group", title=title,
            color_discrete_map=color_map,
        )
        fig.update_layout(height=340, margin=dict(t=40))
        return _fig(fig)

    cmap = {"High Risk": "#ef4444", "Lower Risk": "#22c55e"}
    with col1:
        st.plotly_chart(support_bar(df_plot, "schoolsup", "School Extra Support vs Risk", cmap),
                        use_container_width=True)
    with col2:
        st.plotly_chart(support_bar(df_plot, "famsup", "Family Support vs Risk", cmap),
                        use_container_width=True)

    st.markdown("---")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown("## Correlation with Academic Risk")
    num_cols = [c for c in NUMERICAL_COLUMNS if c in df.columns] + ["academic_risk"]
    corr_vals = df[num_cols].corr()["academic_risk"].drop("academic_risk").sort_values()

    fig_corr = px.bar(
        x=corr_vals.values,
        y=corr_vals.index,
        orientation="h",
        title="Pearson Correlation of Numerical Features with Academic Risk",
        labels={"x": "Correlation", "y": "Feature"},
        color=corr_vals.values,
        color_continuous_scale=["#22c55e", "#f8f8f8", "#ef4444"],
    )
    fig_corr.update_layout(height=480, margin=dict(t=40))
    st.plotly_chart(_fig(fig_corr), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 – MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════

elif page == "🤖  Model Performance":
    st.markdown("<h1>🤖 Model Performance</h1>", unsafe_allow_html=True)

    if not os.path.exists(MODEL_PATH):
        model_not_trained()

    bundle = load_bundle()
    results = bundle["all_results"]
    best_name = bundle["best_model_name"]

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("## Model Comparison")
    st.success(f"✅ Best model selected: **{best_name}** (highest F1-Score)")

    comparison_df = build_comparison_dataframe(bundle["comparison_table"])
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            color="#bbf7d0",
            axis=0,
        ),
        use_container_width=True,
        height=140,
    )

    st.markdown("---")

    # ── Grouped metrics bar chart ─────────────────────────────────────────────
    st.markdown("## Metrics Comparison Chart")
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    fig_metrics = go.Figure()

    colors = {"Logistic Regression": "#3b82f6", "Random Forest": "#22c55e", "XGBoost": "#f59e0b"}
    for _, row in comparison_df.iterrows():
        fig_metrics.add_trace(go.Bar(
            name=row["Model"],
            x=metric_cols,
            y=[row[c] for c in metric_cols],
            marker_color=colors.get(row["Model"], "#94a3b8"),
        ))

    fig_metrics.update_layout(
        barmode="group",
        title="Model Metrics Comparison",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        height=420,
        margin=dict(t=50),
    )
    st.plotly_chart(_fig(fig_metrics), use_container_width=True)

    st.markdown("---")

    # ── Per-model details ─────────────────────────────────────────────────────
    st.markdown("## Detailed Metrics per Model")

    tabs = st.tabs(list(results.keys()))
    for tab, (model_name, metrics) in zip(tabs, results.items()):
        with tab:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
            c2.metric("Precision", f"{metrics['precision']:.4f}")
            c3.metric("Recall",    f"{metrics['recall']:.4f}")
            c4.metric("F1-Score",  f"{metrics['f1_score']:.4f}")
            c5.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}" if metrics.get('roc_auc') else "N/A")

            st.markdown("#### Confusion Matrix")
            cm_df = confusion_matrix_to_df(metrics["confusion_matrix"])
            st.dataframe(cm_df, use_container_width=False)

            # Confusion matrix heatmap
            cm_arr = np.array(metrics["confusion_matrix"])
            fig_cm = px.imshow(
                cm_arr,
                text_auto=True,
                x=["Predicted: Lower Risk", "Predicted: High Risk"],
                y=["Actual: Lower Risk", "Actual: High Risk"],
                color_continuous_scale=[[0, "#dbeafe"], [1, "#1e3a8a"]],
                title=f"Confusion Matrix – {model_name}",
            )
            fig_cm.update_layout(height=340, margin=dict(t=50))
            st.plotly_chart(_fig(fig_cm), use_container_width=True)

            st.markdown("#### Classification Report")
            st.code(metrics["classification_report"])

    st.markdown("---")

    # ── ROC-AUC comparison ────────────────────────────────────────────────────
    roc_vals = {
        name: m["roc_auc"]
        for name, m in results.items()
        if m.get("roc_auc") is not None
    }
    if roc_vals:
        st.markdown("## ROC-AUC Scores")
        fig_roc = px.bar(
            x=list(roc_vals.keys()),
            y=list(roc_vals.values()),
            labels={"x": "Model", "y": "ROC-AUC Score"},
            title="ROC-AUC Score by Model",
            color=list(roc_vals.keys()),
            color_discrete_sequence=["#3b82f6", "#22c55e", "#f59e0b"],
            text_auto=".4f",
        )
        fig_roc.update_layout(
            yaxis=dict(range=[0.5, 1.0]),
            height=380,
            showlegend=False,
            margin=dict(t=50),
        )
        fig_roc.add_hline(y=0.5, line_dash="dot", line_color="gray",
                          annotation_text="Random Baseline")
        st.plotly_chart(_fig(fig_roc), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 4 – STUDENT RISK PREDICTION
# ════════════════════════════════════════════════════════════════════════════════

elif page == "🔮  Student Risk Prediction":
    st.markdown("<h1>🔮 Student Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("Enter a student's information below to receive an academic risk assessment.")

    if not os.path.exists(MODEL_PATH):
        model_not_trained()

    bundle = load_bundle()

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown("### 👤 Personal Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            school   = st.selectbox("School", ["GP", "MS"], help="Gabriel Pereira or Mousinho da Silveira")
            sex      = st.selectbox("Sex", ["F", "M"])
            age      = st.slider("Age", 15, 22, 17)
        with col2:
            address  = st.selectbox("Address", ["U", "R"], help="Urban or Rural")
            famsize  = st.selectbox("Family Size", ["LE3", "GT3"], help="≤3 or >3 members")
            Pstatus  = st.selectbox("Parents' Status", ["T", "A"], help="Together or Apart")
        with col3:
            guardian = st.selectbox("Guardian", ["mother", "father", "other"])
            internet = st.selectbox("Internet at Home", ["yes", "no"])
            romantic = st.selectbox("In a Romantic Relationship", ["no", "yes"])

        st.markdown("### 👨‍👩‍👧 Family Background")
        col1, col2, col3 = st.columns(3)
        with col1:
            Medu = st.select_slider("Mother's Education",
                                    options=[0, 1, 2, 3, 4],
                                    format_func=lambda x: {0:"None",1:"Primary",2:"5th–9th",3:"Secondary",4:"Higher"}[x],
                                    value=2)
            Fedu = st.select_slider("Father's Education",
                                    options=[0, 1, 2, 3, 4],
                                    format_func=lambda x: {0:"None",1:"Primary",2:"5th–9th",3:"Secondary",4:"Higher"}[x],
                                    value=2)
        with col2:
            Mjob = st.selectbox("Mother's Job",
                                ["teacher", "health", "services", "at_home", "other"])
            Fjob = st.selectbox("Father's Job",
                                ["teacher", "health", "services", "at_home", "other"])
        with col3:
            famrel   = st.slider("Family Relationship Quality", 1, 5, 3,
                                 help="1=Very bad … 5=Excellent")
            famsup   = st.selectbox("Family Educational Support", ["yes", "no"])
            paid     = st.selectbox("Extra Paid Classes", ["no", "yes"])

        st.markdown("### 🏫 Academic Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            reason    = st.selectbox("Reason for Choosing This School",
                                     ["course", "home", "reputation", "other"])
            studytime = st.select_slider("Weekly Study Time",
                                         options=[1, 2, 3, 4],
                                         format_func=lambda x: {1:"<2 hrs",2:"2–5 hrs",3:"5–10 hrs",4:">10 hrs"}[x],
                                         value=2)
            failures  = st.select_slider("Past Class Failures", options=[0, 1, 2, 3], value=0)
        with col2:
            schoolsup  = st.selectbox("Extra Educational School Support", ["no", "yes"])
            activities = st.selectbox("Extra-Curricular Activities", ["no", "yes"])
            nursery    = st.selectbox("Attended Nursery School", ["yes", "no"])
        with col3:
            higher     = st.selectbox("Wants Higher Education", ["yes", "no"])
            traveltime = st.select_slider("Travel Time to School",
                                          options=[1, 2, 3, 4],
                                          format_func=lambda x: {1:"<15 min",2:"15–30 min",3:"30–60 min",4:">60 min"}[x],
                                          value=1)
            absences   = st.number_input("Number of Absences", min_value=0, max_value=93, value=4, step=1)

        st.markdown("### 🎉 Lifestyle")
        col1, col2, col3 = st.columns(3)
        with col1:
            freetime = st.slider("Free Time After School", 1, 5, 3, help="1=Very low … 5=Very high")
            goout    = st.slider("Going Out with Friends",  1, 5, 2, help="1=Very low … 5=Very high")
        with col2:
            Dalc = st.slider("Workday Alcohol Consumption", 1, 5, 1, help="1=Very low … 5=Very high")
            Walc = st.slider("Weekend Alcohol Consumption", 1, 5, 1, help="1=Very low … 5=Very high")
        with col3:
            health = st.slider("Current Health Status", 1, 5, 4, help="1=Very bad … 5=Very good")

        submitted = st.form_submit_button("🔮 Predict Academic Risk", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if submitted:
        raw_input = {
            "school": school, "sex": sex, "age": age, "address": address,
            "famsize": famsize, "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu,
            "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
            "traveltime": traveltime, "studytime": studytime, "failures": failures,
            "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher,
            "internet": internet, "romantic": romantic, "famrel": famrel,
            "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
            "health": health, "absences": absences,
        }

        try:
            result = predict_risk(raw_input, bundle)

            risk_class   = result["risk_class"]
            label        = result["label"]
            probability  = result["probability"]
            top_factors  = result["top_factors"]

            card_class = "high-risk" if risk_class == 1 else "lower-risk"
            emoji      = "🔴" if risk_class == 1 else "🟢"

            st.markdown("---")
            st.markdown("### Prediction Result")

            confidence = format_probability(probability)

            heading_color = "#991b1b" if risk_class == 1 else "#166534"
            body_color    = "#7f1d1d" if risk_class == 1 else "#14532d"
            st.markdown(f"""
            <div class="result-card {card_class}">
                <h2 style="margin:0;color:{heading_color};">{emoji} {label}</h2>
                <p style="font-size:1.1rem;margin-top:8px;color:{body_color};">
                    <strong>Confidence:</strong> {confidence} probability of High Academic Risk
                </p>
            </div>
            """, unsafe_allow_html=True)

            if probability is not None:
                st.markdown("#### Risk Probability Gauge")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(probability * 100, 1),
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "High Risk Probability (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "#ef4444" if risk_class == 1 else "#22c55e"},
                        "steps": [
                            {"range": [0, 40],  "color": "#dcfce7"},
                            {"range": [40, 60], "color": "#fef9c3"},
                            {"range": [60, 100],"color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line":  {"color": "#1a2a4a", "width": 3},
                            "thickness": 0.8,
                            "value": 50,
                        },
                    },
                ))
                fig_gauge.update_layout(height=280, margin=dict(t=30, b=10))
                st.plotly_chart(_fig(fig_gauge), use_container_width=True)

            # ── Key contributing factors ───────────────────────────────────────
            if top_factors:
                st.markdown("#### 💡 Top Contributing Factors")
                factor_df = pd.DataFrame(top_factors)
                fig_factors = px.bar(
                    factor_df,
                    x="importance",
                    y="feature",
                    orientation="h",
                    title="Feature Importance (Model-wide)",
                    labels={"importance": "Importance Score", "feature": "Feature"},
                    color="importance",
                    color_continuous_scale=[[0, "#dbeafe"], [1, "#1e3a8a"]],
                )
                fig_factors.update_layout(
                    height=280,
                    margin=dict(t=40),
                    showlegend=False,
                    yaxis=dict(autorange="reversed"),
                )
                st.plotly_chart(_fig(fig_factors), use_container_width=True)

                st.markdown("**What these factors mean:**")
                explanations = {
                    "failures":   "Past failures strongly predict future academic difficulty.",
                    "absences":   "Frequent absences are associated with higher risk.",
                    "studytime":  "More study time correlates with lower academic risk.",
                    "goout":      "High socialising frequency may reduce study focus.",
                    "Walc":       "Weekend alcohol consumption can impair academic performance.",
                    "Dalc":       "Weekday alcohol use is a significant risk factor.",
                    "health":     "Poor health status correlates with academic challenges.",
                    "Medu":       "Mother's higher education level lowers student risk.",
                    "Fedu":       "Father's higher education level lowers student risk.",
                    "higher":     "Students aspiring to higher education perform better.",
                }
                for factor in top_factors[:5]:
                    feat = factor["feature"]
                    exp  = explanations.get(feat, f"'{feat}' influences academic outcomes.")
                    st.markdown(f"- **{feat}**: {exp}")

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 5 – FEATURE INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════

elif page == "💡  Feature Insights":
    st.markdown("<h1>💡 Feature Insights</h1>", unsafe_allow_html=True)
    st.markdown("Explore which features matter most to the model and how they relate to risk.")

    if not os.path.exists(MODEL_PATH):
        model_not_trained()

    bundle = load_bundle()
    feat_imp   = bundle.get("feature_importance", [])
    best_name  = bundle["best_model_name"]

    # ── Feature importance chart ──────────────────────────────────────────────
    st.markdown(f"## Feature Importance — {best_name}")

    if feat_imp:
        fi_df = pd.DataFrame(feat_imp)
        fig_fi = px.bar(
            fi_df,
            x="importance",
            y="feature",
            orientation="h",
            title=f"Feature Importance – {best_name}",
            labels={"importance": "Importance Score", "feature": "Feature"},
            color="importance",
            color_continuous_scale=[[0, "#dbeafe"], [0.5, "#3b82f6"], [1, "#1e3a8a"]],
        )
        fig_fi.update_layout(
            height=600,
            margin=dict(t=50),
            yaxis=dict(autorange="reversed"),
            showlegend=False,
        )
        st.plotly_chart(_fig(fig_fi), use_container_width=True)
    else:
        st.info("Feature importance data not available for this model type.")

    st.markdown("---")

    # ── Feature deep-dive (only if data available) ────────────────────────────
    if not os.path.exists(DATA_PATH):
        st.info("Load `data/student-mat.csv` to see deeper feature analysis.")
    else:
        df = load_data()
        df["Risk"] = df["academic_risk"].map({0: "Lower Risk", 1: "High Risk"})

        st.markdown("## Feature Distribution by Risk Group")

        top_feats = [f["feature"] for f in feat_imp[:6]] if feat_imp else NUMERICAL_COLUMNS[:6]

        num_feats = [f for f in top_feats if f in df.columns and f in NUMERICAL_COLUMNS]

        if num_feats:
            cols_per_row = 2
            for i in range(0, len(num_feats), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, feat in enumerate(num_feats[i:i + cols_per_row]):
                    with row_cols[j]:
                        fig = px.violin(
                            df,
                            x="Risk",
                            y=feat,
                            color="Risk",
                            box=True,
                            points="outliers",
                            title=f"{feat} by Risk Group",
                            color_discrete_map={"High Risk": "#ef4444", "Lower Risk": "#22c55e"},
                        )
                        fig.update_layout(height=320, showlegend=False, margin=dict(t=40))
                        st.plotly_chart(_fig(fig), use_container_width=True)

        st.markdown("---")
        st.markdown("## Alcohol vs Academic Performance")
        fig_alc = make_subplots(rows=1, cols=2,
                                subplot_titles=["Workday Alcohol (Dalc)", "Weekend Alcohol (Walc)"])
        for idx, alc_col in enumerate(["Dalc", "Walc"]):
            for risk_val, risk_label, clr in [(0, "Lower Risk", "#22c55e"), (1, "High Risk", "#ef4444")]:
                sub = df[df["academic_risk"] == risk_val]
                cnt = sub.groupby(alc_col).size().reset_index(name="count")
                fig_alc.add_trace(
                    go.Bar(x=cnt[alc_col], y=cnt["count"], name=risk_label,
                           marker_color=clr, showlegend=(idx == 0)),
                    row=1, col=idx + 1,
                )
        fig_alc.update_layout(barmode="group", height=380, margin=dict(t=60))
        st.plotly_chart(_fig(fig_alc), use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 6 – ABOUT / SDG 4
# ════════════════════════════════════════════════════════════════════════════════

elif page == "🌍  About / SDG 4":
    st.markdown("<h1>🌍 About EduGuard AI & SDG 4</h1>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ## What is EduGuard AI?

        **EduGuard AI** is an early academic risk detection system built as part of a
        machine-learning college project aligned with **United Nations Sustainable
        Development Goal 4: Quality Education**.

        It uses student demographic, behavioural, and academic data to predict which
        students are most at risk of academic failure — enabling educators to intervene
        early and provide targeted support.

        ---

        ## 🎯 UN SDG 4 – Quality Education

        > *"Ensure inclusive and equitable quality education and promote lifelong learning
        > opportunities for all."*

        EduGuard AI directly supports **SDG 4** by:

        - **4.1** — Helping ensure all students complete quality secondary education
        - **4.5** — Identifying and supporting vulnerable students to eliminate disparities
        - **4.a** — Demonstrating how data-driven tools can improve educational outcomes

        ---

        ## 🔬 Dataset

        | Property       | Detail                                        |
        |----------------|-----------------------------------------------|
        | **Name**       | Student Performance Dataset                   |
        | **Source**     | UCI Machine Learning Repository               |
        | **URL**        | https://archive.ics.uci.edu/dataset/320       |
        | **Subject**    | Mathematics (student-mat.csv)                 |
        | **Instances**  | 395 students                                  |
        | **Features**   | 30 attributes (demographics, social, school)  |
        | **Target**     | G3 < 10 → High Risk; G3 ≥ 10 → Lower Risk   |

        ---

        ## 🤖 Models Used

        | Model                | Algorithm Family       |
        |----------------------|------------------------|
        | Logistic Regression  | Linear                 |
        | Random Forest        | Ensemble / Bagging     |
        | XGBoost              | Ensemble / Boosting    |

        ---

        ## ⚠️ Ethical Considerations

        - **No target leakage**: G1, G2, G3 are excluded from model input features.
        - **Balanced classes**: Class weights are used to handle imbalance fairly.
        - **Interpretability**: Feature importance is exposed to support transparent decisions.
        - **Human oversight**: Predictions should complement — not replace — educator judgement.

        ---

        ## 👨‍💻 Technical Stack

        Python · scikit-learn · XGBoost · Streamlit · Plotly · pandas · NumPy
        """)

    with col2:
        st.markdown("### SDG 4 Targets Addressed")
        for t, desc in [
            ("4.1", "Free, equitable, quality education"),
            ("4.5", "Eliminate disparities in education"),
            ("4.a", "Build effective learning environments"),
        ]:
            st.markdown(f"""
            <div class="sdg-card" style="background:#dbeafe;border-left:4px solid #1d4ed8;
                        padding:10px 14px;margin-bottom:10px;border-radius:6px">
                <strong style="color:#1e3a8a;font-size:0.95rem;">Target {t}</strong><br/>
                <small style="color:#374151;">{desc}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Quick Stats")
        if os.path.exists(DATA_PATH):
            try:
                df = load_data()
                high = int(df["academic_risk"].sum())
                total = len(df)
                st.metric("Students Analysed",   f"{total}")
                st.metric("High Risk Students",  f"{high} ({high/total*100:.0f}%)")
                st.metric("Features in Model",   "30")
                st.metric("Models Trained",      "3")
            except Exception:
                pass

        if os.path.exists(MODEL_PATH):
            try:
                bundle = load_bundle()
                best   = bundle["best_model_name"]
                f1     = bundle["all_results"][best]["f1_score"]
                auc    = bundle["all_results"][best].get("roc_auc")
                st.metric("Best Model",  best)
                st.metric("F1-Score",    f"{f1:.4f}")
                if auc:
                    st.metric("ROC-AUC", f"{auc:.4f}")
            except Exception:
                pass
