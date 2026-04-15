from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sqli_pipeline.config import ARTIFACT_DIR, REPORT_DIR  # noqa: E402
from sqli_pipeline.features import SQLiFeatureEngineer  # noqa: E402

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SQLi Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        #MainMenu, footer {visibility: hidden;}
        .block-container {padding-top: 1.8rem; padding-bottom: 2rem;}

        [data-testid="metric-container"] {
            background: #1a1d2e;
            border: 1px solid #2d3250;
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 2px solid #2d3250;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 0.45rem 1.1rem;
            border-radius: 6px 6px 0 0;
            font-weight: 500;
        }
        .badge-attack {
            display:inline-block; background:#ff4b4b22; color:#ff4b4b;
            border:1px solid #ff4b4b66; border-radius:8px;
            padding:0.4rem 1rem; font-weight:700; font-size:1.05rem;
        }
        .badge-benign {
            display:inline-block; background:#21c95422; color:#21c954;
            border:1px solid #21c95466; border-radius:8px;
            padding:0.4rem 1rem; font-weight:700; font-size:1.05rem;
        }
        .pipeline-step {
            background:#1a1d2e; border:1px solid #2d3250;
            border-radius:10px; padding:1.1rem 1.3rem; margin-bottom:0.8rem;
        }
        .step-label {
            font-size:0.72rem; font-weight:700; letter-spacing:0.08em;
            color:#7c8ab8; text-transform:uppercase; margin-bottom:0.2rem;
        }
        .step-title {
            font-size:1.05rem; font-weight:600; color:#e0e4f7;
        }
        .tag {
            display:inline-block; background:#2d3250; color:#adb5d8;
            border-radius:5px; padding:0.15rem 0.55rem;
            font-size:0.78rem; margin:0.15rem 0.1rem;
        }
        .best-badge {
            display:inline-block; background:#f7a34f22; color:#f7a34f;
            border:1px solid #f7a34f55; border-radius:5px;
            padding:0.1rem 0.6rem; font-size:0.75rem; font-weight:700;
        }
        hr.divider {border:none; border-top:1px solid #2d3250; margin:1.2rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Guard ──────────────────────────────────────────────────────────────────────
metadata_path = ARTIFACT_DIR / "model_metadata.json"
model_path = ARTIFACT_DIR / "best_model.joblib"

if not metadata_path.exists() or not model_path.exists():
    st.error("### Model not found — run training first")
    st.code("python scripts/train.py\npython scripts/evaluate.py\nstreamlit run streamlit_app.py", language="bash")
    st.stop()

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_data
def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

@st.cache_data
def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

metadata     = read_json(metadata_path)
metrics_df   = read_csv(REPORT_DIR / "model_metrics.csv")
eda_summary  = read_json(REPORT_DIR / "eda_summary.json")
eda_rows     = read_csv(REPORT_DIR / "eda_rows.csv")
feat_imp     = read_csv(REPORT_DIR / "feature_importance.csv")
eval_report  = read_json(REPORT_DIR / "best_model_evaluation.json")

MODEL_DISPLAY = {
    "random_forest":      "Random Forest",
    "logistic_regression": "Logistic Regression",
    "naive_bayes":        "Naive Bayes",
}
MODEL_FILES = {
    k: ARTIFACT_DIR / "models" / f"{k}.joblib"
    for k in MODEL_DISPLAY
}
best_model_name: str  = metadata.get("best_model", "random_forest")
results: dict         = metadata.get("results", {})

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("# 🛡️ SQL Injection Detector")
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Global top metrics ─────────────────────────────────────────────────────────
bm = results.get(best_model_name, {})
bm_metrics = bm.get("metrics", {})
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Best Model", MODEL_DISPLAY.get(best_model_name, best_model_name))
c2.metric("Threshold", f'{bm.get("threshold", 0):.3f}')
c3.metric("F1 Score",  f'{bm_metrics.get("f1_score", 0):.4f}')
c4.metric("Recall",    f'{bm_metrics.get("recall", 0):.4f}')
c5.metric("Precision", f'{bm_metrics.get("precision", 0):.4f}')
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_detect, tab_pipeline, tab_eda, tab_models, tab_eval, tab_features = st.tabs([
    "🔍 Detect",
    "⚙️ Pipeline",
    "📊 Dataset (EDA)",
    "🤖 Models Trained",
    "📈 Evaluation",
    "🔬 Feature Importance",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DETECT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_detect:
    st.markdown("### Analyse a Query")
    st.markdown("Select a model, enter a SQL query or request parameter, and run the detector.")

    # Model picker
    sel_display = st.radio(
        "Model to query",
        options=list(MODEL_DISPLAY.values()),
        index=list(MODEL_DISPLAY.keys()).index(best_model_name),
        horizontal=True,
        help="Choose any of the three trained models. The best model (by F1) is pre-selected.",
    )
    sel_key = {v: k for k, v in MODEL_DISPLAY.items()}[sel_display]
    sel_result = results.get(sel_key, {})
    sel_threshold = float(sel_result.get("threshold", 0.5))

    st.markdown(
        f"<div style='color:#7c8ab8; font-size:0.85rem; margin-bottom:0.8rem;'>"
        f"Decision threshold for <b>{sel_display}</b>: <code>{sel_threshold:.3f}</code>"
        + (f" &nbsp;<span class='best-badge'>⭐ Best Model</span>" if sel_key == best_model_name else "")
        + "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([3, 2], gap="large")

    with left:
        query = st.text_area(
            "SQL text or request parameter",
            value="' OR 1=1 --",
            height=160,
            label_visibility="collapsed",
            placeholder="Enter a SQL query or HTTP parameter value…",
        )
        run = st.button("Analyse", type="primary", use_container_width=True)

    with right:
        if run and query.strip():
            model_obj = load_model(MODEL_FILES[sel_key])
            prob = float(model_obj.predict_proba([query])[0, 1])
            is_attack = prob >= sel_threshold

            if is_attack:
                st.markdown("<div class='badge-attack'>⚠️ SQL Injection Detected</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='badge-benign'>✅ Benign</div>", unsafe_allow_html=True)

            st.markdown("")
            r1, r2 = st.columns(2)
            r1.metric("Malicious Probability", f"{prob:.3f}")
            r2.metric("Threshold", f"{sel_threshold:.3f}")
            st.progress(min(prob, 1.0), text=f"{prob * 100:.1f}% malicious confidence")

            # Show activated handcrafted features
            st.markdown("##### Triggered Pattern Signals")
            eng = SQLiFeatureEngineer()
            vec = eng.transform([query])[0]
            names = list(eng.get_feature_names_out())
            signals = [(names[i], round(float(vec[i]), 4)) for i in range(len(names)) if vec[i] > 0 and names[i].startswith("has_")]
            if signals:
                sig_df = pd.DataFrame(signals, columns=["Signal", "Value"])
                sig_df["Signal"] = sig_df["Signal"].str.replace("has_", "").str.replace("_", " ").str.title()
                st.dataframe(sig_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No binary pattern signals triggered.")
        elif run:
            st.warning("Please enter a query to analyse.")
        else:
            st.markdown(
                "<div style='color:#7c8ab8; padding-top:1rem; font-size:0.9rem; line-height:2;'>"
                "Enter a query on the left and click <strong>Analyse</strong>.<br>"
                "Triggered SQL pattern signals will also be shown below the result."
                "</div>",
                unsafe_allow_html=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PIPELINE OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("### End-to-End Pipeline")
    st.markdown("All stages executed by `python scripts/train.py` and `python scripts/evaluate.py`.")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    stages = [
        ("1", "Data Loading", "Reads the raw CSV dataset (`data/raw/sql_injection_dataset.csv`). "
            "Validates column names (`Query`, `Label`), drops empties, and reports class balance."),
        ("2", "Exploratory Data Analysis (EDA)", "Computes row count, duplicates, label distribution, "
            "query length statistics (min/mean/median/max) and per-label length breakdown. "
            "Saved to `reports/eda_summary.json` and `reports/eda_rows.csv`."),
        ("3", "Train / Test Split", "Stratified 80 / 20 split using `random_state=42`. "
            "Holdout set saved to `artifacts/holdout_test.csv` for evaluation after training."),
        ("4", "Feature Engineering", "Two parallel feature blocks combined via `FeatureUnion`:\n\n"
            "**① Character TF-IDF** — 2–5 character n-grams, max 2 500 features, sublinear TF, lowercase.\n\n"
            "**② Handcrafted SQL signals** — 50 numeric features capturing query length, quote/operator/keyword counts, "
            "tautology detection (`OR 1=1`), UNION SELECT, stacked queries, time delays, hex literals, encoded chars, etc. "
            "Scaled with `StandardScaler` (RF/LR) or `MinMaxScaler` (Naive Bayes)."),
        ("5", "Model Training", "Three scikit-learn classifiers trained independently on the same feature matrix:\n\n"
            "- **Random Forest** — 250 trees, balanced class weights\n"
            "- **Logistic Regression** — liblinear solver, balanced class weights\n"
            "- **Multinomial Naive Bayes** — α = 0.5 (uses MinMaxScaler for non-negative features)\n\n"
            "For each model, the best decision threshold is found by sweeping 0.05 → 0.95 and maximising **F1 score** on the validation set."),
        ("6", "Model Selection", f"Best model chosen by highest F1 score. "
            f"Saved to `artifacts/best_model.joblib`. Individual models saved under `artifacts/models/`. "
            f"Current best: **{MODEL_DISPLAY.get(best_model_name, best_model_name)}**."),
        ("7", "Evaluation", "The saved best model is loaded and scored against the held-out test set. "
            "Outputs: classification metrics (accuracy, precision, recall, F1, ROC-AUC), "
            "confusion matrix (TN/FP/FN/TP), and per-row predictions. "
            "All saved to `reports/`."),
        ("8", "Reporting", "All artefacts exported:\n\n"
            "`artifacts/`: `best_model.joblib`, `model_metadata.json`, `holdout_test.csv`, `models/*.joblib`\n\n"
            "`reports/`: `eda_summary.json`, `eda_rows.csv`, `model_metrics.csv`, `model_metrics.json`, "
            "`feature_importance.csv`, `confusion_matrix.csv`, `holdout_predictions.csv`, `best_model_evaluation.json`"),
    ]

    for step_num, title, desc in stages:
        st.markdown(
            f"""
            <div class='pipeline-step'>
                <div class='step-label'>Step {step_num}</div>
                <div class='step-title'>{title}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Details — {title}", expanded=False):
            st.markdown(desc)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("#### Feature Block Summary")
    fbl, fbr = st.columns(2, gap="large")
    with fbl:
        st.markdown("**Character TF-IDF**")
        for item in ["analyzer = char", "ngram_range = (2, 5)", "max_features = 2 500",
                     "sublinear_tf = True", "lowercase = True"]:
            st.markdown(f"<span class='tag'>{item}</span>", unsafe_allow_html=True)
    with fbr:
        st.markdown("**Handcrafted SQL Signals (50 features)**")
        cats = ["Length & token stats", "Quote / operator / keyword counts",
                "Tautology (OR 1=1)", "UNION SELECT detection",
                "Stacked queries (;)", "Time delays (SLEEP/WAITFOR)",
                "Error probes", "File access", "Hex literals", "Encoded chars",
                "System procs (sp_, xp_)", "Version probes (@@version)"]
        for item in cats:
            st.markdown(f"<span class='tag'>{item}</span>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eda:
    st.markdown("### Dataset — Exploratory Data Analysis")
    if not eda_summary:
        st.info("EDA data not found. Run `python scripts/train.py` to generate it.")
    else:
        total = eda_summary.get("rows", 0)
        attacks  = eda_summary.get("label_counts", {}).get("1", 0)
        benign   = eda_summary.get("label_counts", {}).get("0", 0)
        dups     = eda_summary.get("duplicates", 0)
        ql       = eda_summary.get("query_length", {})
        at_pct   = eda_summary.get("label_percent", {}).get("1", 0)

        e1, e2, e3, e4, e5, e6 = st.columns(6)
        e1.metric("Total Queries",  f"{total:,}")
        e2.metric("Attack Queries", f"{attacks:,}", f"{at_pct * 100:.1f}%")
        e3.metric("Benign Queries", f"{benign:,}", f"{(1 - at_pct) * 100:.1f}%")
        e4.metric("Duplicates",     f"{dups:,}")
        e5.metric("Avg Length",     f"{ql.get('mean', 0):.0f} chars")
        e6.metric("Max Length",     f"{ql.get('max', 0):,} chars")

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            st.markdown("#### Label Distribution")
            st.bar_chart(
                pd.DataFrame({"Count": [benign, attacks]}, index=["Benign (0)", "Attack (1)"]),
                color=["#4f8ef7"],
            )
        with col_r:
            st.markdown("#### Avg Query Length by Label")
            lbl = eda_summary.get("length_by_label", {}).get("mean", {})
            if lbl:
                st.bar_chart(
                    pd.DataFrame({"Avg Length (chars)": [lbl.get("0", 0), lbl.get("1", 0)]},
                                  index=["Benign", "Attack"]),
                    color=["#f7a34f"],
                )

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("#### Query Length Statistics by Label")
        lbl_full = eda_summary.get("length_by_label", {})
        if lbl_full:
            stat_rows = []
            for stat in ["mean", "std", "min", "25%", "50%", "75%", "max"]:
                row = {"Statistic": stat.capitalize()}
                for lk, ln in [("0", "Benign"), ("1", "Attack")]:
                    v = lbl_full.get(stat, {}).get(lk)
                    row[ln] = f"{v:.1f}" if v is not None else "—"
                stat_rows.append(row)
            st.dataframe(pd.DataFrame(stat_rows).set_index("Statistic"), use_container_width=True)

        if not eda_rows.empty:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("#### Sample Rows")
            display_cols = [c for c in ["Query", "Label", "query_length", "word_count", "special_char_count"] if c in eda_rows.columns]
            st.dataframe(
                eda_rows[display_cols].head(20).rename(columns={
                    "query_length": "Length", "word_count": "Words", "special_char_count": "Special Chars"
                }),
                use_container_width=True, hide_index=True,
            )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODELS TRAINED
# ═══════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.markdown("### Models Trained")
    st.markdown("All three models were trained on the same feature matrix with per-model threshold optimisation (max F1 on validation).")
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    MODEL_ICONS = {"random_forest": "🌲", "logistic_regression": "📉", "naive_bayes": "📐"}
    MODEL_DESC = {
        "random_forest":      "Ensemble of 250 decision trees. Uses balanced class weights and parallel fitting (`n_jobs=-1`). Captures non-linear SQLi patterns and interaction effects between TF-IDF and handcrafted features.",
        "logistic_regression": "Linear classifier with liblinear solver. Fast, interpretable, and robust on high-dimensional sparse data. Balanced class weights handle the label imbalance.",
        "naive_bayes":        "Multinomial Naive Bayes with α = 0.5. Requires non-negative features — uses `MinMaxScaler` instead of `StandardScaler`. Fastest to train and predict.",
    }

    for model_key, model_display in MODEL_DISPLAY.items():
        res = results.get(model_key, {})
        m = res.get("metrics", {})
        cm = res.get("confusion_matrix", {})
        thr = res.get("threshold", "—")
        is_best = model_key == best_model_name

        header = (
            f"{MODEL_ICONS[model_key]} **{model_display}**"
            + ("  ⭐ *Best Model*" if is_best else "")
        )
        st.markdown(f"#### {header}")
        st.markdown(MODEL_DESC[model_key])

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Threshold", f"{thr:.3f}" if isinstance(thr, float) else thr)
        mc2.metric("F1 Score",  f'{m.get("f1_score", 0):.4f}')
        mc3.metric("Recall",    f'{m.get("recall", 0):.4f}')
        mc4.metric("Precision", f'{m.get("precision", 0):.4f}')
        mc5.metric("ROC-AUC",   f'{m.get("roc_auc", 0):.4f}')

        if cm:
            with st.expander("Confusion Matrix (on held-out validation split)"):
                cc1, cc2, cc3, cc4 = st.columns(4)
                cc1.metric("TN", f'{cm.get("tn", 0):,}')
                cc2.metric("FP", f'{cm.get("fp", 0):,}')
                cc3.metric("FN", f'{cm.get("fn", 0):,}', help="Missed attacks ⚠️")
                cc4.metric("TP", f'{cm.get("tp", 0):,}')

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Side-by-side comparison chart ────────────────────────────────────────
    st.markdown("### Comparison Chart")
    if not metrics_df.empty:
        chart_cols = [c for c in ["f1_score", "recall", "precision", "roc_auc"] if c in metrics_df.columns]
        chart_df = metrics_df.set_index("model")[chart_cols].rename(
            columns={c: c.replace("_", " ").title() for c in chart_cols}
        )
        chart_df.index = [MODEL_DISPLAY.get(i, i) for i in chart_df.index]
        st.bar_chart(chart_df)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### Best Model Evaluation — Holdout Test Set")
    st.markdown(
        f"Evaluated on the **20% holdout** split "
        f"({eval_report.get('evaluated_rows', '—')} rows) using "
        f"**{MODEL_DISPLAY.get(eval_report.get('model', ''), eval_report.get('model', '—'))}** "
        f"at threshold `{eval_report.get('threshold', '—')}`."
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    em = eval_report.get("metrics", {})
    ecm = eval_report.get("confusion_matrix", {})

    # ── Metric strip ──────────────────────────────────────────────────────────
    st.markdown("#### Classification Metrics")
    v1, v2, v3, v4, v5, v6 = st.columns(6)
    v1.metric("Accuracy",          f'{em.get("accuracy", 0):.4f}')
    v2.metric("F1 Score",          f'{em.get("f1_score", 0):.4f}')
    v3.metric("Recall",            f'{em.get("recall", 0):.4f}')
    v4.metric("Precision",         f'{em.get("precision", 0):.4f}')
    v5.metric("ROC-AUC",           f'{em.get("roc_auc", 0):.4f}')
    v6.metric("Avg Precision",     f'{em.get("average_precision", 0):.4f}')

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("#### Confusion Matrix")
    if ecm:
        tn, fp, fn, tp = ecm.get("tn", 0), ecm.get("fp", 0), ecm.get("fn", 0), ecm.get("tp", 0)
        total_eval = eval_report.get("evaluated_rows", tn + fp + fn + tp)

        cc1, cc2, cc3, cc4 = st.columns(4)
        cc1.metric("True Negatives (TN)",  f"{tn:,}", help="Benign correctly identified")
        cc2.metric("False Positives (FP)", f"{fp:,}", help="Benign flagged as attack")
        cc3.metric("False Negatives (FN)", f"{fn:,}", help="Attacks missed ⚠️")
        cc4.metric("True Positives (TP)",  f"{tp:,}", help="Attacks correctly caught")

        # ── Derived rates ─────────────────────────────────────────────────────
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("#### Derived Rates")
        r1, r2, r3 = st.columns(3)
        miss_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0
        detection = tp / (fn + tp) if (fn + tp) > 0 else 0
        r1.metric("Detection Rate (TPR)",  f"{detection * 100:.2f}%", help="= Recall")
        r2.metric("Miss Rate (FNR)",       f"{miss_rate * 100:.2f}%", help="Fraction of attacks missed")
        r3.metric("False Alarm Rate (FPR)",f"{fpr * 100:.2f}%",       help="Fraction of benign flagged")

        st.markdown(f"*Evaluated on **{total_eval:,}** rows — model: "
                    f"**{MODEL_DISPLAY.get(eval_report.get('model', ''), eval_report.get('model', ''))}**.*")

    # ── Full metrics table ────────────────────────────────────────────────────
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("#### All Models — Metrics Table")
    if not metrics_df.empty:
        pretty = metrics_df.copy()
        pretty["model"] = pretty["model"].map(MODEL_DISPLAY).fillna(pretty["model"])
        pretty.columns = [c.replace("_", " ").title() for c in pretty.columns]
        st.dataframe(pretty.set_index("Model"), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_features:
    st.markdown("### Feature Importance — Best Model")
    st.markdown(
        "Feature importances are derived from the **best model** "
        f"(**{MODEL_DISPLAY.get(best_model_name, best_model_name)}**). "
        "For Random Forest, this is mean impurity decrease (`feature_importances_`). "
        "For Logistic Regression, it is the absolute value of the coefficient."
    )
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    if feat_imp.empty:
        st.info("Feature importance not available — may not be supported for the selected model type.")
    else:
        # ── Filter by feature type ────────────────────────────────────────────
        col_filter, col_slider = st.columns([2, 1])
        with col_filter:
            feat_type = st.radio(
                "Feature type",
                ["All", "Handcrafted (num:)", "TF-IDF (char:)"],
                horizontal=True,
            )
        with col_slider:
            top_n = st.slider("Top N", min_value=5, max_value=40, value=20, step=5)

        if feat_type == "Handcrafted (num:)":
            filtered = feat_imp[feat_imp["feature"].str.startswith("num:")]
        elif feat_type == "TF-IDF (char:)":
            filtered = feat_imp[feat_imp["feature"].str.startswith("char:")]
        else:
            filtered = feat_imp

        top_features = filtered.head(top_n).copy()
        top_features["feature"] = top_features["feature"].str.replace("num:", "").str.replace("char:", "")

        fl, fr = st.columns([2, 3], gap="large")
        with fl:
            st.markdown("#### Feature Table")
            st.dataframe(
                top_features.rename(columns={"feature": "Feature", "importance": "Importance"}),
                use_container_width=True, hide_index=True,
            )
        with fr:
            st.markdown("#### Importance Chart")
            st.bar_chart(top_features.set_index("feature")["importance"], color="#4f8ef7")
