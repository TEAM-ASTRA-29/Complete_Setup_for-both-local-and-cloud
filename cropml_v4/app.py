"""
CropML Research Studio v4.0 — app.py
=====================================
Entry point. Run with:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import base64, json
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

from backend.models      import (REGRESSION_MODELS, CLASSIFICATION_MODELS,
                                  MODEL_INFO, recommend_models,
                                  train_one, simulate_cloud_vs_local)
from backend.preprocessor import DataProfiler, PreprocessingPipeline
from backend.report       import generate_pdf
from backend.session      import make_sid, save_session, load_sessions, delete_session
from frontend.components  import (CSS, kpi, badge, info_box, sec_hdr,
                                   metric_comparison_table,
                                   chart_primary_metric, chart_error_metrics,
                                   chart_computational, chart_actual_vs_predicted,
                                   chart_residuals, chart_residual_hist,
                                   chart_feature_importance, chart_radar,
                                   chart_confusion_matrix, chart_cloud_vs_local,
                                   color)
from frontend.hyperparams import hyperparam_ui

import warnings
warnings.filterwarnings("ignore")

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_sessions")
os.makedirs(PERSIST_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="CropML Research Studio",
                   page_icon="🌾", layout="wide",
                   initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in [("sessions",[]), ("active_session",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌾 CropML Studio")
    st.caption("v4.0 · Regression & Classification")
    st.divider()
    page = st.radio("", ["🔬 Train","📊 Analysis","📐 Compare","⚖️ Cross-Session","📁 History"],
                    label_visibility="collapsed")
    st.divider()

    # Merge disk sessions
    disk = load_sessions()
    existing = {s["id"] for s in st.session_state.sessions}
    for ds in disk:
        if ds["id"] not in existing:
            st.session_state.sessions.append(ds)
            existing.add(ds["id"])

    st.markdown(f"**Sessions:** `{len(st.session_state.sessions)}`")
    if st.session_state.sessions:
        labels = [f"{s['name']} ({s['timestamp'][:10]})"
                  for s in st.session_state.sessions]
        idx = st.selectbox("Active session", range(len(labels)),
                           format_func=lambda i: labels[i])
        st.session_state.active_session = st.session_state.sessions[idx]
    st.divider()
    st.caption("Upload any CSV · auto-recommends models · exports PDF")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔬 Train":
    st.markdown("# 🔬 Train New Session")

    # ── File Upload ──────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload CSV Dataset (raw or pre-processed)", type=["csv"])
    if not uploaded:
        st.info("Upload a CSV file — raw data or already-preprocessed, both supported.")
        st.stop()

    df_raw = pd.read_csv(uploaded)
    profiler = DataProfiler(df_raw)
    profile  = profiler.profile()

    # Dataset overview cards
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi("Rows",    f"{profile['n_rows']:,}"),                          unsafe_allow_html=True)
    c2.markdown(kpi("Columns", profile["n_cols"]),                                  unsafe_allow_html=True)
    c3.markdown(kpi("Numeric", len(profile["numeric_cols"]),   cls="g"),            unsafe_allow_html=True)
    c4.markdown(kpi("Categorical",len(profile["categorical_cols"]),cls="o"),        unsafe_allow_html=True)
    c5.markdown(kpi("Nulls",   profile["null_total"],          cls="p"),            unsafe_allow_html=True)
    st.markdown("")

    with st.expander("📋 Dataset Preview & Column Details"):
        st.dataframe(df_raw.head(8), use_container_width=True)
        detail_df = pd.DataFrame(profile["col_details"])
        st.dataframe(detail_df, use_container_width=True)

    # Detect pre-processed data and show smart banner
    if profile["looks_preprocessed"]:
        st.success("✅ **Data looks pre-processed** (all numeric, no nulls detected). "
                   "You can skip preprocessing steps below or use them as-is.")
    else:
        st.warning("⚠️ **Raw data detected** (has categorical columns or nulls). "
                   "Configure preprocessing below.")

    # ── Task & Target ────────────────────────────────────────────────────────
    st.markdown(sec_hdr("⚙️","Task & Target Configuration"), unsafe_allow_html=True)
    ca,cb,cc = st.columns([1,2,2])
    task   = ca.selectbox("Task Type", ["Regression","Classification"])
    target = cb.selectbox("Target Column", df_raw.columns.tolist(),
                           index=len(df_raw.columns)-1)
    drop_cols = cc.multiselect("Drop Columns (IDs, leakage, etc.)",
                                [c for c in df_raw.columns if c!=target])

    # ── Preprocessing ────────────────────────────────────────────────────────
    st.markdown(sec_hdr("🔧","Preprocessing Pipeline"), unsafe_allow_html=True)

    already_clean = st.checkbox(
        "⚡ Data is already preprocessed — skip encoding & imputation steps",
        value=profile["looks_preprocessed"],
        help="Check this if your CSV has only numeric columns and no missing values.")

    if not already_clean:
        pc1,pc2,pc3,pc4 = st.columns(4)
        handle_cats  = pc1.selectbox("Categorical Encoding",
                                      ["label","onehot","drop","none"],
                                      format_func=lambda x: {"label":"Label Encode","onehot":"One-Hot",
                                                              "drop":"Drop","none":"None"}[x])
        handle_nulls = pc2.selectbox("Null Handling",
                                      ["mean","median","mode","drop","none"],
                                      format_func=lambda x: {"mean":"Mean Fill","median":"Median Fill",
                                                              "mode":"Mode Fill","drop":"Drop Rows",
                                                              "none":"None"}[x])
        handle_dups  = pc3.checkbox("Remove Duplicate Rows", value=False)
        drop_low_var = pc4.checkbox("Drop Low-Variance Features", value=False)
        scale_method = st.selectbox("Feature Scaling",
                                     ["none","standard","minmax"],
                                     format_func=lambda x: {"none":"No Scaling",
                                                             "standard":"StandardScaler (Z-score)",
                                                             "minmax":"MinMaxScaler [0,1]"}[x])
    else:
        handle_cats=handle_nulls="none"; handle_dups=drop_low_var=False; scale_method="none"
        st.info("Preprocessing skipped. Features will be passed directly to models.")

    # Run pipeline
    pipe_cfg = {"target_col":target,"drop_cols":drop_cols,
                "handle_cats":handle_cats,"handle_nulls":handle_nulls,
                "handle_dups":handle_dups,"drop_low_var":drop_low_var,
                "scale_method":scale_method,"already_clean":already_clean}
    pipeline = PreprocessingPipeline(pipe_cfg)
    try:
        X, y = pipeline.fit_transform(df_raw)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        st.stop()

    final_feats = pipeline.feature_names
    if pipeline.get_log():
        with st.expander("📝 Preprocessing Log"):
            for line in pipeline.get_log():
                st.markdown(f"• {line}")

    st.caption(f"**{len(final_feats)} features** after preprocessing → "
               f"{', '.join(final_feats[:12])}{'...' if len(final_feats)>12 else ''}")

    if task == "Classification":
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)))

    # ── Split ────────────────────────────────────────────────────────────────
    la,lb,lc = st.columns(3)
    tsize = la.slider("Test size", 0.10, 0.40, 0.20, 0.05)
    rseed = lb.number_input("Random seed", 0, 9999, 42)
    scale_models = lc.checkbox(
        "Scale features for model training (SVR/SVC/KNN/MLP)",
        value=(scale_method=="none" and not already_clean))

    X_tr,X_te,y_tr,y_te = train_test_split(X, y, test_size=tsize, random_state=int(rseed))
    y_test_list = y_te.tolist()

    # ── Smart Model Recommendation ───────────────────────────────────────────
    st.markdown(sec_hdr("🤖","Smart Model Recommendation"), unsafe_allow_html=True)

    rec_mode = st.radio("Recommendation Mode",
                         ["🧠 Auto (dataset-aware)","📏 Rule-based (you specify)","✋ Manual selection"],
                         horizontal=True)

    user_rules = {}
    if rec_mode == "📏 Rule-based (you specify)":
        st.markdown("**Set your priorities:**")
        rr1,rr2,rr3,rr4,rr5 = st.columns(5)
        user_rules["prefer_speed"]         = rr1.checkbox("⚡ Prefer speed")
        user_rules["prefer_accuracy"]      = rr2.checkbox("🎯 Prefer accuracy")
        user_rules["prefer_interpretable"] = rr3.checkbox("🔍 Interpretable")
        user_rules["small_model"]          = rr4.checkbox("💾 Small model")
        max_t = rr5.number_input("Max train time (s)", 0, 300, 0)
        if max_t > 0: user_rules["max_training_time_s"] = max_t

    registry = REGRESSION_MODELS if task=="Regression" else CLASSIFICATION_MODELS

    if rec_mode != "✋ Manual selection":
        rec_df  = pd.concat([X,y.rename(target)], axis=1) if hasattr(y,"rename") else X
        rec_out = recommend_models(df_raw, target, task, user_rules if user_rules else None)
        ranked  = rec_out["ranked"]
        dp      = rec_out["dataset_profile"]

        st.markdown(f"""
        <div class='ib' style='margin-bottom:.8rem;'>
        📊 <b>Dataset profile:</b>
        {dp['n_rows']:,} rows · {dp['n_features']} features ·
        {badge('Has nulls' if dp['has_nulls'] else 'No nulls', 'bo' if dp['has_nulls'] else 'bg')}
        {badge(f"Cat ratio {dp['cat_ratio']:.0%}", 'bp')}
        </div>""", unsafe_allow_html=True)

        st.markdown("**Top 5 recommended models:**")
        top5 = [name for name,score in ranked[:5]]
        rec_cols = st.columns(5)
        for i,(name,score) in enumerate(ranked[:5]):
            inf = MODEL_INFO.get(registry.get(name,""),{})
            medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][i]
            rec_cols[i].markdown(
                f"<div class='rec-card'>"
                f"<div class='rec-rank'>{medal} Rank {i+1}</div>"
                f"<div class='rec-name'>{name}</div>"
                f"<div class='rec-score'>Score: {score}</div>"
                f"<div style='font-size:.72rem;color:#7d8590;margin-top:.3rem;'>"
                f"{inf.get('type','')}</div></div>",
                unsafe_allow_html=True)

        with st.expander("🔍 Full recommendation breakdown"):
            for name, score in ranked:
                reasons = rec_out["reasons"][name]
                inf     = MODEL_INFO.get(registry.get(name,""), {})
                st.markdown(f"**{name}** — Score: `{score}` · {inf.get('type','')}")
                for r in reasons:
                    st.markdown(f"  {r}")
                st.markdown(f"  *Strengths:* {inf.get('strengths','')}  "
                            f"*Weaknesses:* {inf.get('weaknesses','')}")
                st.divider()

        default_sel = top5[:4]
    else:
        default_sel = list(registry.keys())[:4]

    # ── Model Selection ──────────────────────────────────────────────────────
    st.markdown(sec_hdr("✅","Select Models to Train"), unsafe_allow_html=True)
    selected = st.multiselect("Models (≥1)", list(registry.keys()),
                               default=[d for d in default_sel if d in registry])
    if not selected:
        st.warning("Select at least one model.")
        st.stop()

    # ── Hyperparameters ──────────────────────────────────────────────────────
    st.markdown(sec_hdr("🎛️","Hyperparameter Configuration"), unsafe_allow_html=True)
    model_cfgs = {}
    tabs = st.tabs(selected)
    for tab, mname in zip(tabs, selected):
        with tab:
            inf = MODEL_INFO.get(registry.get(mname,""),{})
            if inf:
                st.caption(f"**{inf.get('type','')}** · "
                           f"✅ {inf.get('strengths','')} · "
                           f"⚠️ {inf.get('weaknesses','')}")
            mkey, mparams = hyperparam_ui(mname, f"hp_{mname}", task)
            model_cfgs[mname] = (mkey, mparams)

    # ── Session name + Train ─────────────────────────────────────────────────
    st.markdown(sec_hdr("🚀","Launch Training"), unsafe_allow_html=True)
    sn1,sn2 = st.columns([3,1])
    sname = sn1.text_input("Session Name",
               value=f"{uploaded.name.replace('.csv','')}_{task[:3]}_{datetime.now().strftime('%H%M')}")

    if sn2.button("🚀 Train All", type="primary", use_container_width=True):
        prog    = st.progress(0, text="Starting…")
        results = []
        for i,(mname,(mkey,mparams)) in enumerate(model_cfgs.items()):
            prog.progress(i/len(model_cfgs), text=f"Training {mname}…")
            try:
                r = train_one(mname,mkey,mparams,
                              X_tr.copy(),X_te.copy(),
                              y_tr,y_te,task,scale_models,PERSIST_DIR)
                results.append(r)
            except Exception as e:
                st.warning(f"⚠️ {mname} failed: {e}")
        prog.progress(1.0, text="✅ All models trained")

        session = {
            "id":make_sid(sname),"name":sname,
            "timestamp":datetime.now().isoformat(),
            "dataset":uploaded.name,"task":task,
            "n_samples":len(X),"n_features":len(final_feats),
            "feature_names":final_feats,"target_col":target,
            "test_size":tsize,"scale":scale_models,
            "y_test":y_test_list,"model_results":results,
            "prep_log":pipeline.get_log(),
            "already_clean":already_clean,
        }
        st.session_state.sessions.append(session)
        st.session_state.active_session = session
        save_session(session)
        st.success(f"✅ {len(results)} model(s) trained. Session **{sname}** saved.")

        mk = "r2" if task=="Regression" else "accuracy"
        ml = "R²"  if task=="Regression" else "Accuracy"
        best_name = max(results,key=lambda r:r.get(mk,0))["name"]
        cols = st.columns(min(len(results),6))
        for col,r in zip(cols,results):
            star = " ⭐" if r["name"]==best_name else ""
            col.markdown(kpi(r["name"]+star,f"{r.get(mk,0):.4f}",ml,
                             "g" if r["name"]==best_name else ""),
                         unsafe_allow_html=True)
        st.info("➡️ Go to **📊 Analysis** for full visualisations and PDF export.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analysis":
    st.markdown("# 📊 Session Analysis")
    if not st.session_state.active_session:
        st.info("No active session. Train one first.")
        st.stop()

    s       = st.session_state.active_session
    results = s["model_results"]
    task    = s["task"]
    feats   = s["feature_names"]
    y_test  = np.array(s.get("y_test",[]))

    # Header
    st.markdown(f"""
    <div style='display:flex;align-items:flex-start;gap:1rem;margin-bottom:1rem;'>
      <div style='flex:1;'>
        <div style='font-size:.7rem;color:#7d8590;text-transform:uppercase;letter-spacing:.1em;'>Active Session</div>
        <div style='font-size:1.6rem;font-weight:700;letter-spacing:-.02em;'>{s['name']}</div>
        <div style='margin-top:.4rem;'>
          {badge(s['dataset'],'bb')}&nbsp;
          {badge(task,'bg')}&nbsp;
          {badge(f"{s['n_samples']:,} rows · {s['n_features']} features",'bo')}&nbsp;
          {badge(f"{len(results)} models",'bp')}
        </div>
      </div>
      <div style='text-align:right;font-size:.78rem;color:#7d8590;padding-top:.5rem;'>
        {s['timestamp'][:19]}
      </div>
    </div>""", unsafe_allow_html=True)

    # Prep log
    if s.get("prep_log"):
        with st.expander("📝 Preprocessing Log"):
            for line in s["prep_log"]:
                st.markdown(f"• {line}")

    # PDF
    if st.button("📥 Generate & Download PDF Report"):
        with st.spinner("Building PDF…"):
            pdf_bytes = generate_pdf(s)
        b64   = base64.b64encode(pdf_bytes).decode()
        fname = f"{s['name']}_report.pdf"
        st.markdown(f"<a class='dlbtn' href='data:application/pdf;base64,{b64}' "
                    f"download='{fname}'>⬇ Download {fname}</a>", unsafe_allow_html=True)
    st.divider()

    # KPI cards
    if task == "Regression":
        best_r2   = max(results,key=lambda r:r.get("r2",0))
        best_rmse = min(results,key=lambda r:r.get("rmse",999))
        best_mae  = min(results,key=lambda r:r.get("mae",999))
        fastest   = min(results,key=lambda r:r["runtime"])
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(kpi("Best R²",  f"{best_r2.get('r2',0):.4f}", best_r2["name"],  "g"), unsafe_allow_html=True)
        c2.markdown(kpi("Best RMSE",f"{best_rmse.get('rmse',0):.4f}",best_rmse["name"],""), unsafe_allow_html=True)
        c3.markdown(kpi("Best MAE", f"{best_mae.get('mae',0):.4f}",  best_mae["name"], "p"), unsafe_allow_html=True)
        c4.markdown(kpi("Fastest",  f"{fastest['runtime']:.3f}s",  fastest["name"],  "y"), unsafe_allow_html=True)
    else:
        best_acc  = max(results,key=lambda r:r.get("accuracy",0))
        best_f1   = max(results,key=lambda r:r.get("f1",0))
        fastest   = min(results,key=lambda r:r["runtime"])
        best_prec = max(results,key=lambda r:r.get("precision",0))
        c1,c2,c3,c4 = st.columns(4)
        c1.markdown(kpi("Best Accuracy", f"{best_acc.get('accuracy',0):.4f}", best_acc["name"],  "g"), unsafe_allow_html=True)
        c2.markdown(kpi("Best F1",       f"{best_f1.get('f1',0):.4f}",        best_f1["name"],   ""), unsafe_allow_html=True)
        c3.markdown(kpi("Best Precision",f"{best_prec.get('precision',0):.4f}",best_prec["name"],"p"), unsafe_allow_html=True)
        c4.markdown(kpi("Fastest",       f"{fastest['runtime']:.3f}s",        fastest["name"],   "y"), unsafe_allow_html=True)
    st.markdown("")

    # Inferences
    st.markdown(sec_hdr("🧠","Automated Inferences"), unsafe_allow_html=True)
    mk = "r2" if task=="Regression" else "accuracy"
    s_mk  = sorted(results,key=lambda r:r.get(mk,0),reverse=True)
    delta = s_mk[0].get(mk,0)-s_mk[-1].get(mk,0) if len(s_mk)>1 else 0
    rrt   = max(r["runtime"] for r in results)/max(min(r["runtime"] for r in results),.001)
    light = min(results,key=lambda r:r["model_size_mb"])
    ci1,ci2,ci3 = st.columns(3)
    ml_lbl = "R²" if task=="Regression" else "Accuracy"
    ci1.markdown(info_box(f"📈 <b>Best {ml_lbl}:</b><br>{s_mk[0]['name']}<br>"
                           f"<code>{ml_lbl} = {s_mk[0].get(mk,0):.4f} · Δ{delta:.4f} vs worst</code>"), unsafe_allow_html=True)
    ci2.markdown(info_box(f"⚡ <b>Fastest training:</b><br>{fastest['name']}<br>"
                           f"<code>{fastest['runtime']:.3f}s · {rrt:.1f}× vs slowest</code>"), unsafe_allow_html=True)
    ci3.markdown(info_box(f"💾 <b>Lightest model:</b><br>{light['name']}<br>"
                           f"<code>{light['model_size_mb']:.3f} MB on disk</code>"), unsafe_allow_html=True)
    st.markdown("")

    # Primary metric
    st.markdown(sec_hdr("📊",f"{'R²' if task=='Regression' else 'Accuracy'} Comparison"), unsafe_allow_html=True)
    st.plotly_chart(chart_primary_metric(results,task), use_container_width=True)

    # Error metrics
    st.markdown(sec_hdr("📉","Error / Classification Metrics"), unsafe_allow_html=True)
    st.plotly_chart(chart_error_metrics(results,task), use_container_width=True)

    # Actual vs Predicted / Confusion matrix
    if task == "Regression":
        st.markdown(sec_hdr("🎯","Actual vs Predicted"), unsafe_allow_html=True)
        sel_m      = st.selectbox("Select model",[r["name"] for r in results],key="avp")
        r_sel      = next(r for r in results if r["name"]==sel_m)
        y_pred_sel = np.array(r_sel["y_pred"])
        if len(y_test)==len(y_pred_sel):
            st.plotly_chart(chart_actual_vs_predicted(y_test,y_pred_sel,sel_m), use_container_width=True)
            st.markdown(sec_hdr("📉","Residual Analysis"), unsafe_allow_html=True)
            t1,t2 = st.tabs(["Residual Scatter","Error Distribution"])
            with t1:
                st.plotly_chart(chart_residuals(y_test,y_pred_sel,sel_m), use_container_width=True)
            with t2:
                st.plotly_chart(chart_residual_hist(results,y_test), use_container_width=True)
    else:
        st.markdown(sec_hdr("🔲","Confusion Matrix"), unsafe_allow_html=True)
        cm_sel = st.selectbox("Model",[r["name"] for r in results],key="cm_s")
        r_cm   = next(r for r in results if r["name"]==cm_sel)
        st.plotly_chart(chart_confusion_matrix(r_cm["conf_matrix"],cm_sel), use_container_width=True)

    # Feature importance
    fi_res = [r for r in results if r.get("feature_importances")]
    if fi_res:
        st.markdown(sec_hdr("🔑","Feature Importance"), unsafe_allow_html=True)
        fi_tabs = st.tabs([r["name"] for r in fi_res])
        for tab,r in zip(fi_tabs,fi_res):
            with tab:
                st.plotly_chart(chart_feature_importance(feats,r["feature_importances"],
                                                          r["name"],results.index(r)),
                                use_container_width=True)

    # Computational performance
    st.markdown(sec_hdr("⚙️","Computational Performance"), unsafe_allow_html=True)
    st.plotly_chart(chart_computational(results), use_container_width=True)

    # Radar
    st.markdown(sec_hdr("🕸️","Model Radar Profile"), unsafe_allow_html=True)
    st.plotly_chart(chart_radar(results,task), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — METRIC COMPARISON (per session)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Compare":
    st.markdown("# 📐 Model Comparison — Detailed Metrics")
    if not st.session_state.active_session:
        st.info("No active session.")
        st.stop()

    s       = st.session_state.active_session
    results = s["model_results"]
    task    = s["task"]

    st.markdown(f"Session: **{s['name']}** · {badge(task,'bg' if task=='Regression' else 'bb')} · "
                f"{len(results)} models", unsafe_allow_html=True)
    st.divider()

    # Full metric comparison table
    st.markdown(sec_hdr("📋","Full Metric Comparison Table"), unsafe_allow_html=True)
    st.markdown("<p style='font-size:.82rem;color:#7d8590;'>"
                "🟢 Green = best value for that metric. 🟠 Orange = worst value.</p>",
                unsafe_allow_html=True)
    st.markdown(metric_comparison_table(results,task), unsafe_allow_html=True)
    st.markdown("")

    # Metric definitions
    with st.expander("📖 Metric Definitions & When to Use Each"):
        if task == "Regression":
            defs = {
                "R²": ("Coefficient of Determination","Range [0,1]. Measures how much variance in the "
                       "target is explained by the model. R²=1 is perfect; R²=0 means the model "
                       "does no better than predicting the mean.","Use when: you need a normalised "
                       "goodness-of-fit score comparable across datasets."),
                "RMSE": ("Root Mean Squared Error","Same unit as target. Penalises large errors "
                         "heavily due to squaring. Lower=better.","Use when: large errors are "
                         "especially costly (e.g. financial forecasting)."),
                "MAE": ("Mean Absolute Error","Average absolute difference between prediction "
                        "and actual. Robust to outliers. Lower=better.","Use when: you want an "
                        "error metric in the same units as the target without outlier sensitivity."),
                "MSE": ("Mean Squared Error","Square of RMSE. Useful when penalising large "
                        "errors mathematically. Lower=better.","Use when: optimising loss functions "
                        "or comparing with gradient-based methods."),
            }
        else:
            defs = {
                "Accuracy": ("Accuracy","Fraction of correct predictions. Range [0,1].","Use when: "
                             "classes are balanced. Misleading on imbalanced datasets."),
                "F1": ("F1 Score","Harmonic mean of Precision and Recall. Range [0,1].","Use when: "
                       "classes are imbalanced or both precision and recall matter equally."),
                "Precision": ("Precision","Of all predicted positives, fraction that are truly positive.",
                              "Use when: false positives are costly (e.g. spam detection)."),
                "Recall": ("Recall / Sensitivity","Of all actual positives, fraction the model found.",
                           "Use when: false negatives are costly (e.g. disease detection)."),
                "AUC": ("Area Under ROC Curve","Measures model's ability to discriminate between classes. "
                        "Range [0,1]; 0.5=random.","Use when: evaluating ranking/probability quality "
                        "independent of threshold."),
            }
        for name,(full,defn,usage) in defs.items():
            with st.container():
                st.markdown(f"**{name} — {full}**")
                st.markdown(f"  {defn}")
                st.markdown(f"  _{usage}_")
                st.markdown("")

    # Grouped bar for all metrics
    st.markdown(sec_hdr("📊","Visual Metric Breakdown"), unsafe_allow_html=True)
    if task == "Regression":
        metrics_to_plot = [("R²","r2"),("RMSE","rmse"),("MAE","mae")]
    else:
        metrics_to_plot = [("Accuracy","accuracy"),("F1","f1"),
                           ("Precision","precision"),("Recall","recall")]

    import plotly.graph_objects as go
    from frontend.components import theme as _theme, color as _clr

    metric_tab_names = [m[0] for m in metrics_to_plot]
    m_tabs = st.tabs(metric_tab_names)
    for tab,(label,key) in zip(m_tabs,metrics_to_plot):
        with tab:
            vals = [r.get(key,0) for r in results]
            names= [r["name"] for r in results]
            fig  = go.Figure([go.Bar(
                x=names, y=vals,
                marker=dict(color=[_clr(i) for i in range(len(results))],line=dict(width=0)),
                text=[f"{v:.4f}" for v in vals], textposition="outside",
                hovertemplate=f"<b>%{{x}}</b><br>{label} = %{{y:.4f}}<extra></extra>"
            )])
            fig.update_layout(title=f"{label} — all models", **_theme(400))
            st.plotly_chart(fig, use_container_width=True)

    # Radar again here for easy reference
    st.markdown(sec_hdr("🕸️","Radar Profile"), unsafe_allow_html=True)
    st.plotly_chart(chart_radar(results,task), use_container_width=True)

    # Cloud vs Local
    st.markdown(sec_hdr("☁️","Cloud vs Local Deployment Simulation"), unsafe_allow_html=True)
    st.markdown("<p style='font-size:.82rem;color:#7d8590;'>"
                "Estimate how training would perform on various AWS instance types "
                "vs your current local machine.</p>", unsafe_allow_html=True)

    cvc1,cvc2 = st.columns([2,2])
    cloud_model = cvc1.selectbox("Model to simulate", [r["name"] for r in results], key="cloud_m")
    cloud_spec  = cvc2.selectbox("Cloud Instance",
                                  ["t3.micro","t3.medium","t3.large",
                                   "c5.xlarge","c5.4xlarge","m5.4xlarge","p3.2xlarge"],
                                  index=1)
    r_cloud = next(r for r in results if r["name"]==cloud_model)
    cloud_data  = simulate_cloud_vs_local(r_cloud, task, cloud_spec)

    # Store in session for PDF
    s["cloud_comparison"] = {**cloud_data, "model_name": cloud_model}
    save_session(s)

    local = cloud_data["local"]
    cloud = cloud_data["cloud"]
    cv1,cv2,cv3,cv4 = st.columns(4)
    cv1.markdown(kpi("Local Runtime",   f"{local['runtime_s']:.4f}s",  f"{local['vcpu']} vCPU / {local['ram_gb']}GB"), unsafe_allow_html=True)
    cv2.markdown(kpi("Cloud Runtime",   f"{cloud['runtime_s']:.4f}s",  cloud["instance"],   "g"), unsafe_allow_html=True)
    cv3.markdown(kpi("Speedup",         f"{cloud['speedup']}×",        "cloud vs local",    "y"), unsafe_allow_html=True)
    cv4.markdown(kpi("Cost/Run",        f"${cloud['cost_cents']/100:.6f}", "USD estimate",  "p"), unsafe_allow_html=True)
    st.markdown("")
    st.plotly_chart(chart_cloud_vs_local(cloud_data), use_container_width=True)

    st.markdown(info_box(
        f"☁️ On <b>{cloud_spec}</b> ({cloud['vcpu']} vCPU / {cloud['ram_gb']}GB RAM): "
        f"estimated runtime <b>{cloud['runtime_s']:.4f}s</b> "
        f"({'faster' if cloud['speedup']>1 else 'slower'} than local by {cloud['speedup']}×). "
        f"Estimated cost per training run: <b>${cloud['cost_cents']/100:.6f}</b> USD. "
        f"Note: these are estimates based on relative CPU scaling; actual results may vary.",
        "g" if cloud['speedup']>1 else "o"),
        unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — CROSS-SESSION COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Cross-Session":
    st.markdown("# ⚖️ Cross-Session Comparison")
    if len(st.session_state.sessions) < 2:
        st.info("Train at least 2 sessions to compare.")
        st.stop()

    opts  = [f"{s['name']} | {s['dataset']} ({s['task']}) · {s['timestamp'][:10]}"
             for s in st.session_state.sessions]
    sel   = st.multiselect("Sessions to compare", opts, default=opts[:min(4,len(opts))])
    if len(sel) < 2:
        st.warning("Select ≥2 sessions.")
        st.stop()

    sel_s = [st.session_state.sessions[opts.index(o)] for o in sel]
    rows  = []
    for s in sel_s:
        for r in s["model_results"]:
            row = {"Session":s["name"],"Dataset":s["dataset"],"Task":s["task"],
                   "Model":r["name"],"Runtime(s)":r["runtime"],"Memory(MB)":r["memory_mb"]}
            if s["task"]=="Regression":
                row.update({"R²":r.get("r2",0),"RMSE":r.get("rmse",0),"MAE":r.get("mae",0)})
            else:
                row.update({"Accuracy":r.get("accuracy",0),"F1":r.get("f1",0),
                             "Precision":r.get("precision",0),"Recall":r.get("recall",0)})
            rows.append(row)

    df_cmp = pd.DataFrame(rows)
    df_cmp["Label"] = df_cmp["Session"] + " / " + df_cmp["Model"]

    import plotly.graph_objects as go
    import plotly.express as px
    from frontend.components import theme as _theme, PALETTE

    reg_rows = df_cmp[df_cmp["Task"]=="Regression"]
    clf_rows = df_cmp[df_cmp["Task"]=="Classification"]

    if not reg_rows.empty:
        st.markdown(sec_hdr("📊","Regression — R² Across Sessions"), unsafe_allow_html=True)
        fig = px.bar(reg_rows,x="Label",y="R²",color="Session",
                      text=reg_rows["R²"].apply(lambda v:f"{v:.4f}"),
                      color_discrete_sequence=PALETTE, title="R² — all sessions")
        fig.update_traces(textposition="outside")
        fig.update_layout(**_theme(440))
        st.plotly_chart(fig, use_container_width=True)

    if not clf_rows.empty:
        st.markdown(sec_hdr("📊","Classification — Accuracy Across Sessions"), unsafe_allow_html=True)
        fig = px.bar(clf_rows,x="Label",y="Accuracy",color="Session",
                      text=clf_rows["Accuracy"].apply(lambda v:f"{v:.4f}"),
                      color_discrete_sequence=PALETTE, title="Accuracy — all sessions")
        fig.update_traces(textposition="outside")
        fig.update_layout(**_theme(440))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(sec_hdr("⚡","Runtime Across All Sessions"), unsafe_allow_html=True)
    fig_rt = px.bar(df_cmp,x="Label",y="Runtime(s)",color="Session",
                     text=df_cmp["Runtime(s)"].apply(lambda v:f"{v:.3f}s"),
                     color_discrete_sequence=PALETTE, title="Training Runtime")
    fig_rt.update_traces(textposition="outside")
    fig_rt.update_layout(**_theme(440))
    st.plotly_chart(fig_rt, use_container_width=True)

    st.markdown(sec_hdr("🌡️","Performance Heatmap"), unsafe_allow_html=True)
    num_cols = [c for c in ["R²","Accuracy","F1","RMSE","MAE","Runtime(s)","Memory(MB)"]
                if c in df_cmp.columns]
    heat = df_cmp.set_index("Label")[num_cols].fillna(0)
    fig_heat = px.imshow(heat.T,text_auto=".3f",color_continuous_scale="Viridis",
                          aspect="auto",title="Sessions × Metrics Heatmap")
    fig_heat.update_layout(**{k:v for k,v in _theme(max(300,len(heat)*30)).items()
                               if k not in ["xaxis","yaxis"]})
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(sec_hdr("📋","Raw Comparison Table"), unsafe_allow_html=True)
    st.dataframe(df_cmp, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📁 History":
    st.markdown("# 📁 Session History")
    if not st.session_state.sessions:
        st.info("No sessions yet.")
        st.stop()

    for s in reversed(st.session_state.sessions):
        results = s["model_results"]
        task    = s["task"]
        tcls    = "bg" if task=="Regression" else "bb"
        mk      = "r2" if task=="Regression" else "accuracy"
        ml      = "R²"  if task=="Regression" else "Acc"

        with st.expander(f"📌 {s['name']}  ·  {s['dataset']}  ·  {s['timestamp'][:19]}"):
            _ns = f"{s['n_samples']:,} rows"
            _nf = f"{s['n_features']} features"
            _tg = f"Target: {s['target_col']}"
            _nm = f"{len(results)} models"
            st.markdown(
                f"{badge(task,tcls)}&nbsp;"
                f"{badge(_ns,'bo')}&nbsp;"
                f"{badge(_nf,'bb')}&nbsp;"
                f"{badge(_tg,'bp')}&nbsp;"
                f"{badge(_nm,'bg')}",
                unsafe_allow_html=True)
            st.markdown("")
            cols = st.columns(min(len(results),6))
            for col,r in zip(cols,results):
                col.markdown(kpi(r["name"],f"{r.get(mk,0):.4f}",ml),
                             unsafe_allow_html=True)
            if s.get("prep_log"):
                st.caption("Preprocessing: " + " · ".join(s["prep_log"][:3]))
            st.markdown("")
            ca,cb = st.columns([3,1])
            with ca:
                if st.button("📊 Set as active session", key=f"act_{s['id']}"):
                    st.session_state.active_session = s
                    st.success("Active session updated. Go to Analysis.")
            with cb:
                if st.button("🗑️ Delete", key=f"del_{s['id']}"):
                    st.session_state.sessions = [x for x in st.session_state.sessions
                                                  if x["id"]!=s["id"]]
                    delete_session(s["id"])
                    st.rerun()
