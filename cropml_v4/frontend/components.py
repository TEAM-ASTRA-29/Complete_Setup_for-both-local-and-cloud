"""
frontend/components.py
=======================
Reusable HTML/CSS components and Plotly chart builders.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PALETTE = ["#58a6ff","#3fb950","#f78166","#bc8cff","#e3b341",
           "#79c0ff","#56d364","#ffa198","#d2a8ff","#f0883e",
           "#a5d6ff","#26a641","#ff9492","#c5a1f5","#d29922"]

RADAR_FILL = ["rgba(88,166,255,.18)","rgba(63,185,80,.18)","rgba(247,129,102,.18)",
              "rgba(188,140,255,.18)","rgba(227,179,65,.18)","rgba(121,192,255,.18)",
              "rgba(86,211,100,.18)","rgba(255,161,152,.18)","rgba(194,161,245,.18)",
              "rgba(240,136,62,.18)"]

_BG   = "rgba(0,0,0,0)"
_GRID = "#21262d"
_TICK = "#7d8590"
_FONT = dict(family="Inter", color="#e6edf3")


def color(i): return PALETTE[i % len(PALETTE)]


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
:root{
  --bg:#0d1117;--surface:#161b22;--s2:#1c2128;
  --bd:#30363d;--bd2:#21262d;
  --blue:#58a6ff;--green:#3fb950;--org:#f78166;
  --purple:#bc8cff;--yellow:#e3b341;
  --text:#e6edf3;--muted:#7d8590;--r:10px;
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:var(--text);background:var(--bg);}
.block-container{padding:1.6rem 2.2rem 3rem;max-width:1400px;}
h1,h2,h3{font-weight:700;letter-spacing:-.02em;}
/* KPI */
.kc{background:var(--surface);border:1px solid var(--bd2);border-radius:var(--r);
    padding:.9rem 1.1rem;position:relative;overflow:hidden;}
.kc::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;background:var(--blue);}
.kc.g::after{background:var(--green);}
.kc.o::after{background:var(--org);}
.kc.p::after{background:var(--purple);}
.kc.y::after{background:var(--yellow);}
.kl{font-size:.66rem;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
    color:var(--muted);margin-bottom:.3rem;}
.kv{font-family:'JetBrains Mono',monospace;font-size:1.45rem;font-weight:500;
    color:var(--text);line-height:1.1;}
.ks{font-size:.7rem;color:var(--muted);margin-top:.2rem;}
/* Section header */
.sh{font-size:1rem;font-weight:600;color:var(--text);padding:.5rem 0;
    margin:1.6rem 0 .8rem;border-bottom:1px solid var(--bd2);display:flex;
    align-items:center;gap:.4rem;}
/* Badges */
.b{display:inline-block;border-radius:4px;padding:.12rem .5rem;
   font-family:'JetBrains Mono',monospace;font-size:.68rem;font-weight:500;}
.bb{background:rgba(88,166,255,.1);color:var(--blue);border:1px solid rgba(88,166,255,.25);}
.bg{background:rgba(63,185,80,.1);color:var(--green);border:1px solid rgba(63,185,80,.25);}
.bo{background:rgba(247,129,102,.1);color:var(--org);border:1px solid rgba(247,129,102,.25);}
.bp{background:rgba(188,140,255,.1);color:var(--purple);border:1px solid rgba(188,140,255,.25);}
/* Info boxes */
.ib{background:var(--s2);border:1px solid var(--bd);border-left:3px solid var(--blue);
    border-radius:var(--r);padding:.8rem 1rem;font-size:.84rem;line-height:1.6;}
.ib.g{border-left-color:var(--green);}
.ib.o{border-left-color:var(--org);}
.ib.p{border-left-color:var(--purple);}
/* Download */
.dlbtn{display:inline-block;background:var(--blue);color:#0d1117!important;
       font-weight:600;font-size:.84rem;padding:.55rem 1.3rem;border-radius:6px;
       text-decoration:none;}
/* Recommendation card */
.rec-card{background:var(--s2);border:1px solid var(--bd);border-radius:var(--r);
          padding:1rem 1.2rem;margin-bottom:.75rem;}
.rec-rank{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:var(--muted);}
.rec-name{font-size:1rem;font-weight:600;color:var(--text);margin:.2rem 0;}
.rec-score{font-family:'JetBrains Mono',monospace;color:var(--blue);font-size:.85rem;}
/* Preprocessing toggle */
.prep-box{background:var(--s2);border:1px solid var(--bd);border-radius:var(--r);padding:1rem;}
/* Sidebar */
section[data-testid="stSidebar"]{background:var(--surface)!important;
    border-right:1px solid var(--bd2);}
/* Metric comparison table */
.metric-table{width:100%;border-collapse:collapse;font-size:.85rem;}
.metric-table th{background:var(--s2);color:var(--blue);font-weight:600;
    padding:.5rem .8rem;border-bottom:1px solid var(--bd);text-align:left;}
.metric-table td{padding:.45rem .8rem;border-bottom:1px solid var(--bd2);color:var(--text);}
.metric-table tr:nth-child(even) td{background:rgba(255,255,255,.02);}
.metric-table .best{color:var(--green);font-weight:600;}
.metric-table .worst{color:var(--org);}
</style>
"""

def kpi(label, value, sub="", cls=""):
    return (f"<div class='kc {cls}'><div class='kl'>{label}</div>"
            f"<div class='kv'>{value}</div>"
            + (f"<div class='ks'>{sub}</div>" if sub else "") + "</div>")

def badge(text, cls="bb"):
    return f"<span class='b {cls}'>{text}</span>"

def info_box(content, cls=""):
    return f"<div class='ib {cls}'>{content}</div>"

def sec_hdr(icon, text):
    return f"<div class='sh'>{icon} {text}</div>"

def norm_inv(val, best, worst):
    if abs(best - worst) < 1e-9: return 0.5
    return float(max(0.0, min(1.0, (worst-val)/(worst-best))))


# ─────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────
def theme(height=400, legend=True):
    base = dict(
        height=height, plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FONT,
        xaxis=dict(gridcolor=_GRID,linecolor=_GRID,tickfont=dict(color=_TICK),
                   zerolinecolor=_GRID),
        yaxis=dict(gridcolor=_GRID,linecolor=_GRID,tickfont=dict(color=_TICK),
                   zerolinecolor=_GRID),
        margin=dict(l=50,r=20,t=50,b=50),
        hoverlabel=dict(bgcolor="#1c2128",font=dict(color="#e6edf3"),
                        bordercolor=_GRID),
    )
    if legend:
        base["legend"] = dict(font=dict(color="#c9d1d9"), bgcolor=_BG,
                              bordercolor=_GRID, borderwidth=1)
    return base


# ─────────────────────────────────────────────────────────────
# METRIC COMPARISON TABLE (HTML)
# ─────────────────────────────────────────────────────────────
def metric_comparison_table(results: list, task: str) -> str:
    """Build a styled HTML metric comparison table with best/worst highlights."""
    if task == "Regression":
        metrics = [
            ("R²",         "r2",          True,  "Variance explained (higher=better)"),
            ("RMSE",       "rmse",        False, "Root mean squared error (lower=better)"),
            ("MAE",        "mae",         False, "Mean absolute error (lower=better)"),
            ("MSE",        "mse",         False, "Mean squared error (lower=better)"),
            ("Runtime (s)","runtime",     False, "Training time in seconds (lower=better)"),
            ("Memory (MB)","memory_mb",   False, "RAM consumed during training (lower=better)"),
            ("Size (MB)",  "model_size_mb",False,"Serialised model file size (lower=better)"),
        ]
    else:
        metrics = [
            ("Accuracy",   "accuracy",    True,  "Overall correct predictions (higher=better)"),
            ("F1",         "f1",          True,  "Harmonic mean of precision/recall (higher=better)"),
            ("Precision",  "precision",   True,  "Positive predictive value (higher=better)"),
            ("Recall",     "recall",      True,  "Sensitivity / true positive rate (higher=better)"),
            ("AUC",        "auc",         True,  "Area under ROC curve (higher=better)"),
            ("Runtime (s)","runtime",     False, "Training time in seconds (lower=better)"),
            ("Size (MB)",  "model_size_mb",False,"Serialised model file size (lower=better)"),
        ]

    names = [r["name"] for r in results]
    header = "<tr><th>Metric</th><th>Description</th>" + \
             "".join(f"<th>{n}</th>" for n in names) + "</tr>"

    rows_html = ""
    for label, key, higher_better, desc in metrics:
        vals = []
        for r in results:
            v = r.get(key)
            if v is None: v = 0.0
            vals.append(float(v))

        if not any(v != 0 for v in vals):
            continue

        best_val  = max(vals) if higher_better else min(vals)
        worst_val = min(vals) if higher_better else max(vals)

        cells = ""
        for v in vals:
            cls = ""
            if v == best_val:  cls = "best"
            elif v == worst_val: cls = "worst"
            cells += f"<td class='{cls}'>{v:.4f}</td>"

        rows_html += f"<tr><td><b>{label}</b></td><td style='color:#7d8590;font-size:.78rem;'>{desc}</td>{cells}</tr>"

    return f"<table class='metric-table'><thead>{header}</thead><tbody>{rows_html}</tbody></table>"


# ─────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────
def chart_primary_metric(results, task):
    mk = "r2" if task=="Regression" else "accuracy"
    ml = "R²"  if task=="Regression" else "Accuracy"
    df = pd.DataFrame([{"Model":r["name"], ml:r.get(mk,0)} for r in results])
    df = df.sort_values(ml, ascending=False)
    fig = go.Figure([go.Bar(
        x=df["Model"], y=df[ml],
        marker=dict(color=[color(i) for i in range(len(df))], line=dict(width=0)),
        text=[f"{v:.4f}" for v in df[ml]], textposition="outside",
        hovertemplate=f"<b>%{{x}}</b><br>{ml} = %{{y:.4f}}<extra></extra>",
    )])
    yr = ([min(0, df[ml].min()-.05), min(1.12, df[ml].max()+.08)]
          if task=="Regression" else [0, min(1.12, df[ml].max()+.08)])
    fig.update_layout(title=f"{ml} Score Comparison — higher is better",
                       yaxis_range=yr, **theme(420))
    return fig


def chart_error_metrics(results, task):
    if task == "Regression":
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["RMSE (lower=better)","MAE (lower=better)"])
        ax = dict(gridcolor=_GRID, linecolor=_GRID, tickfont=dict(color=_TICK))
        for i,r in enumerate(results):
            c = color(i)
            for col,met in [(1,"rmse"),(2,"mae")]:
                fig.add_trace(go.Bar(x=[r["name"]],y=[r.get(met,0)],marker_color=c,
                                      showlegend=False,text=[f"{r.get(met,0):.4f}"],
                                      textposition="outside"),row=1,col=col)
        fig.update_layout(height=400, plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FONT,
                           margin=dict(l=40,r=20,t=50,b=50),
                           xaxis=ax,yaxis=ax,xaxis2=ax,yaxis2=ax)
    else:
        fig = go.Figure()
        for met,col_c in [("F1","#58a6ff"),("Precision","#3fb950"),("Recall","#f78166")]:
            vals = [r.get(met.lower(),0) for r in results]
            fig.add_trace(go.Bar(name=met,x=[r["name"] for r in results],y=vals,
                                  marker_color=col_c,opacity=.85))
        fig.update_layout(barmode="group", title="F1 / Precision / Recall",
                           **theme(420))
    return fig


def chart_computational(results):
    fig = make_subplots(rows=1, cols=3,
                         subplot_titles=["Runtime (s)","Memory (MB)","Model Size (MB)"])
    ax = dict(gridcolor=_GRID, linecolor=_GRID, tickfont=dict(color=_TICK, size=8))
    for i,r in enumerate(results):
        c = color(i)
        for ci,key,fmt in [(1,"runtime",".3f"),(2,"memory_mb",".1f"),(3,"model_size_mb",".3f")]:
            fig.add_trace(go.Bar(x=[r["name"]],y=[r[key]],marker_color=c,showlegend=False,
                                  text=[f"{r[key]:{fmt}}"],textposition="outside"),row=1,col=ci)
    fig.update_layout(height=400, plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FONT,
                       margin=dict(l=40,r=20,t=50,b=50),
                       xaxis=ax,yaxis=ax,xaxis2=ax,yaxis2=ax,xaxis3=ax,yaxis3=ax)
    return fig


def chart_actual_vs_predicted(y_test, y_pred, model_name):
    dmin = float(min(y_test.min(), y_pred.min()))
    dmax = float(max(y_test.max(), y_pred.max()))
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                              marker=dict(color="#58a6ff",size=5,opacity=.55,line=dict(width=0)),
                              hovertemplate="Actual:%{x:.3f}<br>Predicted:%{y:.3f}<extra></extra>",
                              name="Predictions"))
    fig.add_trace(go.Scatter(x=[dmin,dmax],y=[dmin,dmax],mode="lines",
                              line=dict(color="#ef4444",width=1.5,dash="dash"),
                              name="Perfect fit"))
    fig.update_layout(title=f"{model_name} — Actual vs Predicted",
                       xaxis_title="Actual",yaxis_title="Predicted",**theme(420))
    return fig


def chart_residuals(y_test, y_pred, model_name):
    residuals = y_test - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                              marker=dict(color="#58a6ff",size=5,opacity=.55,line=dict(width=0)),
                              hovertemplate="Predicted:%{x:.3f}<br>Residual:%{y:.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color="#ef4444", line_dash="dash", line_width=1.5)
    fig.update_layout(title=f"{model_name} — Residuals",
                       xaxis_title="Predicted",yaxis_title="Residual",**theme(380))
    return fig


def chart_residual_hist(results, y_test):
    fig = go.Figure()
    for i,r in enumerate(results):
        if len(y_test) == len(r["y_pred"]):
            fig.add_trace(go.Histogram(x=y_test-np.array(r["y_pred"]),
                                        name=r["name"],opacity=.6,nbinsx=40,
                                        marker_color=color(i)))
    fig.update_layout(barmode="overlay",title="Residual Distribution — All Models",
                       xaxis_title="Residual",**theme(380))
    return fig


def chart_feature_importance(feats, importances, model_name, model_idx):
    df_fi = pd.DataFrame({"Feature":feats,"Importance":importances})
    df_fi = df_fi.nlargest(20,"Importance").sort_values("Importance")
    fig = px.bar(df_fi,x="Importance",y="Feature",orientation="h",
                 color="Importance",
                 color_continuous_scale=[[0,"#1c2128"],[1,color(model_idx)]],
                 title=f"{model_name} — Feature Importance (Top 20)")
    fig.update_coloraxes(showscale=False)
    fig.update_layout(**theme(max(300,len(df_fi)*26)))
    return fig


def chart_radar(results, task):
    if task == "Regression":
        dims = ["R²","Low RMSE","Low MAE","Speed","Mem Eff","Size Eff"]
        def sc(r):
            return [max(0.,min(1.,r.get("r2",0))),
                    norm_inv(r.get("rmse",0),min(x.get("rmse",0) for x in results),max(x.get("rmse",0) for x in results)),
                    norm_inv(r.get("mae",0), min(x.get("mae",0)  for x in results),max(x.get("mae",0)  for x in results)),
                    norm_inv(r["runtime"],   min(x["runtime"]     for x in results),max(x["runtime"]     for x in results)),
                    norm_inv(r["memory_mb"], min(x["memory_mb"]   for x in results),max(x["memory_mb"]   for x in results)+.001),
                    norm_inv(r["model_size_mb"],min(x["model_size_mb"] for x in results),max(x["model_size_mb"] for x in results)+.001)]
    else:
        dims = ["Accuracy","F1","Precision","Recall","Speed","Size Eff"]
        def sc(r):
            return [r.get("accuracy",0),r.get("f1",0),r.get("precision",0),r.get("recall",0),
                    norm_inv(r["runtime"],       min(x["runtime"]       for x in results),max(x["runtime"]       for x in results)),
                    norm_inv(r["model_size_mb"], min(x["model_size_mb"] for x in results),max(x["model_size_mb"] for x in results)+.001)]

    fig = go.Figure()
    for i,r in enumerate(results):
        s = sc(r)
        fig.add_trace(go.Scatterpolar(
            r=s+[s[0]], theta=dims+[dims[0]], fill="toself", name=r["name"],
            line=dict(color=color(i),width=2),
            fillcolor=RADAR_FILL[i%len(RADAR_FILL)]))
    fig.update_layout(
        polar=dict(bgcolor=_BG,
                   radialaxis=dict(visible=True,range=[0,1],gridcolor="#21262d",
                                   tickfont=dict(color="#7d8590",size=9),
                                   tickvals=[.25,.5,.75,1.]),
                   angularaxis=dict(gridcolor="#21262d",tickfont=dict(color="#c9d1d9",size=11))),
        height=500, plot_bgcolor=_BG, paper_bgcolor=_BG, font=_FONT,
        legend=dict(font=dict(color="#c9d1d9"),bgcolor=_BG,bordercolor=_GRID,borderwidth=1),
        margin=dict(l=80,r=80,t=60,b=60))
    return fig


def chart_confusion_matrix(conf_matrix, model_name):
    arr = np.array(conf_matrix)
    fig = px.imshow(arr, text_auto=True, color_continuous_scale="Blues",
                    labels=dict(x="Predicted",y="Actual"),
                    title=f"{model_name} — Confusion Matrix")
    fig.update_layout(**{k:v for k,v in theme(380).items()
                          if k not in ["xaxis","yaxis"]})
    return fig


def chart_cloud_vs_local(cloud_data: dict):
    local = cloud_data["local"]
    cloud = cloud_data["cloud"]
    cats  = ["Runtime (s)","Memory (MB)"]
    fig   = go.Figure()
    fig.add_trace(go.Bar(name="Local",x=cats,
                          y=[local["runtime_s"],local["memory_mb"]],
                          marker_color="#58a6ff",opacity=.85,
                          text=[f"{local['runtime_s']:.4f}s",f"{local['memory_mb']:.1f}MB"],
                          textposition="outside"))
    fig.add_trace(go.Bar(name=f"Cloud ({cloud['instance']})",x=cats,
                          y=[cloud["runtime_s"],cloud["memory_mb"]],
                          marker_color="#3fb950",opacity=.85,
                          text=[f"{cloud['runtime_s']:.4f}s",f"{cloud['memory_mb']:.1f}MB"],
                          textposition="outside"))
    fig.update_layout(barmode="group",title="Local vs Cloud Runtime & Memory",**theme(380))
    return fig
