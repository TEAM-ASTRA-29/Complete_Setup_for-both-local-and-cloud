"""
backend/report.py
=================
PDF report generator using ReportLab.
"""

import io, os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak, Image as RLImage,
                                 HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

PALETTE = ["#58a6ff","#3fb950","#f78166","#bc8cff","#e3b341",
           "#79c0ff","#56d364","#ffa198","#d2a8ff","#f0883e"]


def _color(i): return PALETTE[i % len(PALETTE)]

def _theme():
    return dict(plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                font=dict(color="#c9d1d9", family="Inter"),
                xaxis=dict(gridcolor="#21262d", linecolor="#21262d",
                           tickfont=dict(color="#7d8590")),
                yaxis=dict(gridcolor="#21262d", linecolor="#21262d",
                           tickfont=dict(color="#7d8590")),
                margin=dict(l=50,r=20,t=50,b=50))

def _fig_to_img(fig, Wpt, h=380):
    try:
        fig.update_layout(**_theme())
        png = fig.to_image(format="png", width=700, height=h, scale=2)
        return RLImage(io.BytesIO(png), width=Wpt, height=Wpt*h/700)
    except Exception as e:
        from reportlab.lib.styles import getSampleStyleSheet
        ss = getSampleStyleSheet()
        return Paragraph(f"Chart unavailable: {e}",
                         ParagraphStyle("e",parent=ss["Normal"],fontSize=8,
                                        textColor=colors.HexColor("#7d8590")))


def generate_pdf(session: dict) -> bytes:
    buf  = io.BytesIO()
    Wpt  = A4[0] - 4*cm
    doc  = SimpleDocTemplate(buf, pagesize=A4,
                              leftMargin=2*cm, rightMargin=2*cm,
                              topMargin=2*cm,  bottomMargin=2*cm)
    ss   = getSampleStyleSheet()
    S_T  = ParagraphStyle("T", parent=ss["Title"], fontSize=20,
                          textColor=colors.HexColor("#58a6ff"),
                          spaceAfter=4, leading=26)
    S_H1 = ParagraphStyle("H1",parent=ss["Heading1"],fontSize=13,
                          textColor=colors.HexColor("#e6edf3"),
                          spaceBefore=12, spaceAfter=4)
    S_H2 = ParagraphStyle("H2",parent=ss["Heading2"],fontSize=10,
                          textColor=colors.HexColor("#c9d1d9"),
                          spaceBefore=6, spaceAfter=3)
    S_B  = ParagraphStyle("B", parent=ss["Normal"], fontSize=9,
                          textColor=colors.HexColor("#8b949e"),
                          leading=14, spaceAfter=4)
    S_M  = ParagraphStyle("M", parent=ss["Normal"], fontSize=8,
                          fontName="Courier",
                          textColor=colors.HexColor("#7d8590"), leading=12)
    S_FT = ParagraphStyle("FT",parent=ss["Normal"],fontSize=8,
                          textColor=colors.HexColor("#7d8590"),
                          alignment=TA_CENTER)

    def hr(): return HRFlowable(width="100%",thickness=0.4,
                                color=colors.HexColor("#30363d"),
                                spaceAfter=6,spaceBefore=6)

    def tbl_style():
        return TableStyle([
            ("BACKGROUND",    (0,0),(-1,0), colors.HexColor("#1c2128")),
            ("TEXTCOLOR",     (0,0),(-1,0), colors.HexColor("#58a6ff")),
            ("FONTNAME",      (0,0),(-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0),(-1,0), 8),
            ("BOTTOMPADDING", (0,0),(-1,0), 6),
            ("TOPPADDING",    (0,0),(-1,0), 6),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),
             [colors.HexColor("#0d1117"), colors.HexColor("#161b22")]),
            ("TEXTCOLOR",     (0,1),(-1,-1), colors.HexColor("#c9d1d9")),
            ("FONTSIZE",      (0,1),(-1,-1), 8),
            ("GRID",          (0,0),(-1,-1), 0.3, colors.HexColor("#30363d")),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("RIGHTPADDING",  (0,0),(-1,-1), 6),
            ("TOPPADDING",    (0,1),(-1,-1), 4),
            ("BOTTOMPADDING", (0,1),(-1,-1), 4),
        ])

    story = []
    task    = session["task"]
    results = session["model_results"]
    feats   = session["feature_names"]

    # ── Cover page ──────────────────────────────────────────────
    story += [Spacer(1,.5*cm),
              Paragraph("CropML Research Studio", S_T),
              Paragraph("Comprehensive ML Model Performance Report", S_B),
              Spacer(1,.2*cm), hr()]

    meta = [
        ["Session",  session["name"]],
        ["Dataset",  session["dataset"]],
        ["Task",     task],
        ["Samples",  f"{session['n_samples']:,}"],
        ["Features", str(session["n_features"])],
        ["Target",   session["target_col"]],
        ["Test size",f"{int(session['test_size']*100)}%"],
        ["Models trained", str(len(results))],
        ["Generated",datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    mt = Table([[Paragraph(r[0],S_M), Paragraph(str(r[1]),S_B)]
                for r in meta], colWidths=[3.5*cm, Wpt-3.5*cm])
    mt.setStyle(tbl_style())
    story += [mt, PageBreak()]

    # ── Performance summary table ────────────────────────────────
    story += [Paragraph("Model Performance Summary", S_H1), hr()]

    if task == "Regression":
        hdr  = ["Model","R²","RMSE","MAE","MSE","Runtime(s)","Mem(MB)","Size(MB)"]
        rows = [hdr] + [
            [r["name"],
             f"{r.get('r2',0):.4f}", f"{r.get('rmse',0):.4f}",
             f"{r.get('mae',0):.4f}", f"{r.get('mse',0):.4f}",
             f"{r['runtime']:.3f}", f"{r['memory_mb']:.1f}",
             f"{r['model_size_mb']:.3f}"]
            for r in sorted(results, key=lambda x: -x.get("r2",0))
        ]
    else:
        hdr  = ["Model","Accuracy","F1","Precision","Recall","AUC","Runtime(s)","Size(MB)"]
        rows = [hdr] + [
            [r["name"],
             f"{r.get('accuracy',0):.4f}", f"{r.get('f1',0):.4f}",
             f"{r.get('precision',0):.4f}", f"{r.get('recall',0):.4f}",
             f"{r.get('auc',0) or 0:.4f}",
             f"{r['runtime']:.3f}", f"{r['model_size_mb']:.3f}"]
            for r in sorted(results, key=lambda x: -x.get("accuracy",0))
        ]

    cw = Wpt / len(hdr)
    pt = Table(rows, colWidths=[cw]*len(hdr))
    pt.setStyle(tbl_style())
    story += [pt, Spacer(1,.5*cm)]

    # ── Metric explanation ──
    story.append(Paragraph("Metric Definitions", S_H2))
    if task == "Regression":
        defs = [
            ("R²", "Coefficient of determination. Range [0,1]. Higher = better. Explains variance captured."),
            ("RMSE", "Root Mean Squared Error. Same unit as target. Lower = better. Penalises large errors."),
            ("MAE", "Mean Absolute Error. Same unit as target. Lower = better. Average absolute prediction error."),
            ("MSE", "Mean Squared Error. Squared units. Lower = better."),
        ]
    else:
        defs = [
            ("Accuracy", "Fraction of correct predictions. Range [0,1]. Higher = better."),
            ("F1", "Harmonic mean of Precision and Recall. Better for imbalanced classes."),
            ("Precision", "Of all predicted positives, how many are actually positive."),
            ("Recall", "Of all actual positives, how many were correctly predicted."),
            ("AUC", "Area Under ROC Curve. Measures discriminative ability. Range [0,1]. Higher = better."),
        ]
    for term, defn in defs:
        story.append(Paragraph(f"<b>{term}:</b> {defn}", S_B))
    story.append(Spacer(1,.3*cm))

    # ── Hyperparameters ──────────────────────────────────────────
    story += [Paragraph("Hyperparameter Configuration", S_H1), hr()]
    for r in results:
        story.append(Paragraph(r["name"], S_H2))
        if r["params"]:
            ph = [["Parameter","Value"]] + [[k,v] for k,v in r["params"].items()]
            ptbl = Table(ph, colWidths=[5*cm, Wpt-5*cm])
            ptbl.setStyle(tbl_style())
            story.append(ptbl)
        else:
            story.append(Paragraph("No tunable hyperparameters.", S_B))
        story.append(Spacer(1,.2*cm))
    story.append(PageBreak())

    # ── Charts ───────────────────────────────────────────────────
    story += [Paragraph("Visual Analysis", S_H1), hr()]

    mk = "r2" if task == "Regression" else "accuracy"
    ml = "R²"  if task == "Regression" else "Accuracy"

    fig_bar = go.Figure([go.Bar(
        x=[r["name"] for r in results],
        y=[r.get(mk,0) for r in results],
        marker=dict(color=[_color(i) for i in range(len(results))],line=dict(width=0)),
        text=[f"{r.get(mk,0):.4f}" for r in results], textposition="outside")])
    fig_bar.update_layout(title=f"{ml} by Model", height=380)
    story += [Paragraph(f"{ml} Comparison", S_H2),
              _fig_to_img(fig_bar, Wpt, 380), Spacer(1,.3*cm)]

    fig_rt = go.Figure([go.Bar(
        x=[r["name"] for r in results],
        y=[r["runtime"] for r in results],
        marker=dict(color=[_color(i) for i in range(len(results))],line=dict(width=0)),
        text=[f"{r['runtime']:.3f}s" for r in results], textposition="outside")])
    fig_rt.update_layout(title="Training Runtime (s)", height=340)
    story += [Paragraph("Runtime Comparison", S_H2),
              _fig_to_img(fig_rt, Wpt, 340), PageBreak()]

    # ── Feature importance ───────────────────────────────────────
    fi_res = [r for r in results if r.get("feature_importances")]
    if fi_res:
        story += [Paragraph("Feature Importance", S_H1), hr()]
        for r in fi_res[:4]:
            df_fi = pd.DataFrame({"Feature": feats,
                                   "Importance": r["feature_importances"]})
            df_fi = df_fi.nlargest(15,"Importance").sort_values("Importance")
            fig_fi = px.bar(df_fi,x="Importance",y="Feature",orientation="h",
                            color_discrete_sequence=[_color(results.index(r))],
                            title=r["name"])
            h_fi = max(260, len(df_fi)*22)
            fig_fi.update_layout(height=h_fi)
            story += [Paragraph(r["name"], S_H2),
                      _fig_to_img(fig_fi, Wpt, h_fi), Spacer(1,.25*cm)]

    # ── Cloud vs Local ───────────────────────────────────────────
    cloud_data = session.get("cloud_comparison")
    if cloud_data:
        story += [PageBreak(), Paragraph("Cloud vs Local Deployment", S_H1), hr()]
        model_name = cloud_data.get("model_name","")
        story.append(Paragraph(f"Model: {model_name}", S_H2))
        local  = cloud_data["local"]
        cloud  = cloud_data["cloud"]
        cv_rows = [
            ["Metric","Local","Cloud"],
            ["Instance", f"{local['vcpu']} vCPU / {local['ram_gb']}GB RAM",
             f"{cloud['instance']} ({cloud['vcpu']} vCPU / {cloud['ram_gb']}GB RAM)"],
            ["Training Runtime", f"{local['runtime_s']:.4f}s", f"{cloud['runtime_s']:.4f}s"],
            ["Speedup", "1.0×", f"{cloud['speedup']}×"],
            ["Memory Used", f"{local['memory_mb']:.1f} MB", f"{cloud['memory_mb']:.1f} MB"],
            ["Cost per Run", "$0.00", f"${cloud['cost_cents']/100:.6f}"],
        ]
        cvt = Table(cv_rows, colWidths=[Wpt/3]*3)
        cvt.setStyle(tbl_style())
        story.append(cvt)

    # ── Footer ───────────────────────────────────────────────────
    story += [Spacer(1,.8*cm), hr(),
              Paragraph(f"CropML Research Studio v3.0 · "
                        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                        S_FT)]
    doc.build(story)
    return buf.getvalue()
