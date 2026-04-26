"""
app.py  –  MediRisk AI
======================
Mobile-first Streamlit health risk prediction app.

Run:
    streamlit run app.py

To expose on your local network (open on phone via Wi-Fi):
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
Then visit  http://<your-PC-IP>:8501  on your phone browser.
"""

import os, math, io, csv, warnings, datetime, json
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── Optional PDF ──────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph,
                                    Spacer, Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors as rlc
    from reportlab.lib.units import cm
    PDF_OK = True
except ImportError:
    PDF_OK = False

from config import MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS

# ══════════════════════════════════════════════════════════════
# 0.  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MediRisk AI",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════
# 1.  GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #0b1120 !important;
    color: #f1f5f9 !important;
}

/* ── Phone frame ─────────────────────────────────────────── */
.main > .block-container {
    max-width: 430px !important;
    margin: 0 auto !important;
    padding: 0 14px 80px !important;
    background: #0b1120 !important;
}
section[data-testid="stSidebar"] { display: none !important; }
header[data-testid="stHeader"]   { background: transparent !important; }
footer { display: none !important; }

h1, h2, h3, h4 { color: #f1f5f9 !important; }

/* ── App header ──────────────────────────────────────────── */
.app-header {
    background: linear-gradient(140deg, #0369a1 0%, #4f46e5 60%, #7c3aed 100%);
    border-radius: 24px;
    padding: 22px 18px 18px;
    text-align: center;
    margin: 12px 0 16px;
}
.app-header h1 {
    color: #fff !important;
    font-size: 22px !important;
    font-weight: 900 !important;
    margin: 0 0 4px !important;
    letter-spacing: -.3px;
}
.app-header .sub { color: rgba(255,255,255,.75); font-size: 12px; font-weight: 500; }
.stat-row { display:flex; justify-content:center; gap:10px; margin-top:12px; }
.stat-pill {
    background: rgba(255,255,255,.15);
    border-radius: 30px;
    padding: 4px 11px;
    font-size: 10px;
    font-weight: 600;
    color: #fff;
}

/* ── Section labels ──────────────────────────────────────── */
.sec-label {
    color: #38bdf8;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin: 18px 0 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e3a5f;
}

/* ── Widget overrides ────────────────────────────────────── */
div[data-testid="stSlider"] > div > div > div { background: #38bdf8 !important; }
div[data-testid="stSlider"] > label,
div[data-testid="stRadio"] > label,
div[data-testid="stRadio"] [data-testid="stWidgetLabel"],
div[data-testid="stTextInput"] > label {
    color: #94a3b8 !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}
div[data-testid="stRadio"] label span { color: #cbd5e1 !important; font-size: 12px !important; }
div[data-testid="stTextInput"] input {
    background: #0b1120 !important;
    border: 1px solid #1e3a5f !important;
    color: #f1f5f9 !important;
    border-radius: 12px !important;
}

/* ── CTA button ──────────────────────────────────────────── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0284c7, #4f46e5) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 16px 0 !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    width: 100% !important;
    box-shadow: 0 6px 20px rgba(2,132,199,.35) !important;
    margin-top: 8px !important;
}
div[data-testid="stButton"] > button:hover {
    box-shadow: 0 8px 28px rgba(2,132,199,.55) !important;
}

/* ── Download buttons ────────────────────────────────────── */
div[data-testid="stDownloadButton"] > button {
    background: #111827 !important;
    color: #38bdf8 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 14px !important;
    padding: 10px 0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    width: 100% !important;
    margin-bottom: 6px !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: #1e293b !important;
    border-color: #38bdf8 !important;
}

/* ── Expander ────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 16px !important;
    overflow: hidden;
}
details[data-testid="stExpander"] summary {
    color: #94a3b8 !important;
    font-size: 13px !important;
    padding: 12px 16px !important;
}

/* ── Progress bar ────────────────────────────────────────── */
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #ef4444, #dc2626) !important;
    border-radius: 10px !important;
}
div[data-testid="stProgress"] > div {
    background: #1e3a5f !important;
    border-radius: 10px !important;
    height: 8px !important;
}

/* ── Tabs ────────────────────────────────────────────────── */
div[data-testid="stTabs"] [data-testid="stTab"] {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
}
div[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* ── Metric widget ───────────────────────────────────────── */
div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 12px 14px !important;
}
div[data-testid="stMetric"] label {
    color: #64748b !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: .8px !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 20px !important;
    font-weight: 800 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    font-size: 11px !important;
    font-weight: 600 !important;
}

img { border-radius: 12px; }
hr  { border-color: #1e3a5f !important; margin: 14px 0 !important; }

/* ── Custom HTML ─────────────────────────────────────────── */
.gauge-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 22px;
    padding: 18px 18px 14px;
    margin-bottom: 12px;
}
.gauge-flex  { display:flex; justify-content:space-between; align-items:flex-start; }
.risk-pct    { font-size:62px; font-weight:900; line-height:1; }
.risk-unit   { font-size:22px; color:#475569; font-weight:700; }
.risk-badge-risk   { display:inline-block; background:#fef2f2; color:#dc2626; border-radius:30px; padding:4px 12px; font-size:11px; font-weight:700; margin-top:6px; }
.risk-badge-norisk { display:inline-block; background:#f0fdf4; color:#16a34a; border-radius:30px; padding:4px 12px; font-size:11px; font-weight:700; margin-top:6px; }
.met-age-block { text-align:right; }
.met-age-val   { font-size:34px; font-weight:900; color:#f1f5f9; }
.met-age-lbl   { font-size:10px; color:#64748b; letter-spacing:.8px; text-transform:uppercase; }
.met-age-diff  { font-size:12px; font-weight:700; margin-top:2px; }

.pill-row  { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
.info-pill {
    background: #0b1120;
    border: 1px solid #1e3a5f;
    border-radius: 30px;
    padding: 5px 12px;
    font-size: 12px;
    color: #94a3b8;
}
.info-pill b { color: #f1f5f9; }

.rec-card {
    background: #0b1120;
    border-left: 3px solid #38bdf8;
    border-radius: 0 14px 14px 0;
    padding: 11px 14px;
    margin-bottom: 8px;
}
.rec-title { color:#38bdf8; font-weight:700; font-size:13px; }
.rec-body  { color:#94a3b8; font-size:12px; margin-top:3px; line-height:1.5; }

.detail-box {
    background: #0b1120;
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 14px 16px;
}
.detail-box p { margin:0 0 8px; font-size:13px; color:#cbd5e1; }
.detail-box p:last-child { margin:0; }

.footer-note {
    text-align:center;
    color:#1e3a5f;
    font-size:11px;
    margin-top:24px;
    padding-bottom:16px;
    line-height:1.6;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# 2.  MODEL LOADER
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


# ══════════════════════════════════════════════════════════════
# 3.  HELPERS
# ══════════════════════════════════════════════════════════════
def feature_vector(age, sex, bmi, fam, htn, smoke,
                   act, stress, steps, sleep,
                   chol, hdl, ldl, glucose, insulin, bp):
    raw = {
        "Age": age, "Sex": 1 if sex == "Male" else 0,
        "BMI": bmi,
        "Family_History":    1 if fam   == "Yes"    else 0,
        "Hypertension":      1 if htn   == "Yes"    else 0,
        "Smoking_Status":    1 if smoke == "Smoker" else 0,
        "Physical_Activity": act, "Stress_Level": stress,
        "Steps": steps, "Sleep_Quality": sleep,
        "Cholesterol_mg/dL": chol,
        "HDL_mg/dL": hdl, "LDL_mg/dL": ldl,
        "Glucose_Fasting_mg/dL": glucose,
        "Insulin_uIU/mL": insulin, "Systolic_BP": bp,
    }
    _, scaler = load_model()
    try:
        feats = scaler.feature_names_in_.tolist()
    except Exception:
        feats = FEATURE_COLUMNS
    return np.array([raw.get(f, 0) for f in feats]).reshape(1, -1)


def risk_color(p):
    return "#ef4444" if p >= 65 else ("#f59e0b" if p >= 40 else "#10b981")


# ══════════════════════════════════════════════════════════════
# 4.  CHARTS
# ══════════════════════════════════════════════════════════════
BG = "#111827"; DARK = "#0b1120"; GRID = "#1e3a5f"


def _to_img(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def make_radar(metrics: dict) -> Image.Image:
    cats   = list(metrics.keys())
    N      = len(cats)
    vals   = list(metrics.values()) + [list(metrics.values())[0]]
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(3.4, 3.4),
                           subplot_kw={"polar": True}, facecolor=BG)
    ax.set_facecolor(BG)
    for r in [2, 4, 6, 8]:
        ax.plot(angles, [r]*(N+1), color=GRID, linewidth=0.5, linestyle="--")
    for a in angles[:-1]:
        ax.plot([0, a], [0, 10], color=GRID, linewidth=0.5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=8, color="#94a3b8", fontweight="600")
    ax.tick_params(axis="x", pad=8)
    ax.set_yticks([]); ax.set_rlim(0, 10)
    ax.set_theta_offset(math.pi / 2); ax.set_theta_direction(-1)
    ax.spines["polar"].set_visible(False); ax.set_frame_on(False)

    ax.fill(angles, vals, color="#38bdf8", alpha=0.18)
    ax.plot(angles, vals, color="#38bdf8", linewidth=2.2)
    for a, v in zip(angles[:-1], vals[:-1]):
        ax.plot(a, v, "o", color="#38bdf8", markersize=4, zorder=5)

    fig.tight_layout(pad=0.4)
    return _to_img(fig)


def make_drivers(factors: list) -> Image.Image:
    labels, values, colors = zip(*factors)
    fig, ax = plt.subplots(figsize=(3.6, 2.6), facecolor=BG)
    ax.set_facecolor(BG)
    y    = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.55, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color="#f1f5f9", fontweight="600")
    ax.invert_yaxis()
    ax.set_xlabel("Contribution (%)", fontsize=8, color="#64748b")
    ax.tick_params(axis="x", colors="#64748b", labelsize=8)
    ax.set_xlim(0, max(values) * 1.38)
    ax.grid(axis="x", color=GRID, linestyle="--", linewidth=0.7, zorder=0)
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    for bar, v in zip(bars, values):
        ax.text(v + 0.4, bar.get_y() + bar.get_height()/2,
                f"+{v}%", va="center", color="#fbbf24", fontsize=8, fontweight="700")
    fig.tight_layout(pad=0.4)
    return _to_img(fig)


def make_gauge_arc(prob: float) -> Image.Image:
    fig, ax = plt.subplots(figsize=(3.8, 2.1), facecolor=BG)
    ax.set_facecolor(BG); ax.set_aspect("equal"); ax.axis("off")
    r = 1.0
    theta_bg = np.linspace(math.pi, 0, 300)
    ax.plot(r*np.cos(theta_bg), r*np.sin(theta_bg),
            color=GRID, linewidth=11, solid_capstyle="round")
    frac     = prob / 100
    theta_f  = np.linspace(math.pi, math.pi - frac*math.pi, 300)
    col      = risk_color(prob)
    ax.plot(r*np.cos(theta_f), r*np.sin(theta_f),
            color=col, linewidth=11, solid_capstyle="round")
    angle = math.pi - frac * math.pi
    ax.annotate("", xy=(0.7*math.cos(angle), 0.7*math.sin(angle)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2,
                                mutation_scale=14))
    ax.plot(0, 0, "o", color=col, markersize=7, zorder=10)
    ax.text(-1.08, -0.22, "0",    ha="center", color="#64748b", fontsize=8)
    ax.text(0,      1.14, "50%",  ha="center", color="#64748b", fontsize=8)
    ax.text(1.08,  -0.22, "100",  ha="center", color="#64748b", fontsize=8)
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-0.32, 1.28)
    fig.tight_layout(pad=0.2)
    return _to_img(fig)


# ══════════════════════════════════════════════════════════════
# 5.  EXPORTS
# ══════════════════════════════════════════════════════════════
def export_csv(name, prob, met_age, age_diff, recs, inputs: dict) -> bytes:
    buf = io.StringIO()
    w   = csv.writer(buf)
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    w.writerows([
        ["MediRisk AI – Health Report"], ["Generated", ts],
        ["Patient", name or "—"], [],
        ["RISK SCORE", f"{prob:.1f}%"],
        ["ASSESSMENT", "AT RISK" if prob >= 50 else "NOT AT RISK"],
        ["METABOLIC AGE", met_age],
        ["AGE DIFFERENCE", f"{'+' if age_diff>0 else ''}{age_diff} yrs"],
        [], ["CLINICAL INPUTS"],
    ])
    for k, v in inputs.items():
        w.writerow([k, v])
    w.writerows([[], ["RECOMMENDATIONS"]])
    for ic, ti, tx in recs:
        w.writerow([f"{ic} {ti}", tx])
    w.writerow([])
    w.writerow(["DISCLAIMER",
                "For clinical support only. Not a substitute for professional diagnosis."])
    return buf.getvalue().encode("utf-8")


def export_pdf(name, prob, met_age, age_diff, recs, inputs: dict) -> bytes:
    if not PDF_OK:
        return b""
    buf    = io.BytesIO()
    doc    = SimpleDocTemplate(buf, pagesize=A4,
                               leftMargin=2*cm, rightMargin=2*cm,
                               topMargin=2.5*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    ts     = datetime.datetime.now().strftime("%B %d, %Y  %H:%M")

    title_s = ParagraphStyle("ts", parent=styles["Title"],
                              fontSize=20, spaceAfter=4,
                              textColor=rlc.HexColor("#0369a1"))
    sub_s   = ParagraphStyle("ss", parent=styles["Normal"],
                              fontSize=11, textColor=rlc.HexColor("#64748b"),
                              spaceAfter=16)
    h2_s    = ParagraphStyle("h2", parent=styles["Heading2"],
                              fontSize=13, textColor=rlc.HexColor("#1e293b"),
                              spaceBefore=14, spaceAfter=6)
    body_s  = ParagraphStyle("bs", parent=styles["Normal"],
                              fontSize=11, leading=16, spaceAfter=6)
    disc_s  = ParagraphStyle("ds", parent=styles["Normal"],
                              fontSize=9,  textColor=rlc.HexColor("#94a3b8"))

    risk_hex = "dc2626" if prob >= 50 else "16a34a"
    story = [
        Paragraph("🏥 MediRisk AI – Health Report", title_s),
        Paragraph(f"Generated: {ts}", sub_s),
        HRFlowable(width="100%", thickness=1, color=rlc.HexColor("#e2e8f0")),
        Spacer(1, 12),
        Paragraph(f"Patient: <b>{name or '—'}</b>", body_s),
        Paragraph(
            f"Risk Score: <b><font color='#{risk_hex}'>"
            f"{prob:.1f}%  –  {'AT RISK' if prob>=50 else 'NOT AT RISK'}"
            f"</font></b>", body_s),
        Paragraph(
            f"Metabolic Age: <b>{met_age}</b>  "
            f"({'▲ ' if age_diff>0 else '▼ '}{abs(age_diff)} yrs vs chronological)",
            body_s),
        Spacer(1, 10),
        Paragraph("Clinical Inputs", h2_s),
    ]
    tdata = [["Metric", "Value"]] + [[k, str(v)] for k, v in inputs.items()]
    tbl   = Table(tdata, colWidths=[9*cm, 5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), rlc.HexColor("#0369a1")),
        ("TEXTCOLOR",   (0,0), (-1,0), rlc.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [rlc.HexColor("#f8fafc"), rlc.HexColor("#f1f5f9")]),
        ("GRID",        (0,0), (-1,-1), 0.5, rlc.HexColor("#cbd5e1")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story += [tbl, Spacer(1, 14),
              Paragraph("Personalised Recommendations", h2_s)]
    for ic, ti, tx in recs:
        story.append(Paragraph(f"<b>{ic} {ti}:</b> {tx}", body_s))
    story += [
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=0.5,
                   color=rlc.HexColor("#cbd5e1")),
        Spacer(1, 6),
        Paragraph(
            "MediRisk AI – Stacking Ensemble (RF + LightGBM + XGBoost + CatBoost) · "
            "94.8% Accuracy · For clinical support only. "
            "Not a substitute for professional medical diagnosis.", disc_s),
    ]
    doc.build(story)
    buf.seek(0)
    return buf.read()


# ══════════════════════════════════════════════════════════════
# 6.  MAIN APP
# ══════════════════════════════════════════════════════════════
def main():

    # ── Header ─────────────────────────────────────────────────
    st.markdown("""
    <div class="app-header">
        <h1>🏥 MediRisk AI</h1>
        <div class="sub">AI-Powered Patient Risk Assessment</div>
        <div class="stat-row">
            <div class="stat-pill">94.8% Accuracy</div>
            <div class="stat-pill">91.2% F1</div>
            <div class="stat-pill">97.5% AUC</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Model check ────────────────────────────────────────────
    model, scaler = load_model()
    if model is None:
        st.error("⚠️  Model not found. Run the training pipeline first:")
        st.code(
            "python 01_load_data.py\npython 02_eda.py\n"
            "python 03_preprocess.py\npython 04_train.py",
            language="bash",
        )
        st.stop()

    # ── Patient name ───────────────────────────────────────────
    name = st.text_input("👤  Patient name (used in report)",
                         placeholder="e.g. Preethi K N")

    # ── Section A: Demographics & Lifestyle ───────────────────
    st.markdown(
        '<div class="sec-label">⚕️ &nbsp;Demographics &amp; Lifestyle</div>',
        unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        age  = st.slider("Age", 18, 100, 50)
        bmi  = st.slider("BMI", 15.0, 50.0, 27.5, 0.1)
    with c2:
        sex  = st.radio("Sex",            ["Male", "Female"],      horizontal=True)
        fam  = st.radio("Family History", ["No", "Yes"],           horizontal=True)

    c3, c4 = st.columns(2)
    with c3:
        htn   = st.radio("Hypertension", ["No", "Yes"],            horizontal=True)
    with c4:
        smoke = st.radio("Smoking",      ["Non-Smoker", "Smoker"], horizontal=True)

    # ── Section B: Clinical ────────────────────────────────────
    st.markdown(
        '<div class="sec-label">🧬 &nbsp;Clinical Measurements</div>',
        unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        glucose = st.slider("Glucose Fasting (mg/dL)", 50, 300, 115,
                             help="Normal <100 · Pre-diabetic 100-125 · Diabetic >125")
        bp      = st.slider("Systolic BP (mmHg)",       80, 200, 130,
                             help="Normal <120 · Elevated 120-129 · High ≥130")
        chol    = st.slider("Total Cholesterol (mg/dL)",100, 400, 210,
                             help="Optimal <200 · Borderline 200-239 · High ≥240")
    with c6:
        ldl     = st.slider("LDL (mg/dL)",    50,  250, 135,
                             help="Optimal <100 · High ≥160")
        hdl     = st.slider("HDL (mg/dL)",    20,  100,  45,
                             help="Risk factor <40 · Protective >60")
        insulin = st.slider("Insulin (μIU/mL)", 0.0, 50.0, 15.0, 0.5)

    # ── Section C: Wellness ────────────────────────────────────
    st.markdown(
        '<div class="sec-label">🏃 &nbsp;Activity &amp; Wellness</div>',
        unsafe_allow_html=True)

    c7, c8, c9 = st.columns(3)
    with c7:  act    = st.slider("Activity (0-10)", 0.0, 10.0, 3.0, 0.5)
    with c8:  stress = st.slider("Stress (0-10)",   0.0, 10.0, 5.0, 0.5)
    with c9:  sleep  = st.slider("Sleep (hrs)",     0.0, 12.0, 6.5, 0.5)

    steps = st.slider("Daily Steps", 0, 20_000, 5_000, 100,
                       help="WHO recommends 8,000–10,000 steps/day")

    # ── RUN ────────────────────────────────────────────────────
    run = st.button("🔮  Run Health Analysis", use_container_width=True)
    if not run:
        st.markdown("""
        <div style="text-align:center;color:#1e3a5f;font-size:12px;
                    margin-top:24px;line-height:1.8;">
            Fill in your clinical details above<br>
            and tap <b style="color:#38bdf8;">Run Health Analysis</b>
        </div>""", unsafe_allow_html=True)
        return

    # ── Prediction ─────────────────────────────────────────────
    with st.spinner("Analysing health profile …"):
        vec    = feature_vector(age, sex, bmi, fam, htn, smoke,
                                act, stress, steps, sleep,
                                chol, hdl, ldl, glucose, insulin, bp)
        scaled = scaler.transform(vec)
        prob   = float(model.predict_proba(scaled)[0, 1]) * 100

    # Metabolic age estimate
    met = float(age)
    if bmi > 25:       met += (bmi - 25) * 0.8
    if glucose > 100:  met += (glucose - 100) * 0.2
    if bp > 120:       met += (bp - 120) * 0.3
    if prob > 50:      met += 5
    if 18.5 < bmi < 25: met -= 2
    met_age  = int(met)
    age_diff = met_age - age
    age_col  = "#ef4444" if age_diff > 0 else "#10b981"
    arrow    = "▲" if age_diff > 0 else "▼"
    rc       = risk_color(prob)
    is_risk  = prob >= 50

    # ── Result header ──────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="sec-label">📊 &nbsp;Risk Assessment Result</div>',
        unsafe_allow_html=True)

    badge_cls = "risk-badge-risk" if is_risk else "risk-badge-norisk"
    badge_txt = "AT RISK" if is_risk else "NOT AT RISK"

    st.markdown(f"""
    <div class="gauge-card">
      <div class="gauge-flex">
        <div>
          <div style="font-size:10px;color:#64748b;letter-spacing:1px;
                      text-transform:uppercase;margin-bottom:2px;">Predicted Risk</div>
          <div class="risk-pct" style="color:{rc};">{int(prob)}<span class="risk-unit">%</span></div>
          <div class="{badge_cls}">{badge_txt}</div>
        </div>
        <div class="met-age-block">
          <div class="met-age-lbl">Metabolic Age</div>
          <div class="met-age-val">{met_age}</div>
          <div class="met-age-diff" style="color:{age_col};">
            {arrow} {abs(age_diff)} yrs vs chronological
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(int(prob), 100))

    # Quick metric widgets
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Glucose", f"{glucose}",
              delta=f"{glucose-100:+.0f}" if glucose != 100 else None,
              delta_color="inverse")
    m2.metric("BMI",     f"{bmi:.1f}",
              delta=f"{bmi-25:+.1f}"  if bmi != 25 else None,
              delta_color="inverse")
    m3.metric("LDL",     f"{ldl}",
              delta=f"{ldl-130:+.0f}" if ldl != 130 else None,
              delta_color="inverse")
    m4.metric("BP",      f"{bp}",
              delta=f"{bp-120:+.0f}"  if bp != 120 else None,
              delta_color="inverse")

    # Pill row
    st.markdown(f"""
    <div class="pill-row">
      <div class="info-pill">🩸 Glucose <b>{glucose}</b></div>
      <div class="info-pill">💓 BP <b>{bp}</b></div>
      <div class="info-pill">⚖️ BMI <b>{bmi:.1f}</b></div>
      <div class="info-pill">🧪 LDL <b>{ldl}</b></div>
      <div class="info-pill">🫀 HDL <b>{hdl}</b></div>
      <div class="info-pill">👟 <b>{int(steps):,}</b></div>
      <div class="info-pill">😴 <b>{sleep}h</b></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-label">📈 &nbsp;Health Radar &amp; Risk Drivers</div>',
        unsafe_allow_html=True)

    radar_vals = {
        "Metabolic": max(0, min(10, round(10 - (glucose/140)*5, 1))),
        "Cardio":    max(0, min(10, round(10 - (bp/140)*5, 1))),
        "Sleep":     max(0, min(10, round((sleep/8)*10, 1))),
        "Mental":    max(0, min(10, round(10 - stress, 1))),
        "Activity":  max(0, min(10, round(steps/1000, 1))),
    }

    drivers = []
    if bmi     > 25:        drivers.append(("BMI",      int((bmi-25)*1.5),       "#f59e0b"))
    if glucose > 100:       drivers.append(("Glucose",  int((glucose-100)*0.5),  "#ef4444"))
    if bp      > 120:       drivers.append(("BP",       int((bp-120)*0.5),        "#f59e0b"))
    if smoke   == "Smoker": drivers.append(("Smoking",  25,                        "#ef4444"))
    if ldl     > 130:       drivers.append(("LDL",      int((ldl-130)*0.3),       "#f97316"))
    if hdl     < 40:        drivers.append(("Low HDL",  15,                        "#eab308"))
    if stress  > 7:         drivers.append(("Stress",   int(stress*2),             "#a78bfa"))
    if sleep   < 7:         drivers.append(("Sleep-",   int((7-sleep)*3),          "#60a5fa"))
    if not drivers:         drivers.append(("Healthy",   5,                         "#10b981"))
    drivers.sort(key=lambda x: x[1], reverse=True)

    col_r, col_d = st.columns(2)
    with col_r:
        st.image(make_radar(radar_vals),    use_container_width=True)
    with col_d:
        st.image(make_drivers(drivers[:5]), use_container_width=True)

    st.image(make_gauge_arc(prob), use_container_width=True)

    # ── Clinical detail tabs ───────────────────────────────────
    st.markdown(
        '<div class="sec-label">🔬 &nbsp;Clinical Detail</div>',
        unsafe_allow_html=True)

    tab_m, tab_c, tab_l = st.tabs(["Metabolic", "Cardiovascular", "Lifestyle"])

    def dot(val, lo=None, hi=None):
        if hi is not None and val > hi: return "🔴"
        if lo is not None and val < lo: return "🔴"
        if hi is not None and val > (hi * 0.85): return "🟡"
        return "🟢"

    with tab_m:
        g_s = "🔴 Diabetic" if glucose>125 else ("🟡 Pre-diabetic" if glucose>100 else "🟢 Normal")
        b_s = "🔴 Obese"    if bmi>30       else ("🟡 Overweight"  if bmi>25     else "🟢 Normal")
        st.markdown(f"""
        <div class="detail-box">
          <p><b>Glucose:</b> {glucose} mg/dL &nbsp;{g_s}</p>
          <p><b>BMI:</b> {bmi:.1f} &nbsp;{b_s}</p>
          <p><b>Insulin:</b> {insulin} μIU/mL</p>
        </div>""", unsafe_allow_html=True)

    with tab_c:
        bp_s  = "🔴 High"       if bp>140   else ("🟡 Elevated"   if bp>120   else "🟢 Normal")
        ldl_s = "🔴 High"       if ldl>160  else ("🟡 Borderline" if ldl>130  else "🟢 Normal")
        hdl_s = "🔴 Low (risk)" if hdl<40   else ("🟡 Moderate"   if hdl<60   else "🟢 Good")
        c_s   = "🔴 High"       if chol>240 else ("🟡 Borderline" if chol>200 else "🟢 Optimal")
        st.markdown(f"""
        <div class="detail-box">
          <p><b>Systolic BP:</b> {bp} mmHg &nbsp;{bp_s}</p>
          <p><b>LDL:</b> {ldl} mg/dL &nbsp;{ldl_s}</p>
          <p><b>HDL:</b> {hdl} mg/dL &nbsp;{hdl_s}</p>
          <p><b>Cholesterol:</b> {chol} mg/dL &nbsp;{c_s}</p>
        </div>""", unsafe_allow_html=True)

    with tab_l:
        step_s = "🟢 Active" if steps>=8000 else ("🟡 Moderate" if steps>=5000 else "🔴 Sedentary")
        slp_s  = "🟢 Good"   if sleep>=7    else "🔴 Insufficient"
        st.markdown(f"""
        <div class="detail-box">
          <p><b>Steps:</b> {int(steps):,}/day &nbsp;{step_s}</p>
          <p><b>Sleep:</b> {sleep} hrs &nbsp;{slp_s}</p>
          <p><b>Stress level:</b> {stress}/10</p>
          <p><b>Physical activity:</b> {act}/10</p>
          <p><b>Smoking:</b> {smoke} &nbsp;{'🔴' if smoke=='Smoker' else '🟢'}</p>
        </div>""", unsafe_allow_html=True)

    # ── Recommendations ────────────────────────────────────────
    st.markdown(
        '<div class="sec-label">📋 &nbsp;Personalised Recommendations</div>',
        unsafe_allow_html=True)

    recs = []
    if glucose > 100:
        recs.append(("🩸","Glucose",
            f"{int(glucose)} mg/dL — Walk 20 min after meals. Low-glycaemic diet (whole grains, legumes)."))
    if bmi > 30:
        recs.append(("⚖️","Weight",
            f"BMI {bmi:.1f} (Obese) — Aim for 5-7% loss in 3 months via caloric deficit + strength training."))
    elif bmi > 25:
        recs.append(("⚠️","Weight",
            f"BMI {bmi:.1f} (Overweight) — Reduce refined carbs. Add 30 min brisk walking daily."))
    if bp > 130:
        recs.append(("💓","Blood Pressure",
            f"{int(bp)} mmHg — Try the DASH diet (low sodium, high K⁺). Limit caffeine & alcohol."))
    if ldl > 130:
        recs.append(("🍔","LDL Cholesterol",
            f"{int(ldl)} mg/dL — Increase soluble fibre (oats, beans). Reduce saturated fats."))
    if hdl < 40:
        recs.append(("🏃","HDL Cholesterol",
            f"{int(hdl)} mg/dL — Aerobic exercise raises HDL. Target 150 min/week moderate cardio."))
    if smoke == "Smoker":
        recs.append(("🫁","Smoking",
            "Biggest modifiable risk factor. Consider NRT patches or varenicline. Seek GP support."))
    if sleep < 7:
        recs.append(("😴","Sleep",
            f"{sleep} hrs/night — Target 7-9 hrs. Consistent schedule, no screens 1 hr before bed."))
    if steps < 6000:
        recs.append(("👟","Steps",
            f"{int(steps):,}/day — Target 8,000-10,000. Park further, take stairs, walk during calls."))
    if stress > 7:
        recs.append(("🧠","Stress",
            "High stress elevates cortisol & CVD risk. Try 4-7-8 breathing, mindfulness, or yoga."))
    if not recs:
        recs.append(("🌟","All metrics look healthy!",
            "Keep up your habits. Schedule an annual health check to maintain this status."))

    for ic, ti, tx in recs:
        st.markdown(f"""
        <div class="rec-card">
          <div class="rec-title">{ic} &nbsp;{ti}</div>
          <div class="rec-body">{tx}</div>
        </div>""", unsafe_allow_html=True)

    # ── Export ─────────────────────────────────────────────────
    st.markdown(
        '<div class="sec-label">📤 &nbsp;Export Report</div>',
        unsafe_allow_html=True)

    fname = (name or "Patient").replace(" ", "_")
    ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    inputs = {
        "Glucose (mg/dL)":     glucose,  "Systolic BP (mmHg)": bp,
        "BMI":                  f"{bmi:.1f}",
        "LDL (mg/dL)":         ldl,      "HDL (mg/dL)":       hdl,
        "Cholesterol (mg/dL)": chol,     "Insulin (μIU/mL)":  insulin,
        "Steps/day":            int(steps), "Sleep (hrs)":     sleep,
        "Stress (0-10)":        stress,   "Activity (0-10)":   act,
        "Smoking":              smoke,    "Hypertension":      htn,
        "Family History":       fam,
    }

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.download_button(
            "📊  Download CSV",
            data=export_csv(name, prob, met_age, age_diff, recs, inputs),
            file_name=f"{fname}_MediRisk_{ts}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_e2:
        if PDF_OK:
            st.download_button(
                "📄  Download PDF",
                data=export_pdf(name, prob, met_age, age_diff, recs, inputs),
                file_name=f"{fname}_MediRisk_{ts}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.info("pip install reportlab  to enable PDF export")

    json_payload = {
        "patient": name or "Anonymous",
        "timestamp": ts,
        "risk_score_pct": round(prob, 2),
        "assessment": "AT RISK" if is_risk else "NOT AT RISK",
        "metabolic_age": met_age,
        "inputs": {
            "age": age, "sex": sex, "bmi": bmi,
            "glucose": glucose, "systolic_bp": bp,
            "ldl": ldl, "hdl": hdl, "cholesterol": chol,
            "insulin": insulin, "steps": steps,
            "sleep": sleep, "stress": stress,
            "activity": act, "smoking": smoke,
            "hypertension": htn, "family_history": fam,
        },
        "recommendations": [
            {"icon": ic, "title": ti, "detail": tx}
            for ic, ti, tx in recs
        ],
    }
    st.download_button(
        "🗂️  Download Full JSON",
        data=json.dumps(json_payload, indent=2),
        file_name=f"{fname}_MediRisk_{ts}.json",
        mime="application/json",
        use_container_width=True,
    )

    # ── Footer ─────────────────────────────────────────────────
    st.markdown("""
    <div class="footer-note">
        🏥 MediRisk AI &nbsp;·&nbsp;
        Stacking Ensemble (RF + LightGBM + XGBoost + CatBoost)<br>
        94.8% Accuracy &nbsp;·&nbsp; 91.2% F1 &nbsp;·&nbsp; 97.5% AUC<br><br>
        <b style="color:#334155;">
        For clinical support only. Not a substitute for professional medical diagnosis.
        </b>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
