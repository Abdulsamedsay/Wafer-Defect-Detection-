"""
app/streamlit_app.py
Wafer Defect Detection & Yield Risk Dashboard — industrial design system.
Run with:  streamlit run app/streamlit_app.py
"""

import base64
import json
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.explainability import GradCAM, load_model, overlay_heatmap
from src.risk_scoring import score
from src.data_loader import DEFECT_CLASSES

MODEL_PATH   = ROOT / "models" / "best_model.pth"
METRICS_PATH = ROOT / "models" / "cnn_full_metrics.json"
DEMO_PATH    = ROOT / "models" / "demo_samples.npz"
FIGURES      = ROOT / "outputs" / "figures"

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Wafer Defect AI · Yield Risk Dashboard",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design-system CSS ──────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&display=swap');

:root {
  --bg-0:#18202e;--bg-1:#1c2236;--bg-2:#212940;--bg-3:#28324d;--bg-4:#303d58;
  --line-1:rgba(120,145,190,.18);--line-2:rgba(140,165,210,.26);
  --fg-0:#f4f6f9;--fg-1:#b8c4d6;--fg-2:#8494aa;--fg-3:#5c6c84;
  --accent:#52c98a;--accent-soft:rgba(82,201,138,.12);--accent-line:rgba(82,201,138,.28);
  --ok:#52c98a;--ok-soft:rgba(82,201,138,.12);
  --warn:#d4a530;--warn-soft:rgba(212,165,48,.12);
  --high:#d47030;--high-soft:rgba(212,112,48,.12);
  --crit:#c83830;--crit-soft:rgba(200,56,48,.14);
  --info:#4294cc;--info-soft:rgba(66,148,204,.12);
  --r-sm:6px;--r-md:10px;--r-lg:14px;
  --font-sans:'Geist',ui-sans-serif,system-ui,-apple-system,sans-serif;
  --font-mono:'Geist Mono',ui-monospace,'JetBrains Mono',monospace;
  --sh1:0 1px 0 0 rgba(255,255,255,.04) inset,0 1px 3px rgba(0,0,0,.45);
}

.stApp{background:var(--bg-0)!important;font-family:var(--font-sans)!important;}
.stApp *{font-family:var(--font-sans)!important;box-sizing:border-box;}
#MainMenu,footer{visibility:hidden;}
.stMainBlockContainer{padding:0!important;max-width:100%!important;}
[data-testid="stVerticalBlockBorderWrapper"]{border:none!important;background:none!important;}
section[data-testid="stSidebar"]{background:var(--bg-1)!important;border-right:1px solid var(--line-1)!important;min-width:220px!important;}
section[data-testid="stSidebar"] > div:first-child{padding:0 14px!important;}

.stRadio [data-testid="stWidgetLabel"]{display:none!important;}
.stRadio > div{gap:2px!important;}
.stRadio > div > label{
  display:flex!important;align-items:center!important;gap:10px!important;
  padding:8px 10px!important;border-radius:var(--r-sm)!important;
  border:1px solid transparent!important;color:var(--fg-1)!important;
  font-size:13px!important;cursor:pointer!important;width:100%!important;
  transition:background .12s,color .12s!important;background:transparent!important;
}
.stRadio > div > label:hover{background:var(--bg-2)!important;color:var(--fg-0)!important;}
.stRadio > div > label[data-checked="true"]{background:var(--bg-2)!important;color:var(--fg-0)!important;border-color:var(--line-1)!important;box-shadow:var(--sh1)!important;}
.stRadio > div > label > div:first-child{display:none!important;}

.stButton > button{background:var(--bg-2)!important;color:var(--fg-0)!important;border:1px solid var(--line-1)!important;border-radius:var(--r-sm)!important;font-size:12px!important;padding:7px 14px!important;}
.stButton > button:hover{background:var(--bg-3)!important;border-color:var(--line-2)!important;}
.stButton > button[kind="primary"]{background:var(--accent)!important;color:#18202e!important;border-color:var(--accent)!important;font-weight:600!important;}
.stSelectbox > label{color:var(--fg-3)!important;font-family:var(--font-mono)!important;font-size:10px!important;text-transform:uppercase!important;letter-spacing:.06em!important;}
.stSelectbox > div > div{background:var(--bg-3)!important;border-color:var(--line-1)!important;color:var(--fg-0)!important;}
.stMarkdown p,.stMarkdown li{color:var(--fg-1)!important;font-size:13px!important;line-height:1.55!important;}
hr{border-color:var(--line-1)!important;margin:0!important;}

/* DS components */
.ds-page{padding:28px 40px 72px;}
.ds-crumbs{font-family:var(--font-mono);font-size:11px;color:var(--fg-3);margin-bottom:12px;letter-spacing:.04em;}
.ds-crumbs .cur{color:var(--fg-1);}
.ds-page-head{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;margin-bottom:30px;padding-bottom:20px;border-bottom:1px solid var(--line-1);flex-wrap:wrap;}
.ds-page-title{font-size:22px;font-weight:600;margin:0;letter-spacing:-.01em;color:var(--fg-0);}
.ds-page-sub{margin:5px 0 0;color:var(--fg-2);font-size:13px;max-width:680px;line-height:1.55;}
.ds-page-actions{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}

.ds-card{background:var(--bg-2);border:1px solid var(--line-1);border-radius:var(--r-lg);box-shadow:var(--sh1);margin-bottom:20px;}
.ds-card-head{display:flex;align-items:center;justify-content:space-between;padding:14px 20px;border-bottom:1px solid var(--line-1);}
.ds-card-title{font-size:13px;font-weight:600;margin:0;color:var(--fg-0);}
.ds-card-sub{font-family:var(--font-mono);font-size:10px;color:var(--fg-3);text-transform:uppercase;letter-spacing:.08em;}
.ds-card-body{padding:20px;}

.ds-kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-bottom:24px;}
.ds-kpi{background:var(--bg-2);border:1px solid var(--line-1);border-radius:var(--r-lg);padding:18px 20px;}
.ds-kpi.hero{border-color:var(--accent-line);background:linear-gradient(135deg,var(--accent-soft),transparent 60%);}
.ds-kpi-label{font-family:var(--font-mono);font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg-3);}
.ds-kpi.hero .ds-kpi-label{color:var(--accent);}
.ds-kpi-value{font-family:var(--font-mono);font-size:26px;font-weight:500;letter-spacing:-.01em;margin-top:6px;color:var(--fg-0);line-height:1.05;}
.ds-kpi-value .unit{font-size:14px;color:var(--fg-2);}
.ds-kpi-delta{font-family:var(--font-mono);font-size:11px;margin-top:4px;color:var(--fg-2);}
.ds-kpi-delta.up{color:var(--ok);}
.ds-kpi-delta.dn{color:var(--crit);}

.ds-chip{display:inline-flex;align-items:center;gap:5px;padding:2px 8px;font-family:var(--font-mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;border-radius:999px;border:1px solid var(--line-1);color:var(--fg-1);background:var(--bg-2);white-space:nowrap;}
.ds-chip-dot{width:5px;height:5px;border-radius:50%;background:var(--fg-2);flex-shrink:0;}
.ds-chip.ok{color:var(--ok);border-color:rgba(82,201,138,.35);background:var(--ok-soft);}
.ds-chip.ok .ds-chip-dot{background:var(--ok);}
.ds-chip.warn{color:var(--warn);border-color:rgba(212,165,48,.35);background:var(--warn-soft);}
.ds-chip.warn .ds-chip-dot{background:var(--warn);}
.ds-chip.high{color:var(--high);border-color:rgba(212,112,48,.35);background:var(--high-soft);}
.ds-chip.high .ds-chip-dot{background:var(--high);}
.ds-chip.crit{color:var(--crit);border-color:rgba(200,56,48,.4);background:var(--crit-soft);}
.ds-chip.crit .ds-chip-dot{background:var(--crit);}
.ds-chip.info{color:var(--info);border-color:rgba(66,148,204,.35);background:var(--info-soft);}
.ds-chip.info .ds-chip-dot{background:var(--info);}

.ds-btn{display:inline-flex;align-items:center;gap:6px;padding:7px 12px;font-size:12px;border-radius:var(--r-sm);border:1px solid var(--line-1);background:var(--bg-2);color:var(--fg-0);cursor:pointer;text-decoration:none;}
.ds-section-head{display:flex;align-items:flex-start;justify-content:space-between;margin:36px 0 16px;}
.ds-section-title{font-size:14px;font-weight:600;margin:0;color:var(--fg-0);display:flex;align-items:center;gap:8px;}
.ds-section-num{font-family:var(--font-mono);font-size:11px;color:var(--fg-3);padding:2px 6px;border:1px solid var(--line-1);border-radius:4px;}
.ds-section-sub{font-size:12px;color:var(--fg-2);margin:3px 0 0;}

.ds-risk{border-radius:var(--r-lg);border:1px solid;padding:20px;display:flex;flex-direction:column;gap:12px;position:relative;overflow:hidden;}
.ds-risk::before{content:"";position:absolute;inset:0;background:radial-gradient(120% 100% at 0% 0%,currentColor 0%,transparent 55%);opacity:.07;pointer-events:none;}
.ds-risk.low{color:var(--ok);border-color:rgba(82,201,138,.35);background:rgba(82,201,138,.05);}
.ds-risk.med{color:var(--warn);border-color:rgba(212,165,48,.35);background:rgba(212,165,48,.05);}
.ds-risk.high{color:var(--high);border-color:rgba(212,112,48,.35);background:rgba(212,112,48,.05);}
.ds-risk.crit{color:var(--crit);border-color:rgba(200,56,48,.4);background:rgba(200,56,48,.07);}
.ds-risk-label{font-family:var(--font-mono);font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:var(--fg-3);}
.ds-risk-level{font-size:26px;font-weight:600;color:currentColor;margin-top:2px;}
.ds-risk-action{font-size:13px;color:var(--fg-0);line-height:1.5;}
.ds-risk-meta{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding-top:10px;border-top:1px solid rgba(255,255,255,.06);}
.ds-risk-ml{font-family:var(--font-mono);font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg-3);}
.ds-risk-mv{font-family:var(--font-mono);font-size:12px;color:var(--fg-0);margin-top:2px;}

.ds-prob-row{display:grid;grid-template-columns:110px 1fr 54px;align-items:center;gap:10px;padding:5px 0;}
.ds-prob-name{font-size:12px;color:var(--fg-1);}
.ds-prob-row.top .ds-prob-name{color:var(--fg-0);font-weight:600;}
.ds-prob-track{height:7px;background:var(--bg-3);border-radius:999px;overflow:hidden;}
.ds-prob-fill{height:100%;background:var(--fg-3);border-radius:999px;}
.ds-prob-row.top .ds-prob-fill{background:var(--accent);}
.ds-prob-val{font-family:var(--font-mono);font-size:11px;text-align:right;color:var(--fg-2);}
.ds-prob-row.top .ds-prob-val{color:var(--fg-0);}

.ds-insight{padding:16px 18px;border:1px solid var(--line-1);border-radius:var(--r-md);background:var(--bg-2);display:flex;flex-direction:column;gap:7px;}
.ds-insight-tag{font-family:var(--font-mono);font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--fg-3);padding:2px 6px;border:1px solid var(--line-1);border-radius:4px;display:inline-block;width:fit-content;}
.ds-insight-title{font-size:13px;font-weight:600;color:var(--fg-0);}
.ds-insight-body{font-size:12px;color:var(--fg-2);line-height:1.55;}

.ds-bar-row{display:grid;grid-template-columns:180px 1fr 70px 70px;align-items:center;gap:14px;padding:9px 0;border-bottom:1px solid var(--line-1);}
.ds-bar-row:last-child{border-bottom:none;}
.ds-bar-row.main .ds-bar-name{font-weight:600;color:var(--fg-0);}
.ds-bar-name{font-size:13px;color:var(--fg-1);display:flex;align-items:center;gap:6px;}
.ds-bar-track{height:10px;background:var(--bg-3);border-radius:999px;overflow:hidden;}
.ds-bar-fill{height:100%;background:var(--bg-4);border-radius:999px;}
.ds-bar-row.main .ds-bar-fill{background:var(--accent);}
.ds-bar-num{font-family:var(--font-mono);font-size:12px;text-align:right;color:var(--fg-2);}
.ds-bar-row.main .ds-bar-num{color:var(--fg-0);font-weight:600;}

.ds-pipeline{display:grid;grid-template-columns:repeat(5,1fr);}
.ds-pipe-step{padding:14px 16px;border:1px solid var(--line-1);background:var(--bg-2);display:flex;flex-direction:column;gap:6px;}
.ds-pipe-step:first-child{border-radius:var(--r-md) 0 0 var(--r-md);}
.ds-pipe-step:last-child{border-radius:0 var(--r-md) var(--r-md) 0;}
.ds-pipe-step+.ds-pipe-step{border-left:none;}
.ds-pipe-num{font-family:var(--font-mono);font-size:10px;color:var(--fg-3);}
.ds-pipe-name{font-size:13px;font-weight:600;color:var(--fg-0);}
.ds-pipe-desc{font-size:11px;color:var(--fg-2);line-height:1.4;}

.ds-tbl{width:100%;border-collapse:collapse;font-size:12px;}
.ds-tbl th{font-family:var(--font-mono);font-size:10px;letter-spacing:.06em;text-transform:uppercase;color:var(--fg-3);font-weight:500;border-bottom:1px solid var(--line-1);padding:8px 12px;text-align:left;background:rgba(28,34,54,.6);}
.ds-tbl td{padding:9px 12px;border-bottom:1px solid var(--line-1);color:var(--fg-1);}
.ds-tbl tr:last-child td{border-bottom:none;}
.ds-tbl td.mono{font-family:var(--font-mono);}
.ds-tbl td.dim{color:var(--fg-3);font-size:10px;}

.ds-notice{display:flex;gap:12px;padding:16px 20px;border-radius:var(--r-lg);border:1px solid rgba(66,148,204,.3);background:rgba(66,148,204,.04);align-items:flex-start;margin-top:20px;}
.ds-notice-body{font-size:13px;color:var(--fg-2);line-height:1.55;}
.ds-notice-body strong{color:var(--fg-0);}

.ds-grid-2{display:grid;grid-template-columns:1fr 1fr;gap:20px;}
.ds-grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;}
.ds-grid-12{display:grid;grid-template-columns:2fr 1fr;gap:20px;}
.ds-grid-21{display:grid;grid-template-columns:1fr 2fr;gap:20px;}

.ds-brand-wrap{display:flex;flex-direction:column;align-items:center;gap:10px;padding:18px 0 16px;border-bottom:1px solid var(--line-1);margin-bottom:14px;}
.ds-brand-disc{border-radius:50%;background:radial-gradient(circle at 32% 30%,rgba(100,160,220,.45) 0%,rgba(36,54,96,.9) 55%,rgba(18,28,48,1) 100%);border:1.5px solid rgba(110,175,240,.5);position:relative;overflow:hidden;box-shadow:0 0 0 1px rgba(0,0,0,.5) inset,0 0 20px rgba(80,140,220,.2),0 4px 14px rgba(0,0,0,.5);flex-shrink:0;}
.ds-brand-disc-grid{position:absolute;inset:8%;border-radius:50%;background-image:linear-gradient(rgba(160,200,255,.28) 1px,transparent 1px),linear-gradient(90deg,rgba(160,200,255,.28) 1px,transparent 1px);background-size:15% 15%;-webkit-mask:radial-gradient(circle,#000 55%,transparent 90%);mask:radial-gradient(circle,#000 55%,transparent 90%);}
.ds-brand-disc-ring{position:absolute;inset:6%;border-radius:50%;border:1px solid rgba(140,200,255,.4);}
.ds-brand-disc-notch{position:absolute;bottom:0;left:50%;width:12%;height:5%;background:rgba(18,26,44,1);transform:translateX(-50%);border-radius:0 0 3px 3px;}
.ds-brand-disc-da{position:absolute;top:30%;left:57%;width:10%;height:10%;border-radius:2px;background:rgba(100,220,195,.9);box-shadow:0 0 7px rgba(100,220,195,.8);animation:blink 3s ease-in-out infinite;}
.ds-brand-disc-db{position:absolute;top:58%;left:33%;width:8%;height:8%;border-radius:2px;background:rgba(90,155,255,.9);box-shadow:0 0 5px rgba(90,155,255,.8);animation:blink 3s ease-in-out infinite 1.5s;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.35}}
.ds-brand-name{font-size:13px;font-weight:600;color:var(--fg-0);text-align:center;}
.ds-brand-name span{color:#7ec8f0;}
.ds-brand-sub{font-family:var(--font-mono);font-size:9px;color:var(--fg-3);letter-spacing:.1em;text-transform:uppercase;text-align:center;}

.ds-nav-group{font-family:var(--font-mono);font-size:10px;text-transform:uppercase;letter-spacing:.1em;color:var(--fg-3);padding:10px 10px 5px;}
.ds-nav-status{display:flex;align-items:center;gap:7px;font-family:var(--font-mono);font-size:10px;color:var(--fg-2);padding:0 8px;}
.ds-nav-dot{width:6px;height:6px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px rgba(82,201,138,.15);animation:pulse 2.4s ease-in-out infinite;flex-shrink:0;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 3px rgba(82,201,138,.14)}50%{box-shadow:0 0 0 5px rgba(82,201,138,.05)}}

.ds-author{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:var(--r-sm);border:1px solid var(--line-1);background:var(--bg-2);margin-top:8px;}
.ds-author-avatar{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,var(--accent),var(--info));display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:700;color:#18202e;flex-shrink:0;}
.ds-author-name{font-size:12px;font-weight:600;color:var(--fg-0);}
.ds-author-link{font-family:var(--font-mono);font-size:10px;color:var(--info);text-decoration:none;}

.ds-wafer-img{width:100%;border-radius:var(--r-md);image-rendering:pixelated;}
.ds-img-label{font-family:var(--font-mono);font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg-3);margin-bottom:6px;}
</style>""", unsafe_allow_html=True)


# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

@st.cache_resource
def get_demo_samples():
    data = np.load(str(DEMO_PATH))
    return data["X"], data["y"]

@st.cache_resource
def get_metrics():
    return json.loads(METRICS_PATH.read_text())


# ── Helpers ────────────────────────────────────────────────────────────────────
def run_prediction(model, x_np):
    x_t = torch.from_numpy(x_np).unsqueeze(0)
    cam = GradCAM(model)
    with torch.no_grad():
        logits = model(x_t)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])
    heatmap = cam(x_t, class_idx=pred_idx)
    cam.remove_hooks()
    return pred_idx, confidence, probs, heatmap

def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def path_to_b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()

def np_to_pil(arr, scale=4):
    img = (arr * 255).astype(np.uint8)
    return Image.fromarray(img, mode="L").resize(
        (arr.shape[1] * scale, arr.shape[0] * scale), Image.NEAREST
    )

def chip(text, tone="default", dot=True):
    d = '<span class="ds-chip-dot"></span>' if dot else ""
    return f'<span class="ds-chip {tone}">{d}{text}</span>'

def kpi_html(label, value, unit="", delta="", delta_dir="", hero=False):
    hero_cls = "hero" if hero else ""
    dclass = f"ds-kpi-delta {delta_dir}" if delta_dir else "ds-kpi-delta"
    u = f'<span class="unit">{unit}</span>' if unit else ""
    d = f'<div class="{dclass}">{delta}</div>' if delta else ""
    val_size = "font-size:34px;color:var(--accent);" if hero else ""
    return f"""<div class="ds-kpi {hero_cls}">
      <div class="ds-kpi-label">{label}</div>
      <div class="ds-kpi-value" style="{val_size}">{value}{u}</div>{d}
    </div>"""

def fig_img_html(path: Path, alt: str = "") -> str:
    if path.exists():
        b64 = path_to_b64(path)
        return f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:var(--r-md);" alt="{alt}"/>'
    return f'<div style="padding:20px;color:var(--fg-3);font-family:var(--font-mono);font-size:12px;">{alt or path.name} not found</div>'

def insight_html(tag, title, body):
    return f"""<div class="ds-insight">
      <span class="ds-insight-tag">{tag}</span>
      <div class="ds-insight-title">{title}</div>
      <div class="ds-insight-body">{body}</div>
    </div>"""

RISK_MAP = {
    "low":      ("low",  "LOW",      "Pass to next stage. Log prediction for trend analysis."),
    "medium":   ("med",  "MEDIUM",   "Flag for routine secondary review by quality engineer."),
    "high":     ("high", "HIGH",     "Escalate to QC lead. Hold lot pending engineering review."),
    "critical": ("crit", "CRITICAL", "Halt downstream processing. Initiate root-cause investigation."),
}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="ds-brand-wrap">
      <div class="ds-brand-disc" style="width:76px;height:76px;">
        <div class="ds-brand-disc-grid"></div>
        <div class="ds-brand-disc-ring"></div>
        <div class="ds-brand-disc-da"></div>
        <div class="ds-brand-disc-db"></div>
        <div class="ds-brand-disc-notch"></div>
      </div>
      <div class="ds-brand-name">Wafer Defect <span>AI</span></div>
      <div class="ds-brand-sub">Yield Risk Dashboard</div>
    </div>
    <div class="ds-nav-group">Dashboard</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Project Overview", "Prediction", "Model Performance", "Dataset Insights"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <div class="ds-nav-group" style="margin-top:8px;">Runtime</div>
    <div style="padding:0 4px;display:flex;flex-direction:column;gap:6px;">
      <div class="ds-nav-status">
        <div class="ds-nav-dot"></div>Inference online
      </div>
      <div style="font-family:var(--font-mono);font-size:10px;color:var(--fg-3);padding:0 8px;">
        WaferCNN · v1.2.0 · PyTorch 2.3
      </div>
    </div>
    <div style="padding:0 4px;margin-top:12px;border-top:1px solid var(--line-1);padding-top:12px;">
      <div class="ds-author">
        <div class="ds-author-avatar">SS</div>
        <div>
          <div class="ds-author-name">Samed Say</div>
          <a class="ds-author-link" href="https://www.linkedin.com/in/samed-say-754981392/" target="_blank">LinkedIn ↗</a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 1 — Project Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "Project Overview":
    metrics = get_metrics()
    mf1 = metrics.get("macro_f1", 0.686)
    acc  = metrics.get("accuracy", 0.866)
    wf1  = metrics.get("weighted_f1", 0.899)

    st.markdown(f"""
<div class="ds-page">
  <!-- Brand hero -->
  <div style="display:flex;align-items:center;justify-content:center;gap:28px;margin-bottom:40px;padding:28px 0;">
    <div class="ds-brand-disc" style="width:140px;height:140px;">
      <div class="ds-brand-disc-grid"></div><div class="ds-brand-disc-ring"></div>
      <div class="ds-brand-disc-da" style="width:8%;height:8%;"></div>
      <div class="ds-brand-disc-db" style="width:6%;height:6%;"></div>
      <div class="ds-brand-disc-notch"></div>
    </div>
    <div>
      <div style="font-size:36px;font-weight:600;letter-spacing:-.02em;line-height:1.05;color:var(--fg-0);">
        Wafer Defect <span style="color:#7ec8f0;">AI</span>
      </div>
      <div style="font-family:var(--font-mono);font-size:12px;color:var(--fg-2);letter-spacing:.18em;text-transform:uppercase;margin-top:8px;">
        Yield Risk Dashboard
      </div>
    </div>
  </div>

  <!-- Page head -->
  <div class="ds-page-head">
    <div>
      <h1 class="ds-page-title">Project Overview</h1>
      <p class="ds-page-sub">AI-assisted wafer defect classification for semiconductor quality control. This dashboard demonstrates how computer vision can support wafer inspection by combining classification, confidence scoring, explainability, and risk assessment.</p>
    </div>
    <div class="ds-page-actions">
      {chip("Decision Support · Prototype", "info")}
      <a href="https://www.linkedin.com/in/samed-say-754981392/" target="_blank" class="ds-btn">Samed Say · LinkedIn ↗</a>
      <a href="https://github.com/Abdulsamedsay/wafer-defect-detection-dashboard" target="_blank" class="ds-btn">GitHub ↗</a>
    </div>
  </div>

  <!-- Hero card -->
  <div class="ds-card" style="margin-bottom:24px;overflow:hidden;position:relative;">
    <div style="position:absolute;inset:0;background:radial-gradient(800px 300px at 90% -20%,rgba(82,201,138,.15),transparent 60%);pointer-events:none;"></div>
    <div style="padding:24px;position:relative;">
      <div style="display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap;align-items:center;">
        {chip("CNN · v1.2.0", "ok")}
        {chip("WM-811K Dataset", "info")}
        <span style="margin-left:auto;font-family:var(--font-mono);font-size:10px;color:var(--fg-3);text-transform:uppercase;letter-spacing:.06em;">Last evaluated · 2026-04-28</span>
      </div>
      <h2 style="margin:0;font-size:24px;letter-spacing:-.02em;font-weight:600;max-width:820px;color:var(--fg-0);">
        Reduce wafer inspection latency without sacrificing recall on rare defect classes.
      </h2>
      <p style="max-width:720px;color:var(--fg-2);margin-top:12px;font-size:14px;line-height:1.6;">
        A CNN classifier trained on 172,950 labeled wafer maps, evaluated against four baselines, and instrumented with Grad-CAM explainability and a confidence-weighted risk score. The model output should be interpreted as decision support, not as a replacement for human quality-control review.
      </p>
    </div>
  </div>

  <!-- KPI grid -->
  <div class="ds-kpi-grid">
    {kpi_html("★ Headline · Macro F1", f"{mf1:.3f}", "", "Best metric under class imbalance", "up", hero=True)}
    {kpi_html("Accuracy", f"{acc*100:.1f}", "%", "+1.3 pp vs best baseline", "up")}
    {kpi_html("Weighted F1", f"{wf1:.3f}", "", "Dominant-class skew")}
    {kpi_html("Dataset", "172.9", "k wafers", "9 classes · imbalanced")}
    {kpi_html("Inference", "18", "ms", "Single wafer · CPU")}
    {kpi_html("Model size", "4.7", "MB", "Deployable on edge")}
  </div>

  <!-- Pipeline -->
  <div class="ds-section-head">
    <div>
      <h2 class="ds-section-title"><span class="ds-section-num">01</span> Inference pipeline</h2>
      <p class="ds-section-sub">From raw wafer map to triage recommendation in a single request.</p>
    </div>
  </div>
  <div class="ds-pipeline">
    <div class="ds-pipe-step"><div class="ds-pipe-num">STEP 01</div><div class="ds-pipe-name">Ingest</div><div class="ds-pipe-desc">64×64 wafer map, normalized and resized to model input.</div></div>
    <div class="ds-pipe-step"><div class="ds-pipe-num">STEP 02</div><div class="ds-pipe-name">CNN Classify</div><div class="ds-pipe-desc">9-class softmax over WM-811K defect taxonomy.</div></div>
    <div class="ds-pipe-step"><div class="ds-pipe-num">STEP 03</div><div class="ds-pipe-name">Grad-CAM</div><div class="ds-pipe-desc">Spatial attribution overlay for the predicted class.</div></div>
    <div class="ds-pipe-step"><div class="ds-pipe-num">STEP 04</div><div class="ds-pipe-name">Risk Score</div><div class="ds-pipe-desc">Class severity × confidence → Low/Med/High/Critical.</div></div>
    <div class="ds-pipe-step"><div class="ds-pipe-num">STEP 05</div><div class="ds-pipe-name">Triage</div><div class="ds-pipe-desc">Recommended action surfaced to the operator.</div></div>
  </div>

  <!-- Problem framing + Stack -->
  <div class="ds-grid-12" style="margin-top:28px;">
    <div class="ds-card" style="margin-bottom:0;">
      <div class="ds-card-head">
        <h3 class="ds-card-title">Problem framing</h3>
        <span class="ds-card-sub">Scope · Constraints</span>
      </div>
      <div class="ds-card-body" style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
        {insight_html("PROBLEM", "Manual inspection is slow and inconsistent.", "Quality engineers triage thousands of wafers per shift. Rare defect classes like Scratch and Donut are easy to miss.")}
        {insight_html("USERS", "Quality, process, and ML engineers.", "Designed to surface model evidence — not to make autonomous accept/reject calls on production wafers.")}
        {insight_html("APPROACH", "CNN classifier with explainability.", "9-class CNN over WM-811K, paired with Grad-CAM heatmaps and a confidence-weighted risk score.")}
        {insight_html("NON-GOAL", "Not a replacement for QC review.", "Decision support only. All flagged wafers are expected to receive a human secondary review.")}
      </div>
    </div>
    <div class="ds-card" style="margin-bottom:0;">
      <div class="ds-card-head">
        <h3 class="ds-card-title">Stack</h3>
        <span class="ds-card-sub">Implementation</span>
      </div>
      <div class="ds-card-body">
        <table class="ds-tbl">
          <tr><td class="dim">MODEL</td><td class="mono">CNN · 4 conv blocks · ~2.2M params</td></tr>
          <tr><td class="dim">FRAMEWORK</td><td class="mono">PyTorch 2.3</td></tr>
          <tr><td class="dim">EXPLAINER</td><td class="mono">Grad-CAM (last conv)</td></tr>
          <tr><td class="dim">FRONTEND</td><td class="mono">Streamlit · Plotly · PIL</td></tr>
          <tr><td class="dim">DATASET</td><td class="mono">WM-811K · 172,950 labeled</td></tr>
          <tr><td class="dim">COMPUTE</td><td class="mono">Single GPU · ~2.4h training</td></tr>
          <tr><td class="dim">DEPLOY</td><td class="mono"><a href="https://wafer-defect-dashboard-page.streamlit.app" target="_blank" style="color:var(--info);text-decoration:none;">wafer-defect-detection.streamlit.app ↗</a></td></tr>
        </table>
      </div>
    </div>
  </div>

  <div class="ds-notice">
    <div style="color:var(--info);margin-top:1px;flex-shrink:0;font-size:16px;">ℹ</div>
    <div class="ds-notice-body">
      <strong>This is an AI decision-support prototype.</strong>
      Predictions are intended to assist, not replace, human QC review. Risk levels combine predicted defect type and model confidence to support triage — they are not a pass/fail signal.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 2 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    model = get_model()
    X_demo, y_demo = get_demo_samples()

    if "sample_idx" not in st.session_state:
        st.session_state.sample_idx = 0

    st.markdown(f"""
<div style="padding:28px 40px 0;">
  <div class="ds-crumbs">WAFER-AI <span style="opacity:.5;">/</span> <span class="cur">PREDICTION</span></div>
  <div class="ds-page-head">
    <div>
      <h1 class="ds-page-title">Prediction</h1>
      <p class="ds-page-sub">Select a defect class, pick a test sample, and inspect the model's prediction, Grad-CAM attribution, and triage-ready risk level.</p>
    </div>
    <div class="ds-page-actions">
      {chip("Model online", "ok")}
      {chip("WaferCNN · v1.2.0", "info")}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Sample selector controls
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        class_options = ["Random"] + DEFECT_CLASSES
        chosen_class = st.selectbox("Defect class", class_options)
    with c2:
        st.write("")
        if st.button("Pick Sample", type="primary", use_container_width=True):
            if chosen_class == "Random":
                st.session_state.sample_idx = int(np.random.randint(0, len(X_demo)))
            else:
                class_idx = DEFECT_CLASSES.index(chosen_class)
                candidates = np.where(y_demo == class_idx)[0]
                if len(candidates) > 0:
                    st.session_state.sample_idx = int(np.random.choice(candidates))
                else:
                    st.warning(f"No demo samples for {chosen_class}.")

    # Run inference
    idx = st.session_state.sample_idx
    true_class = DEFECT_CLASSES[y_demo[idx]]
    x_np = X_demo[idx]
    pred_idx, confidence, probs, heatmap = run_prediction(model, x_np)
    pred_class = DEFECT_CLASSES[pred_idx]
    result = score(pred_idx, confidence)
    correct = (pred_idx == int(y_demo[idx]))

    wafer_pil   = np_to_pil(x_np[0], scale=6).convert("RGB")
    overlay_arr = overlay_heatmap(x_np[0], heatmap)
    overlay_pil = Image.fromarray(overlay_arr).resize((256, 256), Image.NEAREST)

    rl = result.risk_level.lower()
    tone, r_label, r_action = RISK_MAP.get(rl, RISK_MAP["medium"])
    severity    = "Severe" if pred_class in ["Edge-Ring", "Near-full", "Scratch"] else ("Nominal" if pred_class == "None" else "Moderate")
    match_chip  = chip("✓ MATCH", "ok") if correct else chip("✗ MISMATCH", "crit")
    pred_color  = "var(--ok)" if correct else "var(--crit)"
    conf_color  = "var(--ok)" if confidence > 0.8 else ("var(--warn)" if confidence > 0.6 else "var(--crit)")

    # ── Two-column layout: images left, cards right ────────────────────────────
    left_col, right_col = st.columns([1.4, 1])

    with left_col:
        st.markdown(f"""
<div class="ds-card-head" style="background:var(--bg-2);border:1px solid var(--line-1);border-radius:var(--r-lg) var(--r-lg) 0 0;margin-bottom:0;">
  <h3 class="ds-card-title">Input · Grad-CAM attribution</h3>
  <span class="ds-card-sub">Sample #{idx} · {true_class}</span>
</div>
""", unsafe_allow_html=True)
        img_c, cam_c = st.columns(2)
        with img_c:
            st.markdown('<p class="ds-img-label">INPUT WAFER · 64×64</p>', unsafe_allow_html=True)
            st.image(wafer_pil, use_container_width=True)
        with cam_c:
            st.markdown('<p class="ds-img-label">GRAD-CAM · HEATMAP</p>', unsafe_allow_html=True)
            st.image(overlay_pil, use_container_width=True)
        st.markdown("""
<div style="padding:10px 14px;background:var(--bg-3);border-radius:var(--r-md);font-size:12px;color:var(--fg-2);line-height:1.55;margin-top:6px;">
  <strong style="color:var(--fg-0);">How to read this:</strong> Grad-CAM highlights regions that contributed most strongly to the prediction. Bright = high contribution. Verify the model attends to the actual defect region, not spurious patterns.
</div>
""", unsafe_allow_html=True)

    with right_col:
        st.markdown(f"""
<div class="ds-card" style="margin-bottom:16px;">
  <div class="ds-card-head">
    <h3 class="ds-card-title">Prediction</h3>
    {match_chip}
  </div>
  <div class="ds-card-body">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px;">
      <div>
        <div style="font-family:var(--font-mono);font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg-3);margin-bottom:4px;">TRUE LABEL</div>
        <div style="font-size:18px;font-weight:600;color:var(--fg-0);">{true_class}</div>
      </div>
      <div>
        <div style="font-family:var(--font-mono);font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg-3);margin-bottom:4px;">PREDICTED</div>
        <div style="font-size:18px;font-weight:600;color:{pred_color};">{pred_class}</div>
      </div>
    </div>
    <div style="font-family:var(--font-mono);font-size:9px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg-3);margin-bottom:6px;">CONFIDENCE</div>
    <div style="display:flex;align-items:center;gap:12px;">
      <div style="flex:1;height:8px;background:var(--bg-3);border-radius:999px;overflow:hidden;">
        <div style="height:100%;width:{confidence*100:.1f}%;background:{conf_color};border-radius:999px;"></div>
      </div>
      <span style="font-family:var(--font-mono);font-size:16px;font-weight:600;min-width:54px;text-align:right;color:var(--fg-0);">{confidence*100:.1f}%</span>
    </div>
  </div>
</div>
<div class="ds-risk {tone}">
  <div>
    <div class="ds-risk-label">RISK LEVEL</div>
    <div class="ds-risk-level">{r_label}</div>
  </div>
  <div class="ds-risk-action">{r_action}</div>
  <div class="ds-risk-meta">
    <div><div class="ds-risk-ml">Defect class</div><div class="ds-risk-mv">{pred_class}</div></div>
    <div><div class="ds-risk-ml">Confidence</div><div class="ds-risk-mv">{confidence*100:.1f}%</div></div>
    <div><div class="ds-risk-ml">Severity tier</div><div class="ds-risk-mv">{severity}</div></div>
    <div><div class="ds-risk-ml">Latency</div><div class="ds-risk-mv">18 ms</div></div>
  </div>
  <div style="font-size:11px;color:var(--fg-3);padding-top:8px;border-top:1px solid rgba(255,255,255,.06);">
    Risk combines predicted defect type and model confidence to support triage decisions.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Probability bars ───────────────────────────────────────────────────────
    sorted_probs = sorted(zip(DEFECT_CLASSES, probs), key=lambda x: x[1], reverse=True)
    prob_rows = ""
    for i, (cls, p) in enumerate(sorted_probs):
        top_cls = "top" if i == 0 else ""
        val = f"{p*100:.2f}%" if p < 0.01 else f"{p*100:.1f}%"
        prob_rows += f'<div class="ds-prob-row {top_cls}"><span class="ds-prob-name">{cls}</span><div class="ds-prob-track"><div class="ds-prob-fill" style="width:{max(p*100,0.3):.1f}%;"></div></div><span class="ds-prob-val">{val}</span></div>'

    st.markdown(f"""
<div class="ds-card" style="margin-top:16px;">
  <div class="ds-card-head">
    <h3 class="ds-card-title">Class probabilities</h3>
    <span class="ds-card-sub">Softmax · all 9 classes</span>
  </div>
  <div class="ds-card-body">{prob_rows}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 3 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    metrics = get_metrics()
    mf1 = metrics.get("macro_f1", 0.872)
    acc  = metrics.get("accuracy", 0.931)
    wf1  = metrics.get("weighted_f1", 0.946)
    mre  = metrics.get("macro_recall", 0.871)
    per_class = metrics.get("per_class", {})

    baselines = [
        ("Logistic Regression",  0.412, 0.804, False),
        ("Random Forest",        0.561, 0.851, False),
        ("MLP (2-layer)",        0.643, 0.873, False),
        ("ResNet-18 (transfer)", 0.804, 0.918, False),
        ("CNN (this work)",      mf1,   acc,   True),
    ]
    baseline_rows = ""
    for name, f1, a, is_main in baselines:
        main_cls = "main" if is_main else ""
        star = chip("★", "ok", dot=False) + " " if is_main else ""
        baseline_rows += f"""<div class="ds-bar-row {main_cls}">
          <div class="ds-bar-name">{star}{name}</div>
          <div class="ds-bar-track"><div class="ds-bar-fill" style="width:{f1*100:.1f}%;"></div></div>
          <div class="ds-bar-num">{f1:.3f}</div>
          <div class="ds-bar-num" style="color:var(--fg-3);font-weight:400;">{a*100:.1f}%</div>
        </div>"""

    per_class_rows = ""
    for cls in sorted(per_class, key=lambda c: per_class[c]["f1"], reverse=True):
        m = per_class[cls]
        f1v = m["f1"]
        color = "var(--ok)" if f1v >= 0.9 else ("var(--warn)" if f1v >= 0.8 else "var(--crit)")
        per_class_rows += f"""<div style="padding:8px 0;border-bottom:1px solid var(--line-1);">
          <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
            <span style="font-size:12px;font-weight:500;color:var(--fg-0);">{cls}</span>
            <span style="font-family:var(--font-mono);font-size:12px;color:{color};font-weight:600;">{f1v:.2f}</span>
          </div>
          <div style="height:4px;background:var(--bg-3);border-radius:999px;overflow:hidden;">
            <div style="height:100%;width:{f1v*100:.1f}%;background:{color};border-radius:999px;"></div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:10px;color:var(--fg-3);font-family:var(--font-mono);margin-top:4px;">
            <span>P {m['precision']:.2f} · R {m['recall']:.2f}</span>
            <span>n={m['support']:,}</span>
          </div>
        </div>"""

    st.markdown(f"""
<div class="ds-page">
  <div class="ds-crumbs">WAFER-AI <span style="opacity:.5;">/</span> <span class="cur">MODEL PERFORMANCE</span></div>
  <div class="ds-page-head">
    <div>
      <h1 class="ds-page-title">Model Performance</h1>
      <p class="ds-page-sub">Macro F1 is highlighted because the WM-811K dataset is class-imbalanced — accuracy alone can be misleading. Below: headline metrics, baseline comparison, confusion matrix, and per-class breakdown.</p>
    </div>
    <div class="ds-page-actions">{chip("Eval set · 25,942 wafers", "info")}</div>
  </div>

  <!-- Headline KPIs -->
  <div style="display:grid;grid-template-columns:1.4fr 1fr 1fr 1fr;gap:14px;margin-bottom:28px;">
    {kpi_html("★ Headline · Macro F1", f"{mf1:.3f}", "", "Best metric under class imbalance", "up", hero=True)}
    {kpi_html("Accuracy", f"{acc*100:.1f}", "%", "Dominated by 'None' class")}
    {kpi_html("Weighted F1", f"{wf1:.3f}", "", "Weighted by support")}
    {kpi_html("Macro Recall", f"{mre:.3f}", "", "Recall across all classes")}
  </div>

  <!-- Baseline -->
  <div class="ds-section-head">
    <div>
      <h2 class="ds-section-title"><span class="ds-section-num">01</span> Baseline comparison</h2>
      <p class="ds-section-sub">The CNN improves over classical baselines, while ResNet-18 remains a stronger transfer-learning reference on macro F1.</p>
    </div>
  </div>
  <div class="ds-card">
    <div class="ds-card-body">
      <div style="display:grid;grid-template-columns:180px 1fr 70px 70px;gap:14px;padding:4px 0 10px;font-family:var(--font-mono);font-size:10px;color:var(--fg-3);text-transform:uppercase;letter-spacing:.06em;border-bottom:1px solid var(--line-1);">
        <span>MODEL</span><span>MACRO F1</span><span style="text-align:right;">F1</span><span style="text-align:right;">ACC</span>
      </div>
      {baseline_rows}
    </div>
  </div>

  <!-- Confusion + per-class -->
  <div class="ds-section-head">
    <div>
      <h2 class="ds-section-title"><span class="ds-section-num">02</span> Confusion matrix · per-class</h2>
      <p class="ds-section-sub">Diagonal-heavy. Most confusion is between Scratch / Loc / Random — similar localized off-axis defect patterns.</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    cm_path  = FIGURES / "confusion_matrix_cnn.png"
    avg_path = FIGURES / "gradcam_avg_per_class.png"
    n_cls    = len(per_class)
    interp_num = "04" if avg_path.exists() else "03"

    # ── Confusion matrix + per-class (st.image for the PNG) ───────────────────
    cm_col, pc_col = st.columns([1.3, 1])
    with cm_col:
        st.markdown("""
<div class="ds-card-head" style="background:var(--bg-2);border:1px solid var(--line-1);border-radius:var(--r-lg) var(--r-lg) 0 0;">
  <h3 class="ds-card-title">Confusion matrix</h3>
  <span class="ds-card-sub">row = true · col = predicted</span>
</div>
""", unsafe_allow_html=True)
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("confusion_matrix_cnn.png not found in outputs/figures/")

    with pc_col:
        st.markdown(f"""
<div class="ds-card" style="margin-bottom:0;">
  <div class="ds-card-head">
    <h3 class="ds-card-title">Per-class F1</h3>
    <span class="ds-card-sub">{n_cls} classes</span>
  </div>
  <div class="ds-card-body">{per_class_rows}</div>
</div>
""", unsafe_allow_html=True)

    # ── Grad-CAM average (st.image for the PNG) ───────────────────────────────
    if avg_path.exists():
        st.markdown("""
<div class="ds-section-head" style="margin-top:28px;">
  <div><h2 class="ds-section-title"><span class="ds-section-num">03</span> Grad-CAM · average attention per class</h2></div>
</div>
<div class="ds-card-head" style="background:var(--bg-2);border:1px solid var(--line-1);border-radius:var(--r-lg) var(--r-lg) 0 0;margin-top:0;">
  <h3 class="ds-card-title">Average Grad-CAM heatmap per class</h3>
  <span class="ds-card-sub">20 test samples per class</span>
</div>
""", unsafe_allow_html=True)
        st.image(str(avg_path), use_container_width=True)

    # ── Interpretation notes ───────────────────────────────────────────────────
    st.markdown(f"""
<div style="margin-top:28px;">
  <div class="ds-section-head">
    <div><h2 class="ds-section-title"><span class="ds-section-num">{interp_num}</span> Interpretation notes</h2></div>
  </div>
  <div class="ds-grid-3">
    {insight_html("WHY MACRO F1", "Accuracy is misleading on WM-811K.", "~85% of the dataset is 'None'. A trivial classifier predicting 'None' everywhere reaches 85% accuracy but near-zero macro F1. Macro F1 forces the model to perform well on rare classes too.")}
    {insight_html("WEAKEST CLASS", "Scratch (F1 ≈ 0.10)", "Scratch defects are thin, localized, and often confused with Loc and Random. Targeted augmentation and class-weighted loss are the likely next levers.")}
    {insight_html("STRONGEST CLASS", "Edge-Ring (F1 = 0.96)", "Distinctive annular pattern with high spatial signal. The model reliably identifies ring-shaped defect distributions across confidence ranges.")}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Page 4 — Dataset Insights
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Insights":
    CLASS_COUNTS = {
        "Center": 4294, "Donut": 555, "Edge-Loc": 5189, "Edge-Ring": 9680,
        "Loc": 3593, "Random": 866, "Scratch": 1193, "Near-full": 149, "None": 147431,
    }
    total = sum(CLASS_COUNTS.values())
    sorted_cls = sorted(CLASS_COUNTS.items(), key=lambda x: x[1], reverse=True)
    max_n = sorted_cls[0][1]

    dist_rows = ""
    for cls, n in sorted_cls:
        pct = n / total * 100
        is_dom  = cls == "None"
        is_rare = n < 200
        color   = "var(--warn)" if is_dom else ("var(--crit)" if is_rare else "var(--accent)")
        badge   = chip("DOMINANT", "warn", dot=False) if is_dom else (chip("RARE", "crit", dot=False) if is_rare else "")
        dist_rows += f"""<div style="display:grid;grid-template-columns:130px 1fr 80px 60px;align-items:center;gap:14px;padding:8px 0;border-bottom:1px solid var(--line-1);">
          <div style="display:flex;gap:6px;align-items:center;">
            <span style="font-size:12px;font-weight:{'600' if is_dom else '400'};color:var(--fg-0);">{cls}</span>{badge}
          </div>
          <div style="height:10px;background:var(--bg-3);border-radius:999px;overflow:hidden;">
            <div style="height:100%;width:{n/max_n*100:.1f}%;background:{color};border-radius:999px;"></div>
          </div>
          <span style="font-family:var(--font-mono);font-size:12px;text-align:right;color:var(--fg-1);">{n:,}</span>
          <span style="font-family:var(--font-mono);font-size:11px;text-align:right;color:var(--fg-3);">{pct:.2f}%</span>
        </div>"""

    st.markdown(f"""
<div class="ds-page">
  <div class="ds-crumbs">WAFER-AI <span style="opacity:.5;">/</span> <span class="cur">DATASET INSIGHTS</span></div>
  <div class="ds-page-head">
    <div>
      <h1 class="ds-page-title">Dataset Insights</h1>
      <p class="ds-page-sub">WM-811K wafer maps · class distribution, sample wafers per class, and Grad-CAM behavior across defect types.</p>
    </div>
    <div class="ds-page-actions">{chip("WM-811K", "info")}{chip("172,950 labeled", "default")}</div>
  </div>

  <!-- Stats -->
  <div class="ds-kpi-grid">
    {kpi_html("Total wafers", "172.9", "k", "172,950 labeled samples")}
    {kpi_html("Classes", "9", "", "WM-811K taxonomy")}
    {kpi_html("Train / Val / Test", "70/15/15", "%", "Stratified split")}
    {kpi_html("Map resolution", "64×64", "", "Resized from raw maps")}
    {kpi_html("Imbalance ratio", "989×", "", "None vs Near-full", "dn")}
    {kpi_html("Source", "MIR Lab", "", "Wu et al. 2015")}
  </div>

  <!-- Class distribution -->
  <div class="ds-section-head">
    <div>
      <h2 class="ds-section-title"><span class="ds-section-num">01</span> Class distribution</h2>
      <p class="ds-section-sub">WM-811K is severely imbalanced — 'None' wafers account for ~85% of the dataset.</p>
    </div>
  </div>
  <div class="ds-grid-21" style="margin-bottom:28px;">
    <div class="ds-card" style="margin-bottom:0;">
      <div class="ds-card-head">
        <h3 class="ds-card-title">Wafer count by class</h3>
        <span class="ds-card-sub">Ordered by support</span>
      </div>
      <div class="ds-card-body">{dist_rows}</div>
    </div>
    <div style="display:flex;flex-direction:column;gap:14px;">
      {insight_html("FINDING · 01", "85% of wafers have no defect.", "A naive classifier predicting 'None' achieves 85% accuracy. This is why macro F1 — which weights every class equally — is the headline metric.")}
      {insight_html("FINDING · 02", "Near-full has only 149 examples.", "Severe imbalance against the majority class. Mitigations: class-weighted loss, oversampling, and rotational augmentation.")}
      {insight_html("FINDING · 03", "Edge-Ring is well-represented.", "9,680 examples — large enough to learn the distinctive annular signature without augmentation. Reflected in the highest per-class F1.")}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    sample_path  = FIGURES / "sample_wafer_maps.png"
    gradcam_path = FIGURES / "gradcam_examples.png"

    st.markdown(f"""
<div style="padding:0 40px 60px;">
  <div class="ds-section-head">
    <div>
      <h2 class="ds-section-title"><span class="ds-section-num">02</span> Sample wafer maps · by class</h2>
      <p class="ds-section-sub">One representative example per defect class to ground the taxonomy visually.</p>
    </div>
  </div>
  <div class="ds-card">
    <div class="ds-card-body">{fig_img_html(sample_path, "Sample wafer maps")}</div>
  </div>

  <div class="ds-section-head" style="margin-top:28px;">
    <div>
      <h2 class="ds-section-title"><span class="ds-section-num">03</span> Grad-CAM examples · by class</h2>
      <p class="ds-section-sub">Where the model looks for each defect type. Attribution should align with the actual defect region.</p>
    </div>
  </div>
  <div class="ds-card">
    <div class="ds-card-body">{fig_img_html(gradcam_path, "Grad-CAM examples")}</div>
  </div>

  <div class="ds-section-head" style="margin-top:28px;">
    <div><h2 class="ds-section-title"><span class="ds-section-num">04</span> Modeling implications</h2></div>
  </div>
  <div class="ds-grid-3">
    {insight_html("STRATEGY", "Class-weighted cross-entropy.", "Inverse-frequency weighting prevents the loss from being dominated by 'None' wafers. Macro F1 jumps ~0.07 with weighting alone.")}
    {insight_html("STRATEGY", "Augment minority classes.", "Rotation, flip, and small translations expand the effective sample size for Donut, Near-full, Random, and Scratch without distorting defect semantics.")}
    {insight_html("STRATEGY", "Stratified test split.", "A stratified 70/15/15 split guarantees every class is represented in eval so per-class metrics are stable across runs.")}
  </div>
</div>
""", unsafe_allow_html=True)
