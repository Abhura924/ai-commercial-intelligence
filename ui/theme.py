"""Shared UI theme + reusable executive components (board-grade look)."""
import numpy as np
import streamlit as st

TEAL = "#1F6F6B"; BRICK = "#B1462E"; AMBER = "#C08A2E"; PURPLE = "#5B4B8A"
INK = "#1B1F2A"; INK_PANEL = "#16213E"; INK_PANEL_2 = "#1F2D4D"
MUTED = "#6B7178"; PAPER = "#F4F5F1"; LINE = "#E4E1D6"


def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] {{ font-family:'IBM Plex Sans', sans-serif; }}
    .stApp {{ background:{PAPER}; }}
    header[data-testid="stHeader"] {{ background:transparent; height:0; }}
    #MainMenu, [data-testid="stToolbar"], footer {{ visibility:hidden; }}
    .block-container {{ padding-top:1.4rem; padding-bottom:3rem; max-width:1200px; }}
    h1 {{ font-family:'Source Serif 4',serif !important; font-weight:600 !important; color:{INK}; font-size:27px !important; margin-bottom:2px; }}
    h2,h3 {{ font-family:'Source Serif 4',serif !important; color:{INK}; font-weight:600 !important; }}
    section[data-testid="stSidebar"] {{ background:{INK_PANEL} !important; }}
    section[data-testid="stSidebar"] * {{ color:#C7CEDD; }}
    section[data-testid="stSidebar"] > div {{ padding-top:24px; }}
    .mer-brand {{ font-family:'Source Serif 4',serif; font-size:22px; font-weight:600; color:#FFF !important; padding:0 8px 2px; line-height:1.15; }}
    .mer-brand-sub {{ font-size:12px; color:#9AA6C4 !important; padding:0 8px 16px; border-bottom:1px solid rgba(255,255,255,.12); margin-bottom:12px; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] {{ gap:3px; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{ display:flex; align-items:center; padding:9px 12px; border-radius:6px; cursor:pointer; margin:0; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background:{INK_PANEL_2}; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display:none; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {{ font-size:14px !important; color:#C7CEDD !important; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{ background:{TEAL}; }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {{ color:#FFF !important; font-weight:500; }}
    .mer-side-foot {{ font-size:11px; color:#7C88A6 !important; padding:16px 8px 0; line-height:1.5; }}
    section[data-testid="stSidebar"] .stAlert {{ background:{INK_PANEL_2}; border:none; border-radius:6px; }}
    section[data-testid="stSidebar"] .stAlert p {{ color:#C7CEDD !important; font-size:12px; }}
    .mer-topbar {{ display:flex; align-items:center; gap:12px; margin-bottom:14px; }}
    .mer-pill {{ display:inline-block; background:#E4EEEC; color:{TEAL} !important; font-size:12px; padding:5px 13px; border-radius:16px; font-weight:600; }}
    .mer-pill-amber {{ background:#F3EBD9; color:{AMBER} !important; }}
    .mer-kpi {{ background:#FFF; border:1px solid {LINE}; border-radius:6px; padding:15px 17px; height:100%; }}
    .mer-kpi-label {{ font-size:12.5px; color:{MUTED}; margin-bottom:7px; letter-spacing:.04em; text-transform:uppercase; }}
    .mer-kpi-value {{ font-family:'IBM Plex Mono',monospace; font-size:24px; font-weight:500; color:{INK}; line-height:1.1; }}
    .mer-kpi-sub {{ font-family:'IBM Plex Mono',monospace; font-size:12px; color:{MUTED}; margin-top:3px; }}
    .mer-kpi-delta {{ font-family:'IBM Plex Mono',monospace; font-size:12px; margin-top:8px; }}
    .mer-up {{ color:{TEAL}; }} .mer-down {{ color:{BRICK}; }} .mer-flat {{ color:{MUTED}; }}
    .mer-delta-note {{ color:{MUTED}; }}
    .mer-panel-title {{ font-family:'Source Serif 4',serif; font-size:18px; font-weight:600; color:{INK}; margin-bottom:1px; }}
    .mer-panel-note {{ font-size:13px; color:{MUTED}; margin-bottom:4px; }}
    .mer-card {{ background:#FFF; border:1px solid {LINE}; border-radius:6px; padding:16px 18px; }}
    .mer-what {{ border-left:3px solid {TEAL}; padding-left:12px; }}
    .mer-why {{ border-left:3px solid {AMBER}; padding-left:12px; }}
    .mer-sowhat {{ border-left:3px solid {BRICK}; padding-left:12px; }}
    .mer-understood {{ background:#EEF4F3; border:1px solid #CFE0DD; border-radius:8px; padding:14px 18px; }}
    .mer-badge {{ display:inline-block; background:#E4EEEC; color:{TEAL} !important; font-size:12px; padding:3px 10px; border-radius:12px; font-weight:600; }}
    .mer-note {{ color:{MUTED}; font-size:13px; }}
    div[data-baseweb="select"] > div {{ border-color:{LINE}; border-radius:5px; }}
    </style>""", unsafe_allow_html=True)


def kpi_card(label, value, delta=None, tone="flat", note=None, delta2=None, tone2=None, sub=None):
    arrow = {"up":"▲","down":"▼","flat":"■"}.get(tone,"■")
    cls = {"up":"mer-up","down":"mer-down","flat":"mer-flat"}.get(tone,"mer-flat")
    sub_html = f'<div class="mer-kpi-sub">{sub}</div>' if sub else ""
    d1 = ""
    if delta is not None:
        n = f' <span class="mer-delta-note">{note}</span>' if note else ""
        d1 = f'<div class="mer-kpi-delta {cls}">{arrow} {delta}{n}</div>'
    d2 = ""
    if delta2 is not None:
        a2 = {"up":"▲","down":"▼","flat":"■"}.get(tone2,"■")
        c2 = {"up":"mer-up","down":"mer-down","flat":"mer-flat"}.get(tone2,"mer-flat")
        d2 = f'<div class="mer-kpi-delta {c2}">{a2} {delta2}</div>'
    return (f'<div class="mer-kpi"><div class="mer-kpi-label">{label}</div>'
            f'<div class="mer-kpi-value">{value}</div>{sub_html}{d1}{d2}</div>')


def panel_header(title, note=None):
    n = f'<div class="mer-panel-note">{note}</div>' if note else ""
    st.markdown(f'<div class="mer-panel-title">{title}</div>{n}', unsafe_allow_html=True)


def fmt_m(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    a = abs(v); sign = "-" if v < 0 else ""
    if a >= 1e6: return f"{sign}£{a/1e6:.1f}m"
    if a >= 1e3: return f"{sign}£{a/1e3:.0f}k"
    return f"{sign}£{a:,.0f}"

def fmt_signed_m(v):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return ("+" if v >= 0 else "") + fmt_m(v)

def fmt_pct(v, dp=1):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v*100:+.{dp}f}%"

def fmt_pp(v, dp=2):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    return f"{v*100:+.{dp}f}pp"
