"""
components/ui.py — Premium design system components.
Inspired by Stripe, Linear, and Notion.
"""

import os, sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Style Loading ────────────────────────────────────────────────────────────

def load_css():
    css_path = Path(__file__).parent.parent / "styles" / "theme.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


COMPONENT_CSS = """
<style>
/* ─── Page Header ─────────────────────────────────────────────────────── */
.ds-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
  padding: 0 0 24px;
  border-bottom: 1px solid var(--border-default);
  margin-bottom: 28px;
}
.ds-header-text { flex: 1; min-width: 0; }
.ds-header-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--brand);
  margin-bottom: 8px;
}
.ds-header-eyebrow::before {
  content: '';
  width: 4px; height: 4px;
  border-radius: 50%;
  background: var(--brand);
  box-shadow: 0 0 8px var(--brand);
}
.ds-header-title {
  font-size: 28px;
  font-weight: 700;
  letter-spacing: -0.025em;
  color: var(--text-primary);
  line-height: 1.15;
  margin: 0;
}
.ds-header-sub {
  font-size: 14px;
  color: var(--text-tertiary);
  line-height: 1.5;
  margin-top: 6px;
  max-width: 640px;
}

/* ─── Section heading ─────────────────────────────────────────────────── */
.ds-section-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-quaternary);
  margin: 0 0 14px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.ds-section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border-default);
}

/* ─── Card ─────────────────────────────────────────────────────────────── */
.ds-card {
  position: relative;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--r-lg);
  padding: 20px;
  transition: all var(--duration-base) var(--ease-out);
  overflow: hidden;
}
.ds-card:hover {
  border-color: var(--border-strong);
  background: var(--bg-elevated);
  transform: translateY(-1px);
  box-shadow: var(--shadow-sm);
}
.ds-card-interactive { cursor: pointer; }
.ds-card-interactive:hover { border-color: var(--brand-border); }

/* ─── Stat / Metric Card ──────────────────────────────────────────────── */
.ds-stat {
  position: relative;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--r-lg);
  padding: 18px 20px;
  transition: all var(--duration-base) var(--ease-out);
  overflow: hidden;
}
.ds-stat::after {
  content: '';
  position: absolute;
  inset: 0 0 auto 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-color, var(--brand)), transparent);
  opacity: 0.6;
}
.ds-stat:hover {
  border-color: var(--border-strong);
  transform: translateY(-1px);
}
.ds-stat-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.ds-stat-label {
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0.02em;
  color: var(--text-tertiary);
  text-transform: uppercase;
}
.ds-stat-icon {
  width: 24px; height: 24px;
  border-radius: 6px;
  background: var(--accent-soft, var(--brand-soft));
  color: var(--accent-color, var(--brand));
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 13px;
}
.ds-stat-value {
  font-size: 26px;
  font-weight: 700;
  letter-spacing: -0.025em;
  color: var(--text-primary);
  line-height: 1.1;
  font-variant-numeric: tabular-nums;
}
.ds-stat-sub {
  font-size: 12px;
  color: var(--text-tertiary);
  margin-top: 4px;
}
.ds-stat-trend {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 11px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 4px;
  background: var(--brand-soft);
  color: var(--brand);
  margin-left: 8px;
}

/* ─── Divider ──────────────────────────────────────────────────────────── */
.ds-divider {
  height: 1px;
  background: var(--border-default);
  margin: 28px 0;
  border: none;
}
.ds-divider-spacer { height: 28px; }

/* ─── Empty State ──────────────────────────────────────────────────────── */
.ds-empty {
  text-align: center;
  padding: 56px 24px;
  border: 1px dashed var(--border-strong);
  border-radius: var(--r-xl);
  background: var(--bg-surface);
  margin: 16px 0;
}
.ds-empty-icon {
  width: 56px; height: 56px;
  margin: 0 auto 16px;
  border-radius: 14px;
  background: var(--brand-soft);
  border: 1px solid var(--brand-border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: var(--brand);
}
.ds-empty-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
  letter-spacing: -0.01em;
}
.ds-empty-desc {
  font-size: 13px;
  color: var(--text-tertiary);
  max-width: 360px;
  margin: 0 auto;
  line-height: 1.5;
}

/* ─── Badge ────────────────────────────────────────────────────────────── */
.ds-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 8px;
  border-radius: 5px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: -0.005em;
  line-height: 1.5;
  font-variant-numeric: tabular-nums;
}
.ds-badge-dot {
  width: 5px; height: 5px;
  border-radius: 50%;
  background: currentColor;
}
.ds-badge-brand   { background: rgba(124, 92, 255, 0.10); color: #A18BFF; border: 1px solid rgba(124, 92, 255, 0.20); }
.ds-badge-teal    { background: rgba(0, 212, 200, 0.10);  color: #34E0D6; border: 1px solid rgba(0, 212, 200, 0.20); }
.ds-badge-pink    { background: rgba(255, 107, 157, 0.10); color: #FF8AB4; border: 1px solid rgba(255, 107, 157, 0.20); }
.ds-badge-amber   { background: rgba(255, 181, 71, 0.10);  color: #FFC872; border: 1px solid rgba(255, 181, 71, 0.20); }
.ds-badge-green   { background: rgba(52, 211, 153, 0.10);  color: #5BE5B5; border: 1px solid rgba(52, 211, 153, 0.20); }
.ds-badge-red     { background: rgba(248, 113, 113, 0.10); color: #FF9999; border: 1px solid rgba(248, 113, 113, 0.20); }
.ds-badge-blue    { background: rgba(96, 165, 250, 0.10);  color: #82B7FF; border: 1px solid rgba(96, 165, 250, 0.20); }
.ds-badge-neutral { background: rgba(255, 255, 255, 0.04); color: #B4B4C2; border: 1px solid rgba(255, 255, 255, 0.08); }

/* Legacy alias */
.ak-badge-purple { background: rgba(124, 92, 255, 0.10); color: #A18BFF; border: 1px solid rgba(124, 92, 255, 0.20); padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: 600; }
.ak-badge-teal   { background: rgba(0, 212, 200, 0.10); color: #34E0D6; border: 1px solid rgba(0, 212, 200, 0.20); padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: 600; }
.ak-badge-pink   { background: rgba(255, 107, 157, 0.10); color: #FF8AB4; border: 1px solid rgba(255, 107, 157, 0.20); padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: 600; }
.ak-badge-orange { background: rgba(255, 181, 71, 0.10); color: #FFC872; border: 1px solid rgba(255, 181, 71, 0.20); padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: 600; }
.ak-badge-green  { background: rgba(52, 211, 153, 0.10); color: #5BE5B5; border: 1px solid rgba(52, 211, 153, 0.20); padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: 600; }

/* ─── Info Banner ──────────────────────────────────────────────────────── */
.ds-banner {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 12px 14px;
  background: var(--brand-soft);
  border: 1px solid var(--brand-border);
  border-radius: var(--r-md);
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.5;
}
.ds-banner-icon {
  flex-shrink: 0;
  width: 18px; height: 18px;
  color: var(--brand);
  margin-top: 1px;
}

/* ─── Detection / List Item ────────────────────────────────────────────── */
.ds-list-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--r-md);
  transition: all var(--duration-fast) var(--ease-out);
  margin-bottom: 6px;
}
.ds-list-item:hover {
  background: var(--bg-elevated);
  border-color: var(--border-strong);
}

/* ─── Brand chip / Logo block ──────────────────────────────────────────── */
.ds-logo {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 4px 20px 14px 20px;
}
.ds-logo-mark {
  width: 32px; height: 32px;
  border-radius: 8px;
  background: var(--gradient-brand);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  color: white;
  font-weight: 700;
  box-shadow: 0 4px 12px var(--brand-glow);
  position: relative;
  overflow: hidden;
}
.ds-logo-mark::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, rgba(255,255,255,0.20), transparent 60%);
}
.ds-logo-text {
  display: flex;
  flex-direction: column;
  line-height: 1.1;
}
.ds-logo-name {
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: var(--text-primary);
}
.ds-logo-version {
  font-size: 10px;
  color: var(--text-quaternary);
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
  margin-top: 2px;
}

/* ─── Sidebar utility ──────────────────────────────────────────────────── */
.ds-side-section {
  padding: 0 20px;
  margin-top: 16px;
}
.ds-side-label {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--text-quaternary);
  margin: 0 0 8px;
}
.ds-side-status {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 10px;
  border-radius: var(--r-sm);
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  margin-bottom: 4px;
  font-size: 12px;
}
.ds-side-status-label { color: var(--text-tertiary); }
.ds-side-status-value {
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  display: inline-flex; align-items: center; gap: 6px;
}
.ds-status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  display: inline-block;
}
.ds-status-dot-green { background: var(--green); box-shadow: 0 0 6px var(--green); }
.ds-status-dot-red   { background: var(--red);   box-shadow: 0 0 6px var(--red); }
.ds-status-dot-gray  { background: var(--text-quaternary); }

/* ─── Mono ─────────────────────────────────────────────────────────────── */
.ds-mono {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

/* ─── KBD ──────────────────────────────────────────────────────────────── */
.ds-kbd {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  padding: 2px 5px;
  border-radius: 3px;
  background: var(--bg-elevated);
  border: 1px solid var(--border-strong);
  color: var(--text-secondary);
  line-height: 1.4;
}

/* ─── Hero ─────────────────────────────────────────────────────────────── */
.ds-hero {
  position: relative;
  padding: 4px 0 28px;
}
.ds-hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 5px 10px 5px 8px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: 100px;
  font-size: 11px;
  font-weight: 500;
  color: var(--text-secondary);
  margin-bottom: 24px;
}
.ds-hero-eyebrow-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 6px var(--green);
}
.ds-hero-title {
  font-size: 48px;
  font-weight: 700;
  letter-spacing: -0.04em;
  line-height: 1.05;
  color: var(--text-primary);
  margin: 0 0 16px;
  max-width: 640px;
}
.ds-hero-title .ds-gradient {
  background: linear-gradient(135deg, #7C5CFF 0%, #5CC4FF 50%, #00D4C8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.ds-hero-sub {
  font-size: 17px;
  color: var(--text-tertiary);
  line-height: 1.55;
  max-width: 560px;
  margin: 0 0 28px;
}
.ds-hero-tags { display: flex; gap: 8px; flex-wrap: wrap; }

/* ─── Feature card ─────────────────────────────────────────────────────── */
.ds-feature {
  position: relative;
  padding: 20px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--r-lg);
  height: 100%;
  transition: all var(--duration-base) var(--ease-out);
}
.ds-feature:hover {
  background: var(--bg-elevated);
  border-color: var(--border-strong);
  transform: translateY(-2px);
}
.ds-feature-icon {
  width: 36px; height: 36px;
  border-radius: 9px;
  background: var(--brand-soft);
  border: 1px solid var(--brand-border);
  color: var(--brand);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  margin-bottom: 14px;
}
.ds-feature-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
  letter-spacing: -0.01em;
}
.ds-feature-desc {
  font-size: 13px;
  color: var(--text-tertiary);
  line-height: 1.5;
}

/* ─── Step / Numbered list ─────────────────────────────────────────────── */
.ds-step {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 16px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--r-md);
  margin-bottom: 6px;
  transition: all var(--duration-fast) var(--ease-out);
}
.ds-step:hover {
  background: var(--bg-elevated);
  border-color: var(--border-strong);
}
.ds-step-num {
  flex-shrink: 0;
  width: 28px; height: 28px;
  border-radius: 8px;
  background: var(--brand-soft);
  border: 1px solid var(--brand-border);
  color: var(--brand);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  font-weight: 600;
}
.ds-step-text {
  font-size: 13px;
  color: var(--text-secondary);
  line-height: 1.4;
}

/* ─── Detection result hero ────────────────────────────────────────────── */
.ds-result-card {
  text-align: center;
  padding: 28px 20px;
  background: linear-gradient(180deg, var(--bg-surface), var(--bg-elevated));
  border: 1px solid var(--border-default);
  border-radius: var(--r-xl);
  position: relative;
  overflow: hidden;
}
.ds-result-card::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(circle at 50% 0%, var(--brand-soft), transparent 60%);
  pointer-events: none;
}
.ds-result-label {
  position: relative;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--text-tertiary);
  margin-bottom: 12px;
}
.ds-result-value {
  position: relative;
  font-size: 56px;
  font-weight: 700;
  letter-spacing: -0.03em;
  background: linear-gradient(135deg, #FAFAFA, #B4B4C2);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
}
.ds-result-meta {
  position: relative;
  font-size: 12px;
  color: var(--text-tertiary);
  margin-top: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
}

/* ─── Footer ───────────────────────────────────────────────────────────── */
.ds-footer {
  text-align: center;
  padding: 32px 0 0;
  margin-top: 48px;
  border-top: 1px solid var(--border-default);
  font-size: 12px;
  color: var(--text-quaternary);
}
</style>
"""


def inject_styles():
    load_css()
    st.markdown(COMPONENT_CSS, unsafe_allow_html=True)


# ─── Component API ────────────────────────────────────────────────────────────

def page_header(title: str, subtitle: str = "", eyebrow: str = ""):
    """Page header with optional eyebrow tag and subtitle."""
    eyebrow_html = f'<div class="ds-header-eyebrow">{eyebrow}</div>' if eyebrow else ''
    sub_html = f'<div class="ds-header-sub">{subtitle}</div>' if subtitle else ''
    st.markdown(f"""
    <div class="ds-header">
      <div class="ds-header-text">
        {eyebrow_html}
        <h1 class="ds-header-title">{title}</h1>
        {sub_html}
      </div>
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str):
    """Small uppercase section divider with line."""
    st.markdown(f'<div class="ds-section-label">{text}</div>', unsafe_allow_html=True)


def stat_card(label: str, value: str, sub: str = "", color: str = "#7C5CFF", icon: str = ""):
    """Premium metric card with optional icon and accent color."""
    soft = _hex_to_rgba(color, 0.12)
    icon_html = f'<div class="ds-stat-icon">{icon}</div>' if icon else ''
    sub_html = f'<div class="ds-stat-sub">{sub}</div>' if sub else ''
    st.markdown(f"""
    <div class="ds-stat" style="--accent-color:{color};--accent-soft:{soft}">
      <div class="ds-stat-row">
        <span class="ds-stat-label">{label}</span>
        {icon_html}
      </div>
      <div class="ds-stat-value">{value}</div>
      {sub_html}
    </div>
    """, unsafe_allow_html=True)


def divider(spacing: str = "default"):
    """Horizontal rule with consistent spacing."""
    if spacing == "spacer":
        st.markdown('<div class="ds-divider-spacer"></div>', unsafe_allow_html=True)
    else:
        st.markdown('<hr class="ds-divider"/>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, desc: str):
    """Empty state with icon, title, and description."""
    st.markdown(f"""
    <div class="ds-empty">
      <div class="ds-empty-icon">{icon}</div>
      <div class="ds-empty-title">{title}</div>
      <div class="ds-empty-desc">{desc}</div>
    </div>
    """, unsafe_allow_html=True)


def info_banner(text: str, icon: str = "ⓘ"):
    """Subtle info banner."""
    st.markdown(f"""
    <div class="ds-banner">
      <span class="ds-banner-icon">{icon}</span>
      <span>{text}</span>
    </div>
    """, unsafe_allow_html=True)


def badge(text: str, variant: str = "neutral", dot: bool = False):
    """Inline badge (returns HTML string)."""
    dot_html = '<span class="ds-badge-dot"></span>' if dot else ''
    return f'<span class="ds-badge ds-badge-{variant}">{dot_html}{text}</span>'


def feature_card(icon: str, title: str, desc: str):
    """Feature card with icon, title, description (returns HTML)."""
    return f"""
    <div class="ds-feature">
      <div class="ds-feature-icon">{icon}</div>
      <div class="ds-feature-title">{title}</div>
      <div class="ds-feature-desc">{desc}</div>
    </div>
    """


def step_item(num: str, text: str):
    """Numbered step row (returns HTML)."""
    return f"""
    <div class="ds-step">
      <div class="ds-step-num">{num}</div>
      <div class="ds-step-text">{text}</div>
    </div>
    """


def metric_row(items: list):
    """items = [(label, value, color), ...]"""
    cols = st.columns(len(items))
    for col, (label, value, color) in zip(cols, items):
        with col:
            stat_card(label, value, color=color)


def footer(version: str = None, tagline: str = None):
    """App footer."""
    v = version or config.APP_VERSION
    t = tagline or config.APP_TAGLINE
    st.markdown(
        f'<div class="ds-footer">{config.APP_NAME} <span class="ds-mono">v{v}</span> · {t}</div>',
        unsafe_allow_html=True,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r = int(h[0:2], 16); g = int(h[2:4], 16); b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
