"""
app.py — AksaraDetect
Premium dashboard for Aksara Ulu Rejang detection.
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

st.set_page_config(
    page_title="AksaraDetect — Deteksi Aksara Ulu Rejang",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "AksaraDetect — Deteksi Aksara Ulu Rejang menggunakan YOLOv8."
    },
)

from components.ui import inject_styles
import config, utils

utils.ensure_dirs()
inject_styles()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Brand mark
    st.markdown(f"""
    <div class="ds-logo">
      <div class="ds-logo-mark">◈</div>
      <div class="ds-logo-text">
        <span class="ds-logo-name">AksaraDetect</span>
        <span class="ds-logo-version">v{config.APP_VERSION}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Section: Navigation
    st.markdown("""
    <div class="ds-side-section" style="margin-bottom:8px">
      <div class="ds-side-label">Navigasi</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        [
            "◈  Beranda",
            "⌕  Deteksi",
            "📊  Analitik",
            "🧠  Pelatihan",
            "◷  Riwayat",
            "⚙  Pengaturan",
        ],
        label_visibility="collapsed",
    )

    # Section: Status
    classifier_path = os.path.join(config.BASE_DIR, "best_classifier.pth")
    yolo_path = config.MODEL_BEST_PATH
    model_ok = os.path.isfile(classifier_path) or os.path.isfile(yolo_path)
    log = utils.load_prediction_log()

    dot_class = "ds-status-dot-green" if model_ok else "ds-status-dot-red"
    status_text = "Siap" if model_ok else "Hilang"
    status_color = "var(--green)" if model_ok else "var(--red)"

    st.markdown(f"""
    <div class="ds-side-section">
      <div class="ds-side-label">Sistem</div>
      <div class="ds-side-status">
        <span class="ds-side-status-label">Model</span>
        <span class="ds-side-status-value" style="color:{status_color}">
          <span class="ds-status-dot {dot_class}"></span>{status_text}
        </span>
      </div>
      <div class="ds-side-status">
        <span class="ds-side-status-label">Total Deteksi</span>
        <span class="ds-side-status-value" style="color:var(--brand)">{len(log)}</span>
      </div>
      <div class="ds-side-status">
        <span class="ds-side-status-label">Kelas Aksara</span>
        <span class="ds-side-status-value" style="color:var(--text-secondary)">253</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Section: Team
    st.markdown("""
    <div class="ds-side-section" style="margin-top:24px;margin-bottom:24px">
      <div class="ds-side-label">Kelompok 8</div>
      <div style="padding:12px;background:var(--bg-elevated);border:1px solid var(--border-default);border-radius:var(--r-md);font-size:11.5px;color:var(--text-tertiary);line-height:1.7">
        <div style="display:flex;justify-content:space-between;gap:8px">
          <span>Pebi Heriansyah</span>
          <span class="ds-mono" style="color:var(--text-quaternary);font-size:10.5px">G1A023003</span>
        </div>
        <div style="display:flex;justify-content:space-between;gap:8px">
          <span>Reffki Andrea P.</span>
          <span class="ds-mono" style="color:var(--text-quaternary);font-size:10.5px">G1A023039</span>
        </div>
        <div style="display:flex;justify-content:space-between;gap:8px">
          <span>M. Farhan Dzakki</span>
          <span class="ds-mono" style="color:var(--text-quaternary);font-size:10.5px">G1A023041</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Routing ──────────────────────────────────────────────────────────────────
if   "Beranda"    in page: from views.home      import render; render()
elif "Deteksi"    in page: from views.detect    import render; render()
elif "Analitik"   in page: from views.analytics import render; render()
elif "Pelatihan"  in page: from views.training  import render; render()
elif "Riwayat"    in page: from views.history   import render; render()
elif "Pengaturan" in page: from views.settings  import render; render()
