"""views/settings.py — Configuration & system info."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import config, utils
from components.ui import (
    page_header, divider, section_label, footer, badge,
)


def _size(path):
    if not os.path.isfile(path):
        return "—"
    s = os.path.getsize(path)
    if s >= 1024**2: return f"{s/1024**2:.1f} MB"
    if s >= 1024:    return f"{s/1024:.1f} KB"
    return f"{s} B"


def render():
    page_header(
        "Pengaturan",
        "Periksa artefak model, konfigurasi pelatihan, dan status sistem.",
        eyebrow="Sistem",
    )

    # ─── Model overview ─────────────────────────────────────────────────────
    section_label("Model")

    model_classes = "—"
    if os.path.isfile(config.MODEL_BEST_PATH):
        try:
            from ultralytics import YOLO
            m = YOLO(config.MODEL_BEST_PATH)
            model_classes = str(len(m.names))
        except Exception:
            pass

    cards = [
        ("Arsitektur", f"YOLOv8{config.MODEL_SIZE}", "Backbone"),
        ("Kelas", model_classes, "Varian aksara"),
        ("Bobot", _size(config.MODEL_BEST_PATH), "ukuran best.pt"),
    ]
    cols = st.columns(3, gap="medium")
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="ds-stat" style="--accent-color:#7C5CFF">
              <div class="ds-stat-row">
                <span class="ds-stat-label">{label}</span>
              </div>
              <div class="ds-stat-value" style="font-size:22px">{value}</div>
              <div class="ds-stat-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # ─── Training config ────────────────────────────────────────────────────
    section_label("Konfigurasi pelatihan")
    params = [
        ("Epoch", config.EPOCHS),
        ("Ukuran gambar", f"{config.IMG_SIZE}px"),
        ("Ukuran batch", config.BATCH_SIZE),
        ("Patience", config.PATIENCE),
        ("Learning rate", config.LR0),
        ("Perangkat", config.DEVICE),
        ("Batas kepercayaan", config.CONF_THRESHOLD),
        ("Batas IoU", config.IOU_THRESHOLD),
        ("Pekerja (Workers)", config.WORKERS),
    ]

    cols = st.columns(3, gap="medium")
    for i, (label, value) in enumerate(params):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="ds-list-item">
              <span style="flex:1;font-size:12.5px;color:var(--text-tertiary)">{label}</span>
              <span class="ds-mono" style="color:var(--text-primary);font-weight:500">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    divider()

    # ─── File status ────────────────────────────────────────────────────────
    section_label("Status file")
    files = [
        ("best.pt", config.MODEL_BEST_PATH, "Bobot YOLOv8 yang dilatih"),
        ("last.pt", config.MODEL_LAST_PATH, "Poin pemeriksaan terakhir"),
        ("label_map.json", config.LABEL_MAP_PATH, "Pemetaan indeks kelas"),
        ("metrics.json", config.METRICS_PATH, "Metrik performa"),
        ("prediction_log.json", config.PREDICT_LOG, "Riwayat deteksi"),
        ("data.yaml", config.DATA_YAML, "Konfigurasi dataset"),
    ]

    for name, path, desc in files:
        exists = os.path.isfile(path)
        size = _size(path)
        dot_class = "ds-status-dot-green" if exists else "ds-status-dot-gray"
        status_label = "tersedia" if exists else "hilang"
        status_variant = "green" if exists else "neutral"
        st.markdown(f"""
        <div class="ds-list-item">
          <span class="ds-status-dot {dot_class}"
                style="margin-right:4px"></span>
          <div style="flex:1;min-width:0">
            <div style="font-size:13px;color:var(--text-primary);font-weight:500;
                        font-family:'JetBrains Mono',monospace">{name}</div>
            <div style="font-size:11.5px;color:var(--text-tertiary)">{desc}</div>
          </div>
          {badge(status_label, status_variant)}
          <span class="ds-mono" style="color:var(--text-tertiary);min-width:60px;
                       text-align:right">{size}</span>
        </div>
        """, unsafe_allow_html=True)

    # ─── Class catalog ──────────────────────────────────────────────────────
    if os.path.isfile(config.MODEL_BEST_PATH):
        divider()
        section_label("Katalog kelas")
        with st.expander(f"Lihat semua {model_classes} kelas"):
            try:
                from ultralytics import YOLO
                m = YOLO(config.MODEL_BEST_PATH)
                names = m.names
                html = '<div style="display:flex;flex-wrap:wrap;gap:6px;padding:4px 0">'
                for idx, name in sorted(names.items()):
                    html += f"""
                    <span style="display:inline-flex;align-items:center;gap:6px;
                                 padding:5px 10px;background:var(--bg-elevated);
                                 border:1px solid var(--border-default);
                                 border-radius:var(--r-sm);font-size:12px;
                                 color:var(--text-secondary);transition:all .15s">
                      <span class="ds-mono" style="color:var(--brand);font-size:11px;
                                   font-weight:600">{idx:03d}</span>
                      <span style="color:var(--text-primary);font-weight:500">{name}</span>
                    </span>
                    """
                html += "</div>"
                st.markdown(html, unsafe_allow_html=True)
            except Exception as e:
                st.error(str(e))

    footer()
