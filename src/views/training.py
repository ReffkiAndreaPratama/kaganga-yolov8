"""views/training.py — Training results viewer."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config, utils
from components.ui import (
    page_header, divider, stat_card, empty_state, badge, section_label,
)


CHART_BG = "#0F0F16"
CHART_OUTER = "#0A0A0F"
TEXT_LIGHT = "#FAFAFA"
TEXT_DIM = "#7A7A8C"
ACCENTS = ["#7C5CFF", "#00D4C8", "#FF6B9D", "#FFB547", "#34D399", "#60A5FA"]


def _style_axes(ax, title=""):
    ax.set_facecolor(CHART_BG)
    if title:
        ax.set_title(title, color=TEXT_LIGHT, fontsize=12, fontweight=600,
                     loc="left", pad=14)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(alpha=0.06, color="#FFFFFF", linewidth=0.6)


def render():
    page_header(
        "Pelatihan",
        "Kurva loss, evolusi metrik, dan konfigurasi untuk setiap proses.",
        eyebrow="Riwayat pelatihan",
    )

    csv_path = os.path.join(config.RESULT_DIR, "results.csv")
    alt_path = os.path.join(config.RUNS_DIR, "aksara_ulu", "results.csv")
    if not os.path.isfile(csv_path) and os.path.isfile(alt_path):
        csv_path = alt_path

    if not os.path.isfile(csv_path):
        empty_state(
            "◫",
            "Tidak ada data pelatihan",
            "Unggah results.csv dari sesi pelatihan Colab Anda di bawah ini.",
        )
        divider()
        section_label("Unggah results.csv")
        uploaded = st.file_uploader(
            "Pilih file results.csv",
            type=["csv"],
            key="upload_csv",
            label_visibility="collapsed",
        )
        if uploaded:
            os.makedirs(config.RESULT_DIR, exist_ok=True)
            save_to = os.path.join(config.RESULT_DIR, "results.csv")
            with open(save_to, "wb") as f:
                f.write(uploaded.getvalue())
            st.success("File disimpan. Menyegarkan...")
            st.rerun()
        return

    # Load data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    epochs = len(df)

    metrics = utils.load_metrics()
    yolo_m = metrics.get("yolov8", metrics)
    map50 = yolo_m.get("metrics/mAP50(B)", yolo_m.get("metrics/mAP50", None))

    # ── Stats
    section_label("Ringkasan proses")
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Epoch", str(epochs), "Selesai", "#7C5CFF", icon="◔")
    with c2: stat_card("Arsitektur", f"YOLOv8{config.MODEL_SIZE}",
                       "Jaringan", "#00D4C8", icon="◑")
    with c3: stat_card("mAP Terbaik", f"{map50*100:.1f}%" if map50 else "—",
                       "Skor puncak", "#FF6B9D", icon="◓")
    with c4: stat_card("Ukuran gambar", f"{config.IMG_SIZE}px",
                       "Resolusi", "#FFB547", icon="◕")

    divider()

    # ── Charts
    loss_cols = [c for c in df.columns if "loss" in c.lower()]
    metric_cols = [
        c for c in df.columns
        if "map" in c.lower() or "precision" in c.lower() or "recall" in c.lower()
    ]

    if loss_cols or metric_cols:
        section_label("Kurva")
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        fig.patch.set_facecolor(CHART_OUTER)

        # Loss
        ax = axes[0]
        _style_axes(ax, "Loss")
        for i, col in enumerate(loss_cols[:5]):
            label = col.split("/")[-1] if "/" in col else col
            ax.plot(df.index + 1, df[col], linewidth=2,
                    color=ACCENTS[i % len(ACCENTS)], label=label, alpha=0.95)
        ax.set_xlabel("Epoch", color=TEXT_DIM, fontsize=10)
        ax.legend(facecolor=CHART_BG, labelcolor="#B4B4C2", fontsize=9,
                  framealpha=1, edgecolor="#FFFFFF10")

        # Metrics
        ax = axes[1]
        _style_axes(ax, "Metrik")
        for i, col in enumerate(metric_cols[:5]):
            label = col.split("/")[-1] if "/" in col else col
            ax.plot(df.index + 1, df[col], linewidth=2,
                    color=ACCENTS[i % len(ACCENTS)], label=label, alpha=0.95)
        ax.set_xlabel("Epoch", color=TEXT_DIM, fontsize=10)
        ax.legend(facecolor=CHART_BG, labelcolor="#B4B4C2", fontsize=9,
                  framealpha=1, edgecolor="#FFFFFF10")

        plt.tight_layout(pad=2.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    divider()

    # ── Configuration badges
    section_label("Konfigurasi")
    st.markdown(f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      {badge(f"gambar {config.IMG_SIZE}px", "brand")}
      {badge(f"batch {config.BATCH_SIZE}", "teal")}
      {badge(f"patience {config.PATIENCE}", "pink")}
      {badge(f"lr {config.LR0}", "amber")}
      {badge(f"pekerja {config.WORKERS}", "neutral")}
      {badge(f"perangkat {config.DEVICE}", "neutral")}
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ── Table
    section_label(f"Data Epoch · {epochs} baris")
    with st.expander(f"Lihat semua {epochs} epoch"):
        st.dataframe(df, hide_index=True, use_container_width=True)
