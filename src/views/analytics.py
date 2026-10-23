"""views/analytics.py — Model performance dashboard."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config, utils
from components.ui import (
    page_header, divider, stat_card, empty_state, info_banner, section_label,
)


# ─── Matplotlib styling ──────────────────────────────────────────────────────

CHART_BG = "#0F0F16"
CHART_OUTER = "#0A0A0F"
TEXT_LIGHT = "#FAFAFA"
TEXT_DIM = "#7A7A8C"
GRID = "#FFFFFF10"
ACCENTS = ["#7C5CFF", "#00D4C8", "#FF6B9D", "#FFB547", "#34D399", "#60A5FA"]


def _img(fname):
    p = os.path.join(config.RESULT_DIR, fname)
    return Image.open(p) if os.path.isfile(p) else None


def _style_axes(ax, title=""):
    ax.set_facecolor(CHART_BG)
    if title:
        ax.set_title(title, color=TEXT_LIGHT, fontsize=11, fontweight=600,
                     loc="left", pad=12)
    ax.tick_params(colors=TEXT_DIM, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(alpha=0.06, color="#FFFFFF", linewidth=0.6)


# ─── Render ──────────────────────────────────────────────────────────────────

def render():
    page_header(
        "Analitik",
        "Periksa performa model untuk metrik presisi, recall, dan mAP.",
        eyebrow="Performa",
    )

    metrics = utils.load_metrics()
    if not metrics:
        empty_state(
            "◫",
            "Belum ada metrik",
            "Latih model terlebih dahulu untuk melihat data performa.",
        )
        return

    yolo_m = metrics.get("yolov8", metrics)

    map50   = yolo_m.get("metrics/mAP50(B)",   yolo_m.get("metrics/mAP50",   None))
    map5095 = yolo_m.get("metrics/mAP50-95(B)", yolo_m.get("metrics/mAP50-95", None))
    prec    = yolo_m.get("metrics/precision(B)", yolo_m.get("metrics/precision", None))
    rec     = yolo_m.get("metrics/recall(B)",    yolo_m.get("metrics/recall",    None))

    # ── KPI strip
    section_label("Metrik Utama")
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("mAP@50", f"{map50*100:.1f}%" if map50 else "—",
                       "IoU 0.50", "#7C5CFF", icon="◔")
    with c2: stat_card("mAP@50-95", f"{map5095*100:.1f}%" if map5095 else "—",
                       "Rata-rata rentang IoU", "#00D4C8", icon="◑")
    with c3: stat_card("Presisi", f"{prec*100:.1f}%" if prec else "—",
                       "Tingkat positif benar", "#FF6B9D", icon="◓")
    with c4: stat_card("Recall", f"{rec*100:.1f}%" if rec else "—",
                       "Cakupan", "#FFB547", icon="◕")

    divider()

    # ── Tabs
    tab_overview, tab_charts, tab_data = st.tabs(["Ringkasan", "Grafik", "Data Mentah"])

    # ─── Overview ────────────────────────────────────────────────────────
    with tab_overview:
        if map50 and prec and rec and map5095:
            col_radar, col_summary = st.columns([3, 2], gap="large")

            with col_radar:
                fig = _radar(map50, map5095, prec, rec)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_summary:
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                st.markdown(f"""
                <div class="ds-result-card" style="text-align:left;padding:24px">
                  <div class="ds-result-label" style="text-align:left">F1 Score</div>
                  <div class="ds-result-value" style="font-size:48px;background:linear-gradient(135deg,#7C5CFF,#5CC4FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent">{f1*100:.1f}%</div>
                  <div class="ds-result-meta" style="justify-content:flex-start;margin-top:12px">
                    <span class="ds-status-dot ds-status-dot-green"></span>
                    Rata-rata harmonik dari presisi dan recall
                  </div>
                </div>

                <div style="margin-top:16px">
                  <div class="ds-list-item">
                    <span style="flex:1;font-size:12px;color:var(--text-tertiary)">Arsitektur</span>
                    <span class="ds-mono" style="color:var(--text-secondary)">YOLOv8s</span>
                  </div>
                  <div class="ds-list-item">
                    <span style="flex:1;font-size:12px;color:var(--text-tertiary)">Ukuran gambar</span>
                    <span class="ds-mono" style="color:var(--text-secondary)">{config.IMG_SIZE}px</span>
                  </div>
                  <div class="ds-list-item">
                    <span style="flex:1;font-size:12px;color:var(--text-tertiary)">Batas kepercayaan</span>
                    <span class="ds-mono" style="color:var(--text-secondary)">{config.CONF_THRESHOLD}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # Model files
        divider()
        section_label("Artefak model")
        pt_files = [
            f for f in os.listdir(config.MODEL_DIR) if f.endswith(".pt")
        ] if os.path.isdir(config.MODEL_DIR) else []

        if pt_files:
            for f in sorted(pt_files):
                path = os.path.join(config.MODEL_DIR, f)
                sz = os.path.getsize(path) / (1024 * 1024)
                active = f == "best.pt"
                badge_html = (
                    '<span class="ds-badge ds-badge-green">'
                    '<span class="ds-badge-dot"></span>active</span>'
                    if active else ''
                )
                st.markdown(f"""
                <div class="ds-list-item">
                  <span style="font-size:13px;color:var(--text-primary);
                               font-weight:{'600' if active else '400'};flex:1">{f}</span>
                  {badge_html}
                  <span class="ds-mono" style="color:var(--text-tertiary);min-width:60px;
                               text-align:right">{sz:.1f} MB</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            empty_state("◌", "Tidak ada file model",
                       "Model yang dilatih akan muncul di sini setelah tersedia.")

    # ─── Charts ──────────────────────────────────────────────────────────
    with tab_charts:
        curves = _img("training_curves.png")
        if curves:
            section_label("Kurva pelatihan")
            st.image(curves, use_container_width=True)
            divider()

        pairs = [("PR_curve.png", "F1_curve.png"), ("P_curve.png", "R_curve.png")]
        has_any = False
        for lf, rf in pairs:
            li, ri = _img(lf), _img(rf)
            if li or ri:
                has_any = True
                section_label(f"Kurva {lf.replace('_curve.png','')} & {rf.replace('_curve.png','')}")
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    if li:
                        st.image(li, use_container_width=True,
                                 caption=lf.replace(".png", ""))
                with c2:
                    if ri:
                        st.image(ri, use_container_width=True,
                                 caption=rf.replace(".png", ""))
                divider("spacer")

        cm = _img("confusion_matrix_normalized.png") or _img("confusion_matrix.png")
        if cm:
            section_label("Confusion matrix")
            st.image(cm, use_container_width=True)

        if not has_any and not curves and not cm:
            if map50 and prec and rec and map5095:
                _generate_metric_charts(map50, map5095, prec, rec)
            else:
                empty_state("◎", "Belum ada grafik",
                            "Grafik akan muncul setelah pelatihan selesai.")

    # ─── Raw data ────────────────────────────────────────────────────────
    with tab_data:
        import pandas as pd
        numeric = {k: v for k, v in yolo_m.items() if isinstance(v, (int, float))}
        if numeric:
            df = pd.DataFrame([
                {"Metric": k, "Value": round(v, 6), "Percent": f"{v*100:.2f}%"}
                for k, v in sorted(numeric.items())
            ])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("Tidak ada metrik numerik yang ditemukan.")


# ─── Charts ──────────────────────────────────────────────────────────────────

def _radar(map50, map5095, prec, rec):
    labels = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
    values = [map50 * 100, map5095 * 100, prec * 100, rec * 100]

    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_p = values + [values[0]]
    angles_p = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(CHART_OUTER)
    ax.set_facecolor(CHART_BG)

    ax.plot(angles_p, values_p, color="#7C5CFF", linewidth=2,
            marker="o", markersize=6, markerfacecolor="#FAFAFA",
            markeredgecolor="#7C5CFF", markeredgewidth=2)
    ax.fill(angles_p, values_p, color="#7C5CFF", alpha=0.12)

    for a, v in zip(angles, values):
        ax.text(a, v + 8, f"{v:.0f}%", ha="center", fontsize=10,
                color=TEXT_LIGHT, fontweight=700)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, color="#B4B4C2", fontsize=10, fontweight=500)
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels([], color=TEXT_DIM, fontsize=8)
    ax.spines["polar"].set_color("#FFFFFF10")
    ax.grid(color="#FFFFFF08", linewidth=0.7)
    plt.tight_layout()
    return fig


def _generate_metric_charts(map50, map5095, prec, rec):
    section_label("Rincian performa")
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(CHART_OUTER)
        _style_axes(ax, "Perbandingan metrik")
        names = ["mAP@50", "mAP@50-95", "Precision", "Recall"]
        vals = [map50 * 100, map5095 * 100, prec * 100, rec * 100]
        bars = ax.bar(names, vals, color=ACCENTS[:4], width=0.55,
                      edgecolor="none")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{v:.1f}%", ha="center", fontsize=9,
                    color=TEXT_LIGHT, fontweight=600)
        ax.set_ylim(0, max(vals) * 1.25)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor(CHART_OUTER)
        ax.set_facecolor(CHART_OUTER)

        sizes = [f1 * 100, 100 - f1 * 100]
        wedges, _ = ax.pie(
            sizes, colors=["#7C5CFF", "#FFFFFF08"], startangle=90,
            wedgeprops=dict(width=0.22, edgecolor=CHART_OUTER, linewidth=4),
        )
        ax.text(0, 0.08, f"{f1*100:.1f}%", ha="center", va="center",
                fontsize=28, fontweight=700, color=TEXT_LIGHT)
        ax.text(0, -0.18, "F1 Score", ha="center", va="center",
                fontsize=10, color=TEXT_DIM, fontweight=500)
        ax.set_title("Skor keseluruhan", color=TEXT_LIGHT, fontsize=11,
                     fontweight=600, loc="left", pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    divider()
    info_banner(
        "Tempatkan grafik pelatihan (PR_curve.png, F1_curve.png, dll.) ke dalam "
        "folder results/ untuk visualisasi yang lebih kaya.",
        icon="ⓘ",
    )
