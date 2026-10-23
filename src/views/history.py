"""views/history.py — Detection log & statistics."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

import config, utils
from components.ui import (
    page_header, divider, empty_state, stat_card, section_label, badge,
)


CHART_BG = "#0F0F16"
CHART_OUTER = "#0A0A0F"
TEXT_LIGHT = "#FAFAFA"
TEXT_DIM = "#7A7A8C"


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
        "Riwayat",
        "Setiap deteksi dicatat, diurutkan, dan siap diekspor.",
        eyebrow="Aktivitas",
    )

    log = utils.load_prediction_log()
    if not log:
        empty_state("≡", "Belum ada riwayat", "Jalankan deteksi untuk mulai mencatat aktivitas.")
        return

    total = len(log)
    total_det = sum(e.get("count", 0) for e in log)
    all_dets = [d for e in log for d in e.get("detections", [])]
    avg_conf = np.mean([d["confidence"] for d in all_dets]) * 100 if all_dets else 0
    unique_cls = len(set(d["class_name"] for d in all_dets))

    # ── KPIs
    section_label("Ringkasan")
    c1, c2, c3, c4 = st.columns(4)
    with c1: stat_card("Gambar", f"{total:,}", "Diproses", "#7C5CFF", icon="◔")
    with c2: stat_card("Deteksi", f"{total_det:,}", "Total", "#00D4C8", icon="◑")
    with c3: stat_card("Rata-rata Kepercayaan", f"{avg_conf:.0f}%", "Semua data", "#FF6B9D", icon="◓")
    with c4: stat_card("Kelas", f"{unique_cls}", "Aksara unik", "#FFB547", icon="◕")

    divider()

    tab_log, tab_stats, tab_export = st.tabs(["Aktivitas", "Statistik", "Ekspor"])

    # ─── Activity ────────────────────────────────────────────────────────
    with tab_log:
        col_filter, col_search = st.columns([1, 2])
        with col_filter:
            sort = st.selectbox(
                "Urutkan berdasarkan",
                ["Terbaru", "Terlama", "Deteksi terbanyak"],
            )
        with col_search:
            query = st.text_input(
                "Filter nama file",
                placeholder="mis. sample.jpg",
            )

        if sort == "Terbaru":
            log_s = sorted(log, key=lambda x: x["timestamp"], reverse=True)
        elif sort == "Terlama":
            log_s = sorted(log, key=lambda x: x["timestamp"])
        else:
            log_s = sorted(log, key=lambda x: x.get("count", 0), reverse=True)

        if query:
            log_s = [e for e in log_s if query.lower() in e.get("filename", "").lower()]

        st.markdown(f"""
        <div style="font-size:12px;color:var(--text-tertiary);margin:16px 0 12px">
          Menampilkan {min(len(log_s), 60)} dari {len(log_s)} data
        </div>
        """, unsafe_allow_html=True)

        for entry in log_s[:60]:
            ts = entry["timestamp"]
            date_str = ts[:10] if len(ts) > 10 else ""
            time_str = ts[11:19] if len(ts) > 10 else ts
            fname = entry.get("filename", "—")
            count = entry.get("count", 0)
            dets = entry.get("detections", [])
            top_cls = sorted(set(d["class_name"] for d in dets))[:3]
            cls_str = ", ".join(top_cls) if top_cls else "—"
            count_color = "var(--brand)" if count > 0 else "var(--text-quaternary)"

            st.markdown(f"""
            <div class="ds-list-item">
              <div style="min-width:90px">
                <div class="ds-mono" style="font-size:11px;color:var(--text-secondary);font-weight:600">{date_str}</div>
                <div class="ds-mono" style="font-size:11px;color:var(--text-quaternary)">{time_str}</div>
              </div>
              <div style="width:1px;height:32px;background:var(--border-default)"></div>
              <div style="flex:1;min-width:0">
                <div style="font-size:13px;color:var(--text-primary);font-weight:500;
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{fname}</div>
                <div style="font-size:11.5px;color:var(--text-tertiary);
                            overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{cls_str}</div>
              </div>
              <div style="text-align:right;min-width:72px">
                <div style="font-size:18px;font-weight:700;color:{count_color};
                            font-variant-numeric:tabular-nums;line-height:1.1">{count}</div>
                <div style="font-size:10px;color:var(--text-quaternary);
                            text-transform:uppercase;letter-spacing:0.05em;
                            font-weight:600">deteksi</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ─── Statistics ──────────────────────────────────────────────────────
    with tab_stats:
        if all_dets:
            c1, c2 = st.columns(2, gap="medium")

            with c1:
                section_label("Kelas teratas")
                counts = Counter(d["class_name"] for d in all_dets).most_common(12)
                labels, values = zip(*counts)
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                fig.patch.set_facecolor(CHART_OUTER)
                _style_axes(ax)
                ax.barh(list(labels)[::-1], list(values)[::-1],
                        color="#7C5CFF", height=0.62, edgecolor="none")
                ax.set_xlabel("Jumlah", color=TEXT_DIM, fontsize=9)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with c2:
                section_label("Distribusi kepercayaan")
                confs = [d["confidence"] * 100 for d in all_dets]
                fig, ax = plt.subplots(figsize=(5.5, 4.5))
                fig.patch.set_facecolor(CHART_OUTER)
                _style_axes(ax)
                ax.hist(confs, bins=20, color="#00D4C8", alpha=0.85,
                        edgecolor=CHART_OUTER, linewidth=2)
                ax.axvline(np.mean(confs), color="#FF6B9D", linewidth=2,
                           linestyle="--", label=f"Rata-rata {np.mean(confs):.0f}%")
                ax.set_xlabel("Kepercayaan (%)", color=TEXT_DIM, fontsize=9)
                ax.legend(facecolor=CHART_BG, labelcolor="#B4B4C2",
                          fontsize=9, framealpha=1, edgecolor="#FFFFFF10")
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            empty_state("◌", "Tidak ada data", "Jalankan deteksi untuk memunculkan statistik.")

    # ─── Export ──────────────────────────────────────────────────────────
    with tab_export:
        rows = []
        for e in log:
            for d in e.get("detections", []):
                rows.append({
                    "Timestamp": e["timestamp"],
                    "File": e.get("filename", ""),
                    "Class": d["class_name"],
                    "Confidence": round(d["confidence"], 4),
                    "X1": int(d["bbox"][0]), "Y1": int(d["bbox"][1]),
                    "X2": int(d["bbox"][2]), "Y2": int(d["bbox"][3]),
                })

        if rows:
            df = pd.DataFrame(rows)
            section_label(f"{len(rows):,} deteksi dari {total:,} gambar")
            st.dataframe(df.head(200), hide_index=True, use_container_width=True)
            divider("spacer")
            st.download_button(
                "Unduh sebagai CSV",
                df.to_csv(index=False).encode(),
                "aksaradetect-history.csv",
                "text/csv",
                use_container_width=False,
            )
        else:
            empty_state("◌", "Tidak ada yang diekspor", "Data deteksi akan muncul di sini.")

        divider()
        section_label("Zona bahaya")
        st.markdown("""
        <div class="ds-banner" style="background:rgba(248,113,113,0.06);
             border-color:rgba(248,113,113,0.20);color:var(--text-secondary)">
          <span class="ds-banner-icon" style="color:var(--red)">⚠</span>
          <span>Menghapus riwayat bersifat permanen dan tidak dapat dibatalkan.</span>
        </div>
        """, unsafe_allow_html=True)
        divider("spacer")
        if st.button("Hapus semua riwayat", type="secondary"):
            import json
            with open(config.PREDICT_LOG, "w") as f:
                json.dump([], f)
            st.success("Riwayat dihapus.")
            st.rerun()
