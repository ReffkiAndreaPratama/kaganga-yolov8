"""views/home.py — Landing / overview dashboard."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import config, utils
from components.ui import (
    divider, stat_card, section_label, feature_card, step_item, badge,
)


def render():
    metrics = utils.load_metrics()
    log = utils.load_prediction_log()
    yolo_m = metrics.get("yolov8", metrics)
    map50 = yolo_m.get("metrics/mAP50(B)", yolo_m.get("metrics/mAP50", None))
    prec  = yolo_m.get("metrics/precision(B)", yolo_m.get("metrics/precision", None))
    rec   = yolo_m.get("metrics/recall(B)", yolo_m.get("metrics/recall", None))
    total_det = sum(e.get("count", 0) for e in log)

    num_classes = "253"
    try:
        if os.path.isfile(config.MODEL_BEST_PATH):
            from ultralytics import YOLO
            m = YOLO(config.MODEL_BEST_PATH)
            num_classes = str(len(m.names))
    except Exception:
        pass

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <section class="ds-hero">
      <div class="ds-hero-eyebrow">
        <span class="ds-hero-eyebrow-dot"></span>
        Ditenagai oleh YOLOv8 · {num_classes} kelas
      </div>
      <h1 class="ds-hero-title">
        Deteksi Aksara <span class="ds-gradient">Ulu Rejang</span><br>
        dalam hitungan detik.
      </h1>
      <p class="ds-hero-sub">
        Unggah satu atau banyak gambar. Dapatkan kotak pembatas presisi,
        skor kepercayaan, dan hasil deteksi yang dapat diekspor — semuanya dalam ruang kerja
        yang bersih dan dibuat khusus.
      </p>
      <div class="ds-hero-tags">
        {badge("YOLOv8", "brand", dot=True)}
        {badge("Waktu nyata", "teal", dot=True)}
        {badge("Banyak gambar", "pink", dot=True)}
        {badge("Dataset Terbuka", "neutral")}
      </div>
    </section>
    """, unsafe_allow_html=True)

    divider()

    # ── KPI strip ─────────────────────────────────────────────────────────
    section_label("Sekilas")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        stat_card("Kelas", num_classes, "Varian aksara", "#7C5CFF", icon="◈")
    with c2:
        stat_card("mAP@50", f"{map50*100:.1f}%" if map50 else "—",
                  "Akurasi deteksi", "#00D4C8", icon="◔")
    with c3:
        stat_card("Presisi", f"{prec*100:.1f}%" if prec else "—",
                  "Semua kelas", "#FF6B9D", icon="◑")
    with c4:
        stat_card("Deteksi", f"{total_det:,}", "Total keseluruhan",
                  "#FFB547", icon="◓")

    divider()

    # ── Capabilities grid ─────────────────────────────────────────────────
    section_label("Kemampuan")

    features = [
        ("⌖", "Deteksi Batch", "Unggah banyak gambar, proses paralel dengan progres."),
        ("◫", "Dashboard Analitik", "Grafik radar, kurva pelatihan, confusion matrix, skor F1."),
        ("↓", "Ekspor Kapan Saja", "Unduh gambar dan data CSV hasil deteksi."),
        ("≡", "Riwayat Deteksi", "Linimasa prediksi yang bisa dicari & diurutkan."),
        ("⚡", "Inferensi Cepat", "Arsitektur ringan YOLOv8 nano untuk hasil super cepat."),
        ("⚙", "Dapat Disesuaikan", "Atur batas kepercayaan, IoU, dan deteksi secara langsung."),
    ]

    cols = st.columns(3, gap="medium")
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(feature_card(icon, title, desc), unsafe_allow_html=True)

    divider()

    # ── Get started ──────────────────────────────────────────────────────
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        section_label("Mulai gunakan")
        steps_html = ""
        for num, text in [
            ("01", "Buka halaman <b>Deteksi</b> dari sidebar"),
            ("02", "Masukkan satu atau beberapa gambar aksara"),
            ("03", "Tinjau hasil prediksi — ekspor bila sudah siap"),
        ]:
            steps_html += step_item(num, text)
        st.markdown(steps_html, unsafe_allow_html=True)

    with col_right:
        section_label("Tentang aksara")
        st.markdown("""
        <div class="ds-card" style="padding:18px;height:100%">
          <div style="font-size:13px;color:var(--text-secondary);line-height:1.65">
            <b style="color:var(--text-primary)">Aksara Ulu Rejang</b> adalah
            tulisan tradisional masyarakat suku Rejang di Bengkulu, Sumatera.
            Sebagai bagian dari keluarga turunan Brahmi, aksara ini menyimpan
            warisan sastra yang unik — dan alat bantu ini berusaha melestarikannya
            melalui visi komputer (computer vision).
          </div>
        </div>
        """, unsafe_allow_html=True)
