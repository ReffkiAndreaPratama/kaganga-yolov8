"""views/detect.py — Detection workspace."""

import os, sys, io, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

import config
from components.ui import page_header, divider, empty_state, badge, section_label


# ─── Model loader ─────────────────────────────────────────────────────────────

@st.cache_resource
def _load_resnet_model():
    model_path = os.path.join(config.BASE_DIR, "best_classifier.pth")
    classes_path = os.path.join(config.BASE_DIR, "classifier_classes.json")

    if not os.path.exists(model_path) or not os.path.exists(classes_path):
        return None, None, None, None

    with open(classes_path, "r") as f:
        classes = json.load(f)

    # Force CPU for web app inference — avoids CUDA memory issues
    # and is fast enough for single-image classification
    device = torch.device("cpu")

    try:
        model = models.resnet34()
        model.fc = nn.Linear(model.fc.in_features, len(classes))
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device).eval()

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        return model, classes, transform, device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None, None, None


def _predict(model, classes, transform, device, pil_img):
    img = pil_img.convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model(tensor)
        probs = torch.softmax(out, dim=1)[0]
        conf, idx = probs.max(0)
        text = classes[idx.item()]
        confidence = conf.item()
    return text, confidence, time.time() - t0


def _to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─── Render ───────────────────────────────────────────────────────────────────

def render():
    page_header(
        "Deteksi",
        "Unggah gambar dan biarkan pengklasifikasi mengidentifikasi setiap karakter aksara.",
        eyebrow="Ruang Kerja",
    )

    model, classes, transform, device = _load_resnet_model()

    if model is None:
        empty_state(
            "⚠",
            "Model tidak tersedia",
            "Tempatkan best_classifier.pth dan classifier_classes.json di direktori utama proyek.",
        )
        return

    # ── Status row
    device_label = "GPU" if str(device) == "cuda" else "CPU"
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:24px;flex-wrap:wrap">
      {badge(f"ResNet-34 · {len(classes)} kelas", "brand", dot=True)}
      {badge(f"Perangkat: {device_label}", "teal")}
      {badge("Model dimuat", "green", dot=True)}
    </div>
    """, unsafe_allow_html=True)

    # ── Upload zone
    section_label("Unggah")
    files = st.file_uploader(
        "Letakkan gambar atau klik untuk menelusuri",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not files:
        empty_state(
            "▤",
            "Letakkan gambar di sini",
            "JPG, PNG, BMP, WEBP — mendukung satu atau beberapa file.",
        )
        return

    # ── Process
    results = []

    # Loading state with skeleton + spinner
    loading_placeholder = st.empty()
    loading_placeholder.markdown(f"""
    <div style="padding:20px;background:var(--bg-surface);
                border:1px solid var(--border-default);border-radius:var(--r-lg);
                margin-top:8px">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px">
        <div style="width:18px;height:18px;border:2px solid var(--bg-elevated);
                    border-top-color:var(--brand);border-right-color:var(--brand);
                    border-radius:50%;animation:spin 720ms linear infinite;
                    filter:drop-shadow(0 0 6px var(--brand-glow))"></div>
        <span style="font-size:13px;color:var(--text-primary);font-weight:500">
          Menganalisis {len(files)} gambar…
        </span>
      </div>
      <div class="ds-skeleton ds-skeleton-line" style="width:80%"></div>
      <div class="ds-skeleton ds-skeleton-line" style="width:60%"></div>
      <div class="ds-skeleton ds-skeleton-line" style="width:70%;margin-bottom:0"></div>
    </div>
    """, unsafe_allow_html=True)

    progress = st.progress(0)
    for i, f in enumerate(files):
        img = Image.open(f).convert("RGB")
        text, conf, dur = _predict(model, classes, transform, device, img)
        results.append({
            "name": f.name,
            "image": img,
            "text": text,
            "confidence": conf,
            "duration": dur,
        })
        progress.progress((i + 1) / len(files))

    progress.empty()
    loading_placeholder.empty()

    # ── Summary stats
    avg_conf = sum(r["confidence"] for r in results) / len(results)
    total_time = sum(r["duration"] for r in results)

    divider()
    section_label("Hasil")

    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:24px;flex-wrap:wrap;
                padding:14px 18px;background:var(--bg-surface);
                border:1px solid var(--border-default);border-radius:var(--r-lg);
                margin-bottom:20px">
      <div>
        <div style="font-size:11px;color:var(--text-tertiary);
                    text-transform:uppercase;letter-spacing:0.06em;
                    font-weight:600;margin-bottom:2px">Diproses</div>
        <div style="font-size:18px;font-weight:700;color:var(--text-primary);
                    font-variant-numeric:tabular-nums">{len(results)}</div>
      </div>
      <div style="width:1px;height:32px;background:var(--border-default)"></div>
      <div>
        <div style="font-size:11px;color:var(--text-tertiary);
                    text-transform:uppercase;letter-spacing:0.06em;
                    font-weight:600;margin-bottom:2px">Rata-rata Kepercayaan</div>
        <div style="font-size:18px;font-weight:700;color:var(--brand);
                    font-variant-numeric:tabular-nums">{avg_conf*100:.1f}%</div>
      </div>
      <div style="width:1px;height:32px;background:var(--border-default)"></div>
      <div>
        <div style="font-size:11px;color:var(--text-tertiary);
                    text-transform:uppercase;letter-spacing:0.06em;
                    font-weight:600;margin-bottom:2px">Total Waktu</div>
        <div style="font-size:18px;font-weight:700;color:var(--text-primary);
                    font-variant-numeric:tabular-nums">{total_time:.2f}s</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Render results
    if len(results) == 1:
        _single_view(results[0])
    else:
        _gallery_view(results)


def _single_view(r: dict):
    img, text, name, conf, dur = (
        r["image"], r["text"], r["name"], r["confidence"], r["duration"]
    )

    col_img, col_info = st.columns([5, 4], gap="large")

    with col_img:
        st.image(img, use_container_width=True)
        st.download_button(
            "Unduh gambar",
            _to_bytes(img),
            file_name=name,
            mime="image/png",
            key="dl_single",
            use_container_width=True,
        )

    with col_info:
        # Result hero
        st.markdown(f"""
        <div class="ds-result-card">
          <div class="ds-result-label">Aksara Terdeteksi</div>
          <div class="ds-result-value">{text if text.strip() else "—"}</div>
          <div class="ds-result-meta">
            <span class="ds-status-dot ds-status-dot-green"></span>
            Inferensi {dur*1000:.0f} ms
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Detail rows
        st.markdown(f"""
        <div style="margin-top:20px">
          <div class="ds-list-item">
            <span style="flex:1;font-size:12px;color:var(--text-tertiary)">Kepercayaan</span>
            <span class="ds-mono" style="color:var(--brand);font-weight:600">{conf*100:.2f}%</span>
          </div>
          <div class="ds-list-item">
            <span style="flex:1;font-size:12px;color:var(--text-tertiary)">Nama File</span>
            <span class="ds-mono" style="color:var(--text-secondary);max-width:60%;
                       overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{name}</span>
          </div>
          <div class="ds-list-item">
            <span style="flex:1;font-size:12px;color:var(--text-tertiary)">Model</span>
            <span class="ds-mono" style="color:var(--text-secondary)">ResNet-34</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


def _gallery_view(results: list):
    cols = st.columns(3, gap="medium")
    for i, r in enumerate(results):
        with cols[i % 3]:
            _gallery_card(r)


def _gallery_card(r: dict):
    img, text, name, conf, dur = (
        r["image"], r["text"], r["name"], r["confidence"], r["duration"]
    )

    # Truncate filename
    short = name if len(name) <= 24 else name[:21] + "…"

    # Use a wrapper card
    st.markdown(f"""
    <div class="ds-card" style="padding:0;overflow:hidden;margin-bottom:16px">
      <div style="padding:12px 14px;border-bottom:1px solid var(--border-default);
                  display:flex;align-items:center;justify-content:space-between;gap:8px">
        <span style="font-size:12px;color:var(--text-tertiary);
                     overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
                     font-family:'JetBrains Mono',monospace">{short}</span>
        {badge(text if text.strip() else "—", "brand")}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.image(img, use_container_width=True)

    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;
                padding:8px 0 16px;font-size:11.5px;color:var(--text-tertiary)">
      <span>Kepercayaan
        <span class="ds-mono" style="color:var(--brand);font-weight:600">
          {conf*100:.1f}%
        </span>
      </span>
      <span class="ds-mono">{dur*1000:.0f}ms</span>
    </div>
    """, unsafe_allow_html=True)
