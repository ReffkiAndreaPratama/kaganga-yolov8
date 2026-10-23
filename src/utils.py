"""
utils.py — Fungsi pembantu: label map, logging prediksi, visualisasi.
"""

import os
import json
import datetime
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import config

PALETTE = {
    "primary":   "#6C63FF",
    "secondary": "#3ECFCF",
    "accent":    "#FF6584",
    "dark":      "#1A1A2E",
    "surface":   "#16213E",
    "text":      "#E0E0E0",
}


# ── Direktori ─────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [config.MODEL_DIR, config.RESULT_DIR, config.LOG_DIR, config.RUNS_DIR]:
        os.makedirs(d, exist_ok=True)


# ── Label Map ─────────────────────────────────────────────────────────────────

def load_label_map_from_yaml(yaml_path: str = config.DATA_YAML) -> dict:
    """Baca nama kelas dari data.yaml → {index: nama}."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return {i: name for i, name in enumerate(data["names"])}


def save_label_map(label_map: dict, path: str = config.LABEL_MAP_PATH):
    ensure_dirs()
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    print(f"[utils] Label map → {path}")


def load_label_map(path: str = config.LABEL_MAP_PATH) -> dict:
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


# ── Metrics ───────────────────────────────────────────────────────────────────

def save_metrics(metrics: dict, path: str = config.METRICS_PATH):
    ensure_dirs()
    existing = {}
    if os.path.isfile(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(metrics)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def load_metrics(path: str = config.METRICS_PATH) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── Prediction Log ────────────────────────────────────────────────────────────

def log_prediction(filename: str, detections: list):
    """
    detections: list of dict {class_name, confidence, bbox: [x1,y1,x2,y2]}
    """
    ensure_dirs()
    log = []
    if os.path.isfile(config.PREDICT_LOG):
        with open(config.PREDICT_LOG) as f:
            log = json.load(f)

    log.append({
        "timestamp":  datetime.datetime.now().isoformat(),
        "filename":   filename,
        "detections": detections,
        "count":      len(detections),
    })

    with open(config.PREDICT_LOG, "w") as f:
        json.dump(log, f, indent=2)


def load_prediction_log() -> list:
    if not os.path.isfile(config.PREDICT_LOG):
        return []
    with open(config.PREDICT_LOG) as f:
        return json.load(f)


# ── Visualisasi Deteksi ───────────────────────────────────────────────────────

def draw_detections(pil_image: Image.Image, detections: list) -> Image.Image:
    """
    Gambar bounding box + label di atas gambar PIL.
    detections: list of dict {class_name, confidence, bbox: [x1,y1,x2,y2]}
    """
    import random
    fig, ax = plt.subplots(1, figsize=(10, 8))
    fig.patch.set_facecolor(PALETTE["dark"])
    ax.set_facecolor(PALETTE["dark"])
    ax.imshow(pil_image)

    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"],
              "#FFD166", "#06D6A0", "#EF476F", "#118AB2"]

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        w = x2 - x1
        h = y2 - y1
        color = colors[i % len(colors)]

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        label = f"{det['class_name']} {det['confidence']:.0%}"
        ax.text(
            x1, y1 - 6, label,
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8)
        )

    ax.axis("off")
    plt.tight_layout(pad=0)

    # Konversi figure ke PIL Image
    fig.canvas.draw()
    w_fig, h_fig = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(h_fig, w_fig, 3)
    plt.close(fig)
    return Image.fromarray(buf)


# ── Training Curves ───────────────────────────────────────────────────────────

def plot_training_results(results_csv: str):
    """Baca results.csv dari YOLO training dan buat grafik."""
    import pandas as pd
    if not os.path.isfile(results_csv):
        return

    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.patch.set_facecolor(PALETTE["dark"])

    plots = [
        ("train/box_loss",  "Train Box Loss",   PALETTE["primary"]),
        ("train/cls_loss",  "Train Class Loss",  PALETTE["secondary"]),
        ("train/dfl_loss",  "Train DFL Loss",    PALETTE["accent"]),
        ("metrics/mAP50",   "mAP@50",            "#FFD166"),
        ("metrics/mAP50-95","mAP@50-95",         "#06D6A0"),
        ("val/cls_loss",    "Val Class Loss",    "#EF476F"),
    ]

    for ax, (col, title, color) in zip(axes.flat, plots):
        ax.set_facecolor(PALETTE["surface"])
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, linewidth=2)
            ax.set_title(title, color=PALETTE["text"], fontsize=11)
            ax.set_xlabel("Epoch", color=PALETTE["text"], fontsize=9)
            ax.tick_params(colors=PALETTE["text"])
            ax.grid(True, alpha=0.15, color="#555577")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")
        else:
            ax.text(0.5, 0.5, f"'{col}'\nnot found",
                    ha="center", va="center", color=PALETTE["text"],
                    transform=ax.transAxes)
            ax.set_facecolor(PALETTE["surface"])

    plt.suptitle("YOLOv8 Training Results",
                 color=PALETTE["text"], fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(config.RESULT_DIR, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=PALETTE["dark"])
    plt.close()
    print(f"[utils] Training curves → {out}")
    return out
