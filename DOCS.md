# 📖 Dokumentasi Kode — AksaraDetect

Dokumen ini menjelaskan setiap file kode dalam project secara detail.

---

## Daftar Isi

1. [src/app.py — Entry Point](#srcapppy)
2. [src/config.py — Konfigurasi](#srcconfigpy)
3. [src/utils.py — Fungsi Utilitas](#srcutilspy)
4. [src/train.py — Training CLI](#srctrainpy)
5. [src/predict.py — Prediksi CLI](#srcpredictpy)
6. [src/components/ui.py — Komponen UI](#srccomponentsuipy)
7. [src/styles/theme.css — Styling](#srcstylesthemecss)
8. [src/views/home.py — Halaman Home](#srcviewshomepy)
9. [src/views/detect.py — Halaman Detect](#srcviewsdetectpy)
10. [src/views/analytics.py — Halaman Analytics](#srcviewsanalyticspy)
11. [src/views/training.py — Halaman Training](#srcviewstrainingpy)
12. [src/views/history.py — Halaman History](#srcviewshistorypy)
13. [src/views/settings.py — Halaman Settings](#srcviewssettingspy)
14. [train_colab.ipynb — Notebook Training](#train_colabipynb)

---

## src/app.py

**Fungsi:** Entry point aplikasi Streamlit. File ini yang dijalankan pertama kali.

**Alur kerja:**
1. Set konfigurasi halaman Streamlit (title, icon, layout)
2. Inject CSS styles dan pastikan folder output ada
3. Render sidebar (brand, navigasi, status system)
4. Route ke halaman yang dipilih user

**Komponen utama:**
- `st.set_page_config()` — konfigurasi halaman (wide layout, icon, title)
- `inject_styles()` — load CSS theme + component styles
- `st.sidebar` — navigasi menggunakan `st.radio` dengan emoji
- Routing — conditional import berdasarkan pilihan navigasi

**Kenapa `sys.path.insert`?**
Supaya Python bisa import module dari folder `src/` tanpa masalah relative import.

---

## src/config.py

**Fungsi:** Menyimpan semua konfigurasi project dalam satu tempat.

**Isi:**
- `BASE_DIR` — path root project (dihitung otomatis dari lokasi file)
- Path ke semua file penting (model, metrics, log, dataset)
- Parameter deteksi (confidence threshold, IoU threshold)
- Parameter training (epochs, batch size, image size, dll)
- Info aplikasi (nama, versi, tagline)

**Kenapa pakai `os.path.join`?**
Supaya path bisa jalan di Windows maupun Linux/Mac tanpa hardcode separator.

**Kenapa `BASE_DIR` dihitung dari `__file__`?**
Supaya tidak perlu hardcode path absolut. File ini ada di `src/`, jadi `dirname(dirname(__file__))` = root project.

---

## src/utils.py

**Fungsi:** Kumpulan fungsi helper yang dipakai di banyak tempat.

### Fungsi-fungsi:

| Fungsi | Kegunaan |
|--------|----------|
| `ensure_dirs()` | Buat folder models/, results/, logs/, runs/ jika belum ada |
| `load_label_map_from_yaml()` | Baca nama kelas dari data.yaml → dict {index: nama} |
| `save_label_map()` | Simpan label map ke JSON |
| `load_label_map()` | Load label map dari JSON |
| `save_metrics()` | Simpan/update metrics ke JSON (merge dengan existing) |
| `load_metrics()` | Load metrics dari JSON |
| `log_prediction()` | Catat satu prediksi ke prediction_log.json |
| `load_prediction_log()` | Load semua log prediksi |
| `draw_detections()` | Gambar bounding box di atas gambar (untuk CLI) |
| `plot_training_results()` | Generate chart dari results.csv |

### Format prediction log:
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "filename": "sample.jpg",
  "detections": [
    {"class_name": "ka", "confidence": 0.85, "bbox": [x1, y1, x2, y2]}
  ],
  "count": 1
}
```

### Format metrics.json:
```json
{
  "metrics/mAP50(B)": 0.2744,
  "metrics/mAP50-95(B)": 0.1966,
  "metrics/precision(B)": 0.3388,
  "metrics/recall(B)": 0.3760
}
```

---

## src/train.py

**Fungsi:** Script CLI untuk training model YOLOv8.

**Alur kerja:**
1. Parse arguments (model size, epochs, batch, resume)
2. Validasi dataset (cek data.yaml ada)
3. Load model pretrained atau resume dari checkpoint
4. Jalankan training dengan `model.train()`
5. Copy best.pt dan last.pt ke folder models/
6. Simpan label map dan metrics
7. Generate training curves dari results.csv
8. Copy chart files (confusion matrix, PR curve, dll) ke results/

**Arguments:**
```
--model   n/s/m/l/x    Ukuran model (default: n = nano)
--epochs  int          Jumlah epoch (default: 50)
--imgsz   int          Ukuran gambar input (default: 640)
--batch   int          Batch size (default: 32)
--resume               Lanjut dari checkpoint terakhir
```

**Catatan:** Training biasanya dilakukan di Google Colab (GPU gratis), bukan di lokal.

---

## src/predict.py

**Fungsi:** Script CLI untuk prediksi/deteksi tanpa GUI.

**Alur kerja:**
1. Load model YOLOv8 dari best.pt
2. Terima input: satu gambar (`--image`) atau folder (`--batch`)
3. Jalankan prediksi per gambar
4. Log hasil ke prediction_log.json
5. Print hasil ke terminal

**Output terminal:**
```
=========================================
  Gambar     : sample.jpg
  Deteksi    : 3 aksara
-----------------------------------------
  [1] ka           conf=0.85  bbox=(10,20,50,60)
  [2] ga           conf=0.72  bbox=(70,30,110,80)
  [3] nga          conf=0.61  bbox=(120,25,160,75)
=========================================
```

---

## src/components/ui.py

**Fungsi:** Komponen UI reusable yang dipakai di semua halaman.

### Komponen:

| Komponen | Fungsi |
|----------|--------|
| `inject_styles()` | Load theme.css + inject component CSS |
| `page_header(title, subtitle)` | Header halaman dengan title + subtitle |
| `stat_card(label, value, sub, color)` | Card statistik (angka besar + label) |
| `divider()` | Garis pemisah horizontal |
| `empty_state(icon, title, desc)` | Placeholder saat tidak ada data |
| `info_banner(text, icon)` | Banner informasi/tips |
| `metric_row(items)` | Baris metrics horizontal |

### CSS Classes:
- `.ak-card` — card dengan hover effect
- `.ak-stat` — stat card centered
- `.ak-divider` — divider line
- `.ak-empty` — empty state container
- `.ak-badge` — badge/tag kecil (purple, teal, pink, green, orange)
- `.ak-info` — info banner
- `.ak-det-item` — item deteksi dalam list
- `.ak-mono` — monospace text

---

## src/styles/theme.css

**Fungsi:** Custom CSS untuk override default Streamlit styling.

**Yang di-style:**
- Background app (`#09090F` — hampir hitam)
- Sidebar (dark, border subtle)
- Radio navigation items (hover, active state)
- Buttons (primary purple, download outline)
- File uploader (dashed border, hover glow)
- Slider, selectbox, number input
- Expander, tabs, dataframe
- Progress bar (gradient purple → teal)
- Scrollbar (thin, subtle)
- Hide default Streamlit nav, menu, footer

**Font stack:**
- Headings: Space Grotesk (geometric, modern)
- Body: Inter (readable, clean)
- Code/data: JetBrains Mono (monospace)

---

## src/views/home.py

**Fungsi:** Landing page / halaman beranda.

**Sections:**
1. **Hero** — judul besar "Deteksi Aksara Ulu Rejang" dengan gradient text + badges
2. **Stats** — 4 stat cards (classes, mAP, precision, total detections)
3. **Capabilities** — 6 feature cards dalam grid 3 kolom
4. **Get Started** — 3 langkah numbered steps

**Data yang ditampilkan:**
- Jumlah kelas dari model (load YOLO → `len(model.names)`)
- Metrics dari metrics.json
- Total deteksi dari prediction_log.json

---

## src/views/detect.py

**Fungsi:** Halaman utama untuk deteksi aksara. File terpanjang dan paling kompleks.

### Fungsi internal:

| Fungsi | Kegunaan |
|--------|----------|
| `_load_model()` | Load YOLOv8 model (cached dengan `@st.cache_resource`) |
| `_draw_boxes()` | Gambar bounding box di atas gambar menggunakan matplotlib |
| `_detect()` | Jalankan prediksi pada satu gambar |
| `_to_bytes()` | Convert PIL Image ke bytes (untuk download) |
| `_single_view()` | Render detail view untuk 1 gambar |
| `_gallery_view()` | Render gallery untuk multi-gambar |
| `_gallery_card()` | Render satu card di gallery |

### Alur kerja:
1. Load model (cached — hanya load sekali)
2. Tampilkan controls (confidence, IoU, max objects)
3. File uploader (accept multiple)
4. Process semua gambar dengan progress bar
5. Tampilkan summary stats (images, detections, avg conf)
6. Jika 1 gambar → single view (detail + detection list)
7. Jika banyak → gallery view (grid/list toggle)

### `_draw_boxes()` detail:
- Buat matplotlib figure dengan background gelap
- Loop setiap deteksi → gambar Rectangle + text label
- Warna berbeda per deteksi (12 warna cycling)
- Font size adaptif berdasarkan ukuran gambar
- Save ke BytesIO → convert ke PIL Image
- Close figure (penting untuk memory)

### `@st.cache_resource`:
Decorator Streamlit yang memastikan model hanya di-load sekali ke memory, tidak setiap kali halaman di-refresh.

---

## src/views/analytics.py

**Fungsi:** Dashboard evaluasi performa model.

### Tabs:

**Overview:**
- Radar chart (polar plot) 4 metrics
- F1 Score calculation
- Model files list

**Charts:**
- Cari file PNG di folder results/ (training_curves, PR_curve, dll)
- Jika tidak ada → generate chart dari metrics.json:
  - Bar chart perbandingan metrics
  - Donut chart F1 Score

**Data:**
- Tabel raw metrics (semua angka dari metrics.json)

### `_radar()`:
Membuat polar/radar chart menggunakan matplotlib:
- 4 axis: mAP@50, mAP@50-95, Precision, Recall
- Fill area + markers
- Value labels di setiap titik

### `_generate_metric_charts()`:
Fallback ketika tidak ada file chart dari training:
- Bar chart horizontal dengan value labels
- Donut chart (pie dengan hole) untuk F1 Score

---

## src/views/training.py

**Fungsi:** Viewer untuk hasil training (results.csv).

**Alur:**
1. Cari results.csv di results/ atau runs/aksara_ulu/
2. Jika tidak ada → tampilkan file uploader
3. Jika ada → parse CSV, tampilkan:
   - Stat cards (epochs, model, best mAP)
   - Loss curves (semua kolom yang mengandung "loss")
   - Metrics curves (mAP, precision, recall)
   - Training config badges
   - Epoch table (expandable)

**Kenapa bisa upload CSV?**
Karena training dilakukan di Colab, user perlu cara mudah untuk memasukkan results.csv tanpa harus copy file manual.

---

## src/views/history.py

**Fungsi:** Log dan statistik semua deteksi yang pernah dilakukan.

### Tabs:

**Log:**
- List semua prediksi (timestamp, filename, count, top classes)
- Sort: newest, oldest, most detections
- Maksimal 60 entries ditampilkan

**Statistics:**
- Bar chart horizontal: top 12 kelas yang paling sering terdeteksi
- Histogram: distribusi confidence score + mean line

**Export:**
- Tabel semua deteksi (flatten: 1 row per detection)
- Download CSV button
- Clear history button (danger zone)

### Counter dari collections:
Dipakai untuk menghitung frekuensi setiap kelas aksara yang terdeteksi → `Counter(class_names).most_common(12)`

---

## src/views/settings.py

**Fungsi:** Informasi system dan konfigurasi.

**Sections:**
1. **Model info** — type, jumlah classes, file size (load dari YOLO model)
2. **Training config** — grid parameter dari config.py
3. **Files** — status setiap file penting (ada/tidak + ukuran)
4. **All classes** — expandable grid semua 253 nama aksara

### `_size()`:
Helper function untuk format ukuran file (bytes → KB → MB).

---

## train_colab.ipynb

**Fungsi:** Notebook Jupyter untuk training di Google Colab (GPU gratis).

### Cells:

| Cell | Fungsi |
|------|--------|
| 1 | Cek GPU tersedia (nvidia-smi) |
| 2 | Install ultralytics + roboflow |
| 3 | Download dataset dari Roboflow API |
| 3B | Alternatif: upload ZIP manual |
| 4 | Fix path di data.yaml (absolute path) |
| 5 | Training YOLOv8 (50 epochs, GPU) |
| 6 | Validasi model (mAP, precision, recall) |
| 7 | Download best.pt ke PC |
| 8 | Simpan ke Google Drive (opsional) |

**Kenapa Colab?**
- Training butuh GPU (CUDA)
- Colab menyediakan T4 GPU gratis
- Training 50 epoch ~5-10 menit di Colab vs berjam-jam di CPU

---

## Arsitektur & Flow Data

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  User       │────▶│  Streamlit   │────▶│  YOLOv8     │
│  (Browser)  │◀────│  (app.py)    │◀────│  (best.pt)  │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
             ┌──────────┐  ┌──────────┐
             │  logs/   │  │  models/ │
             │  (JSON)  │  │  (.pt)   │
             └──────────┘  └──────────┘
```

**Flow deteksi:**
1. User upload gambar via browser
2. Streamlit terima file → convert ke PIL Image
3. PIL Image dikirim ke YOLOv8 model (`model.predict()`)
4. Model return: bounding boxes + class IDs + confidence scores
5. Hasil di-render sebagai gambar dengan matplotlib
6. Log disimpan ke prediction_log.json
7. Gambar + data ditampilkan ke user

---

## Library & Perannya

| Library | Import | Dipakai untuk |
|---------|--------|---------------|
| `ultralytics` | `from ultralytics import YOLO` | Load & run YOLOv8 model |
| `streamlit` | `import streamlit as st` | Web framework (UI, widgets, layout) |
| `PIL/Pillow` | `from PIL import Image` | Load, convert, resize gambar |
| `matplotlib` | `import matplotlib.pyplot as plt` | Gambar bounding box, charts |
| `numpy` | `import numpy as np` | Operasi array, statistik (mean, dll) |
| `pandas` | `import pandas as pd` | DataFrame untuk tabel & CSV export |
| `yaml` | `import yaml` | Parse data.yaml (konfigurasi dataset) |
| `json` | `import json` | Read/write metrics.json, prediction_log.json |
| `os` | `import os` | Path manipulation, file existence check |
| `io` | `import io` | BytesIO untuk convert gambar ke bytes |
| `datetime` | `import datetime` | Timestamp untuk log prediksi |
| `collections` | `from collections import Counter` | Hitung frekuensi kelas |
| `shutil` | `import shutil` | Copy file (model, charts) |
| `argparse` | `import argparse` | Parse CLI arguments (train.py, predict.py) |

---

## Konvensi Kode

- **Naming:** snake_case untuk fungsi/variabel, PascalCase tidak dipakai
- **Private functions:** prefix `_` (contoh: `_load_model`, `_draw_boxes`)
- **Caching:** `@st.cache_resource` untuk model (heavy object), `@st.cache_data` untuk data
- **HTML in Streamlit:** `st.markdown(..., unsafe_allow_html=True)` untuk custom styling
- **Error handling:** `try/except` saat load model (graceful fallback)
- **File I/O:** selalu cek `os.path.isfile()` sebelum baca file
