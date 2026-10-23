# ◈ AksaraDetect

**Sistem deteksi aksara Ulu Rejang berbasis YOLOv8** — upload gambar, dapatkan bounding box otomatis untuk setiap aksara yang terdeteksi.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red)
![Classes](https://img.shields.io/badge/Classes-253-green)

---

## Tentang Project

AksaraDetect adalah aplikasi web untuk mendeteksi aksara **Ulu Rejang** (aksara tradisional Bengkulu) dalam gambar menggunakan model YOLOv8 object detection. Aplikasi ini mampu mendeteksi **253 kelas aksara** dengan bounding box dan confidence score.

### Aksara Ulu Rejang

Aksara Ulu Rejang adalah sistem tulisan tradisional suku Rejang di Provinsi Bengkulu, Sumatera. Aksara ini termasuk rumpun aksara Brahmi dan digunakan untuk menulis bahasa Rejang. Project ini bertujuan melestarikan aksara tersebut melalui teknologi computer vision.

---

## Fitur

| Fitur | Deskripsi |
|-------|-----------|
| **Multi-Image Detection** | Upload banyak gambar sekaligus, proses batch dengan progress bar |
| **Gallery View** | Tampilan grid/list untuk hasil deteksi multi-gambar |
| **Bounding Box Visualization** | Gambar hasil deteksi dengan box berwarna + label + confidence |
| **Download Results** | Download gambar dengan bounding box (PNG) atau data (CSV) |
| **Analytics Dashboard** | Radar chart, bar chart, donut chart, metrics table |
| **Training Monitor** | Upload results.csv, lihat loss & metrics curves per epoch |
| **Detection History** | Log lengkap setiap deteksi — filter, sort, statistik, export |
| **Configurable** | Atur confidence threshold, IoU, max detection dari UI |
| **253 Classes** | Deteksi semua variasi aksara Ulu Rejang |
| **Dark UI** | Interface minimal dark theme, clean dan profesional |

---

## Tech Stack

| Komponen | Teknologi | Versi | Fungsi |
|----------|-----------|-------|--------|
| **Object Detection** | Ultralytics YOLOv8 | 8.2.103 | Model deteksi aksara |
| **Deep Learning** | PyTorch | 2.12+ | Backend neural network |
| **Web Framework** | Streamlit | 1.39.0 | GUI web application |
| **Data Processing** | Pandas | 2.2.3 | Manipulasi data & export CSV |
| **Numerical** | NumPy | 1.26.4 | Operasi array & statistik |
| **Image Processing** | Pillow | 10.4.0 | Load, resize, convert gambar |
| **Visualization** | Matplotlib | 3.9.2 | Charts, bounding box drawing |
| **Config** | PyYAML | 6.0.2 | Parsing data.yaml dataset |
| **Computer Vision** | OpenCV | 4.11+ | Image preprocessing (via ultralytics) |
| **Dataset** | Roboflow | — | Aksara Ulu Rejang v4 (253 classes) |

---

## Struktur Project

```
AksaraDetect_Project/
├── src/
│   ├── app.py              ← Entry point (streamlit run app.py)
│   ├── config.py           ← Semua konfigurasi & path
│   ├── utils.py            ← Fungsi helper (logging, metrics, dll)
│   ├── train.py            ← Training CLI (untuk lokal)
│   ├── predict.py          ← Prediksi CLI
│   ├── components/
│   │   └── ui.py           ← Komponen UI reusable
│   ├── styles/
│   │   └── theme.css       ← Custom dark theme CSS
│   └── views/
│       ├── home.py         ← Landing page
│       ├── detect.py       ← Halaman deteksi (multi-image)
│       ├── analytics.py    ← Dashboard evaluasi model
│       ├── training.py     ← Training monitor & curves
│       ├── history.py      ← Riwayat deteksi & statistik
│       └── settings.py     ← Konfigurasi & info system
├── models/
│   ├── best.pt             ← Model YOLOv8 terlatih
│   ├── label_map.json      ← Mapping index → nama aksara
│   └── metrics.json        ← Metrics hasil training
├── logs/
│   └── prediction_log.json ← Log semua deteksi
├── results/                ← Chart & results.csv dari training
├── runs/                   ← Output training runs
├── train_colab.ipynb       ← Notebook training untuk Google Colab
├── requirements.txt        ← Dependencies
└── README.md
```

---

## Instalasi & Menjalankan

### Prerequisites

- Python 3.11 (recommended) atau 3.12
- pip

### Setup

```bash
# Clone / download project
cd AksaraDetect_Project

# Buat virtual environment
py -3.11 -m venv venv

# Aktivasi (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Aktivasi (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Menjalankan Aplikasi

```bash
# Dari root project
python -m streamlit run src/app.py

# Atau dari folder src/
cd src
streamlit run app.py
```

Buka browser di **http://localhost:8501**

---

## Training Model

Training dilakukan di **Google Colab** (gratis GPU):

1. Upload `train_colab.ipynb` ke Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Ganti `API_KEY` dengan API key Roboflow kamu
4. Jalankan semua cell
5. Download `best.pt` → taruh di `models/best.pt`

### Dataset

- **Sumber:** Roboflow — Aksara Ulu Rejang v4
- **Format:** YOLOv8
- **Jumlah kelas:** 253 aksara
- **Split:** Train / Valid / Test

### Parameter Training

| Parameter | Nilai | Keterangan |
|-----------|-------|------------|
| Model | YOLOv8n | Nano (cepat, ringan) |
| Epochs | 50 | Maksimum epoch |
| Image Size | 640px | Input resolution |
| Batch Size | 32 | Per batch |
| Patience | 100 | Early stopping |
| Learning Rate | 0.01 | Initial LR (Adam) |
| Device | GPU (0) | CUDA |

---

## Halaman Aplikasi

### 🏠 Home

Landing page dengan:
- Statistik model (classes, mAP, precision, total deteksi)
- Feature cards
- Quick start guide

### 🔍 Detect

Halaman utama untuk deteksi:
- Upload single atau multiple images
- Adjustable confidence & IoU threshold
- Progress bar untuk batch processing
- Single view (detail) atau gallery view (grid/list)
- Download hasil dengan bounding box
- Export semua deteksi ke CSV
- Tabel detail (class, confidence, bbox coordinates)

### 📊 Analytics

Dashboard evaluasi model:
- **Overview tab:** Radar chart 4 metrics, F1 score, model files
- **Charts tab:** Training curves, PR/F1/P/R curves, confusion matrix
- **Data tab:** Raw metrics table

### 📈 Training

Monitor hasil training:
- Upload results.csv langsung dari UI
- Loss curves (box, cls, dfl) & metrics curves (mAP, precision, recall)
- Training config badges
- Epoch-by-epoch data table

### 📋 History

Log semua deteksi:
- **Log tab:** Timeline semua prediksi (sort by time/count)
- **Statistics tab:** Top classes bar chart, confidence distribution histogram
- **Export tab:** Download semua data ke CSV, clear history

### ⚙️ Settings

Informasi system:
- Model info (type, classes, file size)
- Training configuration grid
- File status (semua file penting + ukuran)
- Daftar lengkap 253 kelas aksara

---

## Konfigurasi

Semua parameter ada di `src/config.py`:

```python
CONF_THRESHOLD  = 0.25    # Minimum confidence untuk deteksi
IOU_THRESHOLD   = 0.45    # IoU threshold untuk NMS
MODEL_SIZE      = "n"     # n/s/m/l/x
EPOCHS          = 50
IMG_SIZE        = 640
BATCH_SIZE      = 32
PATIENCE        = 100
LR0             = 0.01
```

---

## Prediksi via CLI

```bash
cd src

# Satu gambar
python predict.py --image ../sample.jpg

# Batch (folder)
python predict.py --batch ../folder_gambar/

# Custom confidence
python predict.py --image ../sample.jpg --conf 0.3
```

---

## Metrics Model Saat Ini

| Metric | Nilai |
|--------|-------|
| mAP@50 | 27.45% |
| mAP@50-95 | 19.67% |
| Precision | 33.89% |
| Recall | 37.61% |

> **Note:** Akurasi bisa ditingkatkan dengan:
> - Training lebih lama (100+ epochs)
> - Pakai model lebih besar (YOLOv8s/m)
> - Augmentasi data lebih agresif
> - Dataset lebih banyak & bersih

---

## Daftar Kelas Aksara (253)

Aksara yang dapat dideteksi mencakup semua kombinasi konsonan + vokal dalam sistem aksara Ulu Rejang:

```
a, ah, an, ang, ar, aw, ay,
ba, bah, ban, bang, bar, baw, bay, be, bi, bo, bu,
ca, cah, can, cang, car, caw, cay, ce, ci, co, cu,
da, dah, dan, dang, dar, daw, day, de, di, do, du,
ga, gah, gan, gang, gar, gaw, gay, ge, gi, go, gu,
ha, hah, han, hang, har, haw, hay, he, hi, ho, hu,
ja, jah, jan, jang, jar, jaw, jay, je, ji, jo, ju,
ka, kah, kan, kang, kar, kaw, kay, ke, ki, ko, ku,
la, lah, lan, lang, lar, law, lay, le, li, lo, lu,
ma, mah, man, mang, mar, maw, may, me, mi, mo, mu,
na, nah, nan, nang, nar, naw, nay, ne, ni, no, nu,
nga, ngah, ngan, ngang, ngar, ngaw, ngay, nge, ngi, ngo, ngu,
pa, pah, pan, pang, par, paw, pay, pe, pi, po, pu,
ra, rah, ran, rang, rar, raw, ray, re, ri, ro, ru,
sa, sah, san, sang, sar, saw, say, se, si, so, su,
ta, tah, tan, tang, tar, taw, tay, te, ti, to, tu,
wa, wah, wan, wang, war, waw, way, we, wi, wo, wu,
ya, yah, yan, yang, yar, yaw, yay, ye, yi, yo, yu
```

---

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| `streamlit` not recognized | Aktifkan venv: `.\venv\Scripts\Activate.ps1` |
| Model tidak ditemukan | Pastikan `models/best.pt` ada |
| Python 3.14 error | Gunakan Python 3.11 atau 3.12 |
| Port sudah dipakai | `python -m streamlit run src/app.py --server.port 8502` |
| Deteksi lambat | Normal tanpa GPU — pakai Colab untuk inference berat |
| Akurasi rendah | Training ulang dengan epoch lebih banyak / model lebih besar |

---

## Pengembangan Selanjutnya

- [ ] Training dengan model YOLOv8s/m untuk akurasi lebih tinggi
- [ ] Tambah data augmentation
- [ ] Deploy ke Streamlit Cloud / Hugging Face Spaces
- [ ] API endpoint (FastAPI) untuk integrasi
- [ ] Mobile-friendly responsive layout
- [ ] Transliterasi otomatis (aksara → latin)

---

## Lisensi

Project ini dibuat untuk keperluan akademik / penelitian.

---

## Credits

- **Model:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Dataset:** [Roboflow - Aksara Ulu Rejang](https://app.roboflow.com)
- **Framework:** [Streamlit](https://streamlit.io)
- **Font:** Space Grotesk, Inter, JetBrains Mono
