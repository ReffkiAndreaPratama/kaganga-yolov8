# 🗺️ Roadmap — AksaraDetect

Rencana pengembangan project ke depan, diurutkan berdasarkan prioritas dan effort.

---

## Status Saat Ini (v1.0.0)

✅ Deteksi aksara Ulu Rejang (253 kelas)
✅ Multi-image upload + batch processing
✅ Bounding box visualization + download
✅ Analytics dashboard (radar, bar, donut chart)
✅ Detection history + export CSV
✅ Training monitor (upload results.csv)
✅ Dark minimal UI
✅ Model: YOLOv8n (mAP@50: 27.4%)

---

## Phase 1 — Improve Akurasi Model (Prioritas Tinggi)

**Target:** mAP@50 > 60%

| Task | Detail | Effort |
|------|--------|--------|
| Training lebih lama | 150-200 epochs (bukan 50) | Rendah |
| Model lebih besar | YOLOv8s atau YOLOv8m | Rendah |
| Data augmentation | Mosaic, mixup, copy-paste augmentation | Sedang |
| Hyperparameter tuning | lr, warmup, weight decay, mosaic ratio | Sedang |
| Dataset cleanup | Review & fix label yang salah di Roboflow | Tinggi |
| Tambah data | Collect lebih banyak sampel per kelas | Tinggi |
| Class balancing | Oversample kelas yang jarang | Sedang |

**Cara:**
```python
# Di train_colab.ipynb, ganti:
MODEL = 'yolov8s.pt'  # s bukan n
# dan:
results = model.train(
    epochs=150,        # lebih lama
    patience=20,       # early stop lebih sabar
    augment=True,      # augmentasi agresif
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
)
```

---

## Phase 2 — Fitur Baru (Prioritas Sedang)

### 2.1 Transliterasi Otomatis
**Apa:** Setelah deteksi aksara, susun urutan aksara → konversi ke teks Latin.

**Cara implementasi:**
- Sort deteksi berdasarkan posisi X (kiri ke kanan)
- Group per baris berdasarkan posisi Y
- Map class_name → huruf Latin
- Tampilkan hasil transliterasi di UI

**Effort:** Sedang (perlu riset urutan baca aksara Rejang)

### 2.2 Comparison Mode
**Apa:** Upload 2 gambar, bandingkan hasil deteksi side-by-side.

**Effort:** Rendah (UI sudah support 2 kolom)

### 2.3 Crop Individual Aksara
**Apa:** Setelah deteksi, user bisa klik/pilih satu aksara → crop + zoom.

**Effort:** Sedang

### 2.4 Annotation Tool
**Apa:** User bisa koreksi/tambah bounding box manual → export sebagai training data baru.

**Effort:** Tinggi (butuh canvas interaktif)

### 2.5 Model Selector
**Apa:** Jika punya beberapa model (nano, small, medium), user bisa pilih dari dropdown.

**Effort:** Rendah

---

## Phase 3 — Deployment (Prioritas Sedang)

### 3.1 Deploy ke Streamlit Cloud
**Apa:** Hosting gratis di share.streamlit.io

**Langkah:**
1. Push ke GitHub (tanpa best.pt — terlalu besar)
2. Upload best.pt ke Google Drive / Hugging Face
3. Tambah script download model saat startup
4. Deploy via streamlit.io

**Blocker:** File best.pt ~6MB, masih OK untuk Streamlit Cloud.

### 3.2 Deploy ke Hugging Face Spaces
**Apa:** Hosting gratis dengan GPU inference.

**Keuntungan:** Bisa pakai GPU untuk inference lebih cepat.

### 3.3 Docker Container
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "src/app.py", "--server.port=8501"]
```

### 3.4 REST API (FastAPI)
**Apa:** Endpoint API untuk integrasi dengan aplikasi lain.

```python
# api.py
from fastapi import FastAPI, UploadFile
from ultralytics import YOLO

app = FastAPI()
model = YOLO("models/best.pt")

@app.post("/detect")
async def detect(file: UploadFile):
    img = Image.open(file.file)
    results = model.predict(img)
    return {"detections": [...]}
```

---

## Phase 4 — UX Enhancement (Prioritas Rendah)

### 4.1 Migrate ke Reflex/Next.js
**Apa:** Pindah dari Streamlit ke framework yang lebih fleksibel.

**Keuntungan:**
- Custom animations & transitions
- Real SPA routing
- Better mobile support
- WebSocket real-time updates

**Effort:** Sangat tinggi (rewrite total)

### 4.2 Mobile App (Flutter/React Native)
**Apa:** Aplikasi mobile yang bisa foto langsung → deteksi.

**Arsitektur:**
- Mobile app → REST API (FastAPI) → YOLOv8 model
- Atau: export model ke TFLite/ONNX → run di device

### 4.3 Progressive Web App (PWA)
**Apa:** Streamlit app yang bisa di-install di HP sebagai app.

**Effort:** Rendah (tambah manifest.json + service worker)

---

## Phase 5 — Research & Advanced (Long-term)

### 5.1 Model Optimization
| Teknik | Keuntungan |
|--------|-----------|
| ONNX export | Inference lebih cepat tanpa PyTorch |
| TensorRT | 2-5x speedup di NVIDIA GPU |
| Quantization (INT8) | Model lebih kecil, inference di edge device |
| Knowledge distillation | Model kecil dengan akurasi model besar |

### 5.2 Active Learning Pipeline
**Apa:** System yang otomatis identify gambar yang model kurang yakin → minta user label → retrain.

### 5.3 Multi-script Support
**Apa:** Extend ke aksara lain (Lampung, Batak, Bugis, Jawa).

**Cara:** Train model baru per aksara, atau satu model multi-script.

### 5.4 OCR Integration
**Apa:** Setelah deteksi individual aksara, gabungkan dengan OCR untuk baca teks lengkap.

### 5.5 Handwriting Recognition
**Apa:** Deteksi aksara yang ditulis tangan (bukan hanya cetak/ukir).

---

## Backlog (Nice to Have)

- [ ] Dark/Light theme toggle
- [ ] Keyboard shortcuts (Ctrl+U = upload, Ctrl+D = detect)
- [ ] Drag & drop reorder deteksi
- [ ] Zoom & pan pada gambar hasil
- [ ] Undo/redo detection parameters
- [ ] Notification saat batch selesai
- [ ] Multi-language UI (ID/EN)
- [ ] PDF report generation
- [ ] Integration dengan Google Sheets
- [ ] Webhook notification (Telegram/Discord)
- [ ] Auto-update model dari cloud
- [ ] A/B testing antar model
- [ ] User authentication (multi-user)
- [ ] Detection confidence calibration
- [ ] Ensemble multiple models

---

## Changelog

### v1.0.0 (Current)
- Initial release
- YOLOv8n model (253 classes)
- Streamlit GUI with dark theme
- Multi-image detection
- Analytics dashboard
- History & export
- Training monitor

---

## Kontribusi

Jika ingin berkontribusi:
1. Fork repository
2. Buat branch baru (`feature/nama-fitur`)
3. Commit changes
4. Push & buat Pull Request

**Area yang butuh bantuan:**
- Dataset labeling (lebih banyak data = model lebih akurat)
- Testing di berbagai jenis gambar aksara
- Riset transliterasi aksara Rejang → Latin
- UI/UX feedback
