"""
convert_dataset.py — Konversi dataset Roboflow multiclass CSV
ke struktur folder yang dibutuhkan flow_from_directory.

Jalankan dari folder kaganga-detector/:
    python convert_dataset.py

Hasilnya:
    dataset/train/<nama_kelas>/<gambar>.jpg
    dataset/test/<nama_kelas>/<gambar>.jpg
"""

import os
import shutil
import pandas as pd

# ── Konfigurasi ───────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MULTICLASS   = os.path.join(BASE_DIR, "Aksara Ulu Rejang.v4-fix.multiclass")

OUTPUT_TRAIN = os.path.join(BASE_DIR, "dataset", "train")
OUTPUT_TEST  = os.path.join(BASE_DIR, "dataset", "test")

# Untuk gambar yang punya BANYAK label (multi-label):
# "first"  → ambil label pertama yang bernilai 1
# "all"    → duplikat gambar ke semua folder kelas yang relevan (lebih banyak data)
MULTI_LABEL_STRATEGY = "all"


# ── Fungsi ────────────────────────────────────────────────────────────────────

def convert_split(csv_path: str, img_dir: str, output_dir: str, split_name: str):
    """Konversi satu split (train/test/valid) dari CSV ke struktur folder."""
    if not os.path.isfile(csv_path):
        print(f"[SKIP] CSV tidak ditemukan: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    filename_col = df.columns[0]   # kolom pertama = nama file
    class_cols   = df.columns[1:]  # kolom sisanya = nama kelas

    total   = 0
    skipped = 0

    for _, row in df.iterrows():
        filename = str(row[filename_col]).strip()
        src_path = os.path.join(img_dir, filename)

        if not os.path.isfile(src_path):
            skipped += 1
            continue

        # Cari kelas yang aktif (nilai = 1)
        active_classes = [col for col in class_cols if row[col] == 1]

        if not active_classes:
            skipped += 1
            continue

        if MULTI_LABEL_STRATEGY == "first":
            targets = [active_classes[0]]
        else:  # "all"
            targets = active_classes

        for cls in targets:
            dest_dir  = os.path.join(output_dir, cls)
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = os.path.join(dest_dir, filename)

            if not os.path.isfile(dest_path):
                shutil.copy2(src_path, dest_path)
            total += 1

    print(f"[{split_name}] {total} gambar dipindahkan, {skipped} dilewati")


def main():
    print("=" * 55)
    print("  Konversi Dataset Aksara Ulu Rejang")
    print("=" * 55)

    # Train
    convert_split(
        csv_path   = os.path.join(MULTICLASS, "train", "_classes.csv"),
        img_dir    = os.path.join(MULTICLASS, "train"),
        output_dir = OUTPUT_TRAIN,
        split_name = "TRAIN",
    )

    # Valid → gabung ke train (lebih banyak data training)
    convert_split(
        csv_path   = os.path.join(MULTICLASS, "valid", "_classes.csv"),
        img_dir    = os.path.join(MULTICLASS, "valid"),
        output_dir = OUTPUT_TRAIN,
        split_name = "VALID→TRAIN",
    )

    # Test
    convert_split(
        csv_path   = os.path.join(MULTICLASS, "test", "_classes.csv"),
        img_dir    = os.path.join(MULTICLASS, "test"),
        output_dir = OUTPUT_TEST,
        split_name = "TEST",
    )

    # Ringkasan
    print("\n" + "=" * 55)
    print("  Hasil konversi:")
    print(f"  Train → {OUTPUT_TRAIN}")
    print(f"  Test  → {OUTPUT_TEST}")
    print("=" * 55)

    # Hitung kelas dan gambar
    for split, path in [("Train", OUTPUT_TRAIN), ("Test", OUTPUT_TEST)]:
        if not os.path.isdir(path):
            continue
        classes = [d for d in os.listdir(path)
                   if os.path.isdir(os.path.join(path, d))]
        total_imgs = sum(
            len(os.listdir(os.path.join(path, c))) for c in classes
        )
        print(f"  {split}: {len(classes)} kelas, {total_imgs} gambar")

    print("\n  Sekarang jalankan:")
    print("  cd src && python train.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
