"""
predict.py — Prediksi/deteksi aksara pada gambar via CLI.

Contoh:
    python predict.py --image ../sample.jpg
    python predict.py --batch ../folder_gambar/
    python predict.py --image ../sample.jpg --conf 0.3
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import utils


def load_model():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics belum terinstall. Jalankan: pip install ultralytics")
        sys.exit(1)

    if not os.path.isfile(config.MODEL_BEST_PATH):
        print(f"[ERROR] Model tidak ditemukan: {config.MODEL_BEST_PATH}")
        print("Jalankan train.py terlebih dahulu.")
        sys.exit(1)

    return YOLO(config.MODEL_BEST_PATH)


def predict_image(image_path: str, model, conf: float) -> list:
    """Prediksi satu gambar, kembalikan list deteksi."""
    results = model.predict(
        source    = image_path,
        conf      = conf,
        iou       = config.IOU_THRESHOLD,
        verbose   = False,
        save      = False,
    )

    detections = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        names = results[0].names
        for box in boxes:
            cls_id = int(box.cls[0])
            detections.append({
                "class_name": names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox":       [float(x) for x in box.xyxy[0].tolist()],
            })

    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path ke satu gambar")
    parser.add_argument("--batch", help="Path ke folder gambar")
    parser.add_argument("--conf",  type=float, default=config.CONF_THRESHOLD)
    args = parser.parse_args()

    if not args.image and not args.batch:
        parser.print_help()
        sys.exit(1)

    model = load_model()

    images = []
    if args.image:
        images = [args.image]
    elif args.batch:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [
            os.path.join(args.batch, f)
            for f in os.listdir(args.batch)
            if os.path.splitext(f)[1].lower() in exts
        ]

    for img_path in images:
        if not os.path.isfile(img_path):
            print(f"[SKIP] {img_path}")
            continue

        detections = predict_image(img_path, model, args.conf)
        utils.log_prediction(os.path.basename(img_path), detections)

        print("\n" + "=" * 55)
        print(f"  Gambar     : {os.path.basename(img_path)}")
        print(f"  Deteksi    : {len(detections)} aksara")
        print("-" * 55)
        for i, det in enumerate(detections, 1):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            print(f"  [{i}] {det['class_name']:<12} "
                  f"conf={det['confidence']:.2f}  "
                  f"bbox=({x1},{y1},{x2},{y2})")
        print("=" * 55)


if __name__ == "__main__":
    main()
