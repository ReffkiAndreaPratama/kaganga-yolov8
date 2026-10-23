"""
train.py — Training YOLOv8 untuk deteksi Aksara Ulu Rejang.

Jalankan dari folder src/:
    python train.py
    python train.py --model m        # pakai YOLOv8m (lebih akurat)
    python train.py --epochs 100
    python train.py --resume         # lanjut training dari checkpoint
"""

import os
import sys
import argparse
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import utils


def main():
    parser = argparse.ArgumentParser(description="Training YOLOv8 Aksara Ulu Rejang")
    parser.add_argument("--model",   default=config.MODEL_SIZE,
                        choices=["n", "s", "m", "l", "x"],
                        help="Ukuran model: n=nano, s=small, m=medium, l=large, x=xlarge")
    parser.add_argument("--epochs",  type=int, default=config.EPOCHS)
    parser.add_argument("--imgsz",   type=int, default=config.IMG_SIZE)
    parser.add_argument("--batch",   type=int, default=config.BATCH_SIZE)
    parser.add_argument("--resume",  action="store_true",
                        help="Lanjut training dari checkpoint terakhir")
    args = parser.parse_args()

    # Validasi dataset
    if not os.path.isfile(config.DATA_YAML):
        print(f"[ERROR] data.yaml tidak ditemukan: {config.DATA_YAML}")
        sys.exit(1)

    utils.ensure_dirs()

    # Import YOLO
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics belum terinstall.")
        print("Jalankan: pip install ultralytics")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  Training YOLOv8{args.model} — Aksara Ulu Rejang")
    print(f"  Epochs   : {args.epochs}")
    print(f"  Img Size : {args.imgsz}")
    print(f"  Batch    : {args.batch}")
    print(f"  Device   : {config.DEVICE}")
    print("=" * 60 + "\n")

    # Load model
    if args.resume and os.path.isfile(config.MODEL_LAST_PATH):
        print(f"[train] Resume dari: {config.MODEL_LAST_PATH}")
        model = YOLO(config.MODEL_LAST_PATH)
    else:
        model_name = f"yolov8{args.model}.pt"
        print(f"[train] Load pretrained: {model_name}")
        model = YOLO(model_name)

    # Training
    results = model.train(
        data      = config.DATA_YAML,
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        patience  = config.PATIENCE,
        device    = config.DEVICE,
        workers   = config.WORKERS,
        project   = config.RUNS_DIR,
        name      = config.TRAIN_RUN_NAME,
        exist_ok  = True,
        verbose   = True,
    )

    # Salin model terbaik ke models/
    run_dir  = os.path.join(config.RUNS_DIR, config.TRAIN_RUN_NAME)
    best_src = os.path.join(run_dir, "weights", "best.pt")
    last_src = os.path.join(run_dir, "weights", "last.pt")

    if os.path.isfile(best_src):
        shutil.copy2(best_src, config.MODEL_BEST_PATH)
        print(f"[train] Model terbaik → {config.MODEL_BEST_PATH}")

    if os.path.isfile(last_src):
        shutil.copy2(last_src, config.MODEL_LAST_PATH)

    # Simpan label map
    label_map = utils.load_label_map_from_yaml()
    utils.save_label_map(label_map)

    # Simpan metrics
    metrics_dict = {}
    if hasattr(results, "results_dict"):
        metrics_dict = {k: float(v) for k, v in results.results_dict.items()
                        if isinstance(v, (int, float))}
    utils.save_metrics({"yolov8": metrics_dict})

    # Plot training curves
    results_csv = os.path.join(run_dir, "results.csv")
    if os.path.isfile(results_csv):
        utils.plot_training_results(results_csv)
        # Salin ke results/
        shutil.copy2(results_csv,
                     os.path.join(config.RESULT_DIR, "results.csv"))

    # Salin confusion matrix jika ada
    for fname in ["confusion_matrix.png", "confusion_matrix_normalized.png",
                  "PR_curve.png", "F1_curve.png", "P_curve.png", "R_curve.png"]:
        src = os.path.join(run_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(config.RESULT_DIR, fname))

    print("\n" + "=" * 60)
    print("  Training selesai!")
    print(f"  Model  → {config.MODEL_BEST_PATH}")
    print(f"  Hasil  → {config.RESULT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
