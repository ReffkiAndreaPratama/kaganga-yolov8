import os

# Base directory = root project (satu level di atas src/)
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_YAML       = os.path.join(BASE_DIR, "dataset", "data.yaml")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
RESULT_DIR      = os.path.join(BASE_DIR, "results")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
RUNS_DIR        = os.path.join(BASE_DIR, "runs")
MODEL_BEST_PATH = os.path.join(BASE_DIR, "models", "best.pt")
MODEL_LAST_PATH = os.path.join(BASE_DIR, "models", "last.pt")
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "models", "label_map.json")
METRICS_PATH    = os.path.join(BASE_DIR, "models", "metrics.json")
PREDICT_LOG     = os.path.join(BASE_DIR, "logs", "prediction_log.json")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

CONF_THRESHOLD  = 0.25
IOU_THRESHOLD   = 0.45
APP_NAME        = "AksaraDetect"
APP_VERSION     = "1.0.0"
APP_TAGLINE     = "Aksara Ulu Rejang - YOLOv8"
MODEL_SIZE      = "n"
EPOCHS          = 50
IMG_SIZE        = 640
BATCH_SIZE      = 32
PATIENCE        = 100
LR0             = 0.01
DEVICE          = "0"
WORKERS         = 8
SEED            = 0
TRAIN_RUN_NAME  = "aksara_ulu"
