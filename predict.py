import argparse
from ultralytics import YOLO
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(weights_path, source, conf_thres, save_dir):
    logger.info(f"Loading model from {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Running inference on {source}")
    # Run prediction
    results = model.predict(
        source=source,
        conf=conf_thres,
        save=True,
        project=save_dir,
        name="predict_output",
        exist_ok=True
    )
    
    logger.info(f"Inference completed. Results saved to {os.path.join(save_dir, 'predict_output')}")
    
    # Print a summary of detections
    for i, result in enumerate(results):
        boxes = result.boxes
        file_path = result.path
        filename = os.path.basename(file_path)
        logger.info(f"Image [{filename}]: Found {len(boxes)} characters.")
        
        # Optional: Print detailed class info
        # if len(boxes) > 0:
        #     for box in boxes:
        #         cls_id = int(box.cls[0])
        #         conf = float(box.conf[0])
        #         logger.info(f" - Detected {model.names[cls_id]} with confidence {conf:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 static inference")
    parser.add_argument("--weights", type=str, default="runs/detect/train_optimized/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--source", type=str, required=True, help="Path to image, video, or directory to predict on")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--save_dir", type=str, default="runs/inference", help="Directory to save visual results")
    args = parser.parse_args()
    
    main(args.weights, args.source, args.conf, args.save_dir)
