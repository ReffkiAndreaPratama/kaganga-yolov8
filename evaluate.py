import argparse
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(weights_path, data_yaml, imgsz):
    logger.info(f"Loading model from {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    logger.info("Starting evaluation on validation set...")
    # Validate the model
    # YOLOv8 automatically generates Confusion Matrix, Precision curve, Recall curve, PR curve, etc.
    # Results will be saved in runs/detect/val
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=0.25, # Confidence threshold
        iou=0.6,   # NMS IOU threshold
        split='val',
        plots=True # Generate evaluation plots
    )
    
    logger.info("Evaluation completed.")
    # Extract and log key metrics
    logger.info(f"mAP50-95: {metrics.box.map:.4f}")
    logger.info(f"mAP50: {metrics.box.map50:.4f}")
    logger.info(f"mAP75: {metrics.box.map75:.4f}")
    
    # Save directory
    save_dir = getattr(metrics, 'save_dir', 'runs/detect/val')
    logger.info(f"Plots and complete metrics saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 model for Kaganga/Katakana OCR")
    parser.add_argument("--weights", type=str, default="runs/detect/train_optimized/weights/best.pt", help="Path to trained best.pt")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to dataset YAML")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for evaluation")
    args = parser.parse_args()
    
    main(args.weights, args.data, args.imgsz)
