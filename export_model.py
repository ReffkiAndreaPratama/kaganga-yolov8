import argparse
from ultralytics import YOLO
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(weights_path, format_type):
    logger.info(f"Loading model from {weights_path}")
    try:
        model = YOLO(weights_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    logger.info(f"Exporting model to '{format_type}' format...")
    # Export the model
    # Supported formats: 'onnx', 'engine' (TensorRT - great for deployment), 'tflite', etc.
    try:
        path = model.export(format=format_type)
        logger.info(f"Export completed successfully. Model saved to: {path}")
    except Exception as e:
        logger.error(f"Export failed: {e}")
        logger.info("Note: Exporting to 'engine' requires TensorRT to be installed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 model for deployment")
    parser.add_argument("--weights", type=str, default="runs/detect/train_optimized/weights/best.pt", help="Path to best.pt")
    parser.add_argument("--format", type=str, default="onnx", 
                        choices=['onnx', 'engine', 'tflite', 'torchscript', 'openvino', 'pb', 'ncnn'], 
                        help="Target export format")
    args = parser.parse_args()
    
    main(args.weights, args.format)
