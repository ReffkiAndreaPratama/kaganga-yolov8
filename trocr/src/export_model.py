import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(model_path, output_dir):
    """
    To export TrOCR (VisionEncoderDecoder) to ONNX, the HuggingFace Optimum library is highly recommended.
    This script uses optimum-cli to perform the export automatically.
    """
    logger.info("Exporting TrOCR model to ONNX using HuggingFace Optimum...")
    logger.info("Please ensure optimum is installed: pip install optimum[onnxruntime]")
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Run optimum-cli for VisionEncoderDecoder export
    command = f"optimum-cli export onnx --model {model_path} --task image-to-text {output_dir}"
    logger.info(f"Running command: {command}")
    
    result = os.system(command)
    if result == 0:
        logger.info(f"Export successful! ONNX model saved to {output_dir}")
        logger.info("You can now use this ONNX model for faster inference without PyTorch overhead.")
    else:
        logger.error("Export failed. Make sure you have installed optimum: `pip install optimum[onnxruntime]`")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TrOCR to ONNX")
    parser.add_argument("--model", type=str, default="../models/checkpoints/best_model", help="Path to trained model")
    parser.add_argument("--output", type=str, default="../models/onnx", help="Output directory")
    args = parser.parse_args()
    
    main(args.model, args.output)
