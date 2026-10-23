import argparse
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Loading model from {model_path} on {device}...")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path).to(device)
    model.eval()

    logger.info(f"Processing image {image_path}...")
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)

    logger.info("Generating text prediction...")
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if "cuda" in str(device) else "cpu"):
            generated_ids = model.generate(pixel_values)
            
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    logger.info(f"--- PREDICTION RESULT ---")
    logger.info(f"Text: {generated_text}")
    print(f"\n====================================")
    print(f"Extracted Text: {generated_text}")
    print(f"====================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TrOCR Prediction")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="../models/checkpoints/best_model", help="Path to trained model")
    args = parser.parse_args()
    
    main(args.image, args.model)
