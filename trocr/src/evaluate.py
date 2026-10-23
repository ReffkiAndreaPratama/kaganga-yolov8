import torch
import evaluate
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import TrOCRDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="../config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    config = load_config()
    model_path = config['training']['save_dir'] + "/best_model"
    
    logger.info(f"Loading model from {model_path}")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    logger.info("Loading test dataset...")
    test_dataset = TrOCRDataset(
        root_dir=config['dataset']['test_dir'],
        processor=processor,
        max_target_length=config['model']['max_length']
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=config['training']['eval_batch_size'], shuffle=False)

    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    predictions = []
    references = []

    logger.info("Evaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Predict
            generated_ids = model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Decode references
            labels[labels == -100] = processor.tokenizer.pad_token_id
            reference_text = processor.batch_decode(labels, skip_special_tokens=True)
            
            predictions.extend(generated_text)
            references.extend(reference_text)

    # Compute metrics
    cer = cer_metric.compute(predictions=predictions, references=references)
    wer = wer_metric.compute(predictions=predictions, references=references)

    # Calculate Character Accuracy
    accuracy = 1.0 - cer

    logger.info(f"--- Test Set Evaluation Results ---")
    logger.info(f"Character Error Rate (CER): {cer:.4f}")
    logger.info(f"Word Error Rate (WER): {wer:.4f}")
    logger.info(f"Character Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
