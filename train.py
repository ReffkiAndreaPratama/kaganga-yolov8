import yaml
from ultralytics import YOLO
import logging
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="config.yaml"):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    logger.info("Starting YOLOv8 Optimized Training Pipeline")
    
    # Load configuration
    try:
        config = load_config()
        logger.info("Loaded configuration successfully.")
    except Exception as e:
        logger.error(f"Failed to load config.yaml: {e}")
        return

    model_name = config['model']['name']
    data_yaml = config['dataset']['yaml_path']
    
    # Check if dataset yaml exists, if not, download from Roboflow
    if not os.path.exists(data_yaml):
        logger.info(f"Dataset YAML tidak ditemukan di {data_yaml}. Mencoba mengunduh dari Roboflow...")
        try:
            from roboflow import Roboflow
            
            # API Key dari user
            rf = Roboflow(api_key="WXAik1Lmv3OyGUObmQAk")
            project = rf.workspace("novalrizkiansyah-ymail-com").project("aksara-ulu-rejang")
            
            # Download versi 4 dengan format yolov8
            dataset = project.version(4).download("yolov8")
            
            # Update data_yaml location
            data_yaml = os.path.join(dataset.location, "data.yaml")
            logger.info(f"Dataset berhasil diunduh ke: {dataset.location}")
            
            # Fix absolute paths di data.yaml
            with open(data_yaml, 'r') as f:
                d_yaml = yaml.safe_load(f)
                
            base = os.path.abspath(dataset.location)
            d_yaml['train'] = os.path.join(base, 'train', 'images')
            d_yaml['val'] = os.path.join(base, 'valid', 'images')
            if 'test' in d_yaml:
                d_yaml['test'] = os.path.join(base, 'test', 'images')
                
            with open(data_yaml, 'w') as f:
                yaml.dump(d_yaml, f, default_flow_style=False)
                
            logger.info("Path pada data.yaml berhasil diupdate ke absolute path.")
            
        except ImportError:
            logger.error("Paket 'roboflow' belum terinstall. Silakan jalankan: pip install roboflow")
            return
        except Exception as e:
            logger.error(f"Gagal mengunduh dataset: {e}")
            return
        
    logger.info(f"Initializing model: {model_name}")
    # Load a pretrained model for transfer learning
    model = YOLO(model_name)

    logger.info("Starting training...")
    
    # Extract training and augmentation parameters
    train_params = config['training']
    aug_params = config['augmentations']
    
    # Train the model with specified parameters and modern augmentations
    # Note: Mixed precision FP16 is handled automatically by ultralytics via amp=True
    results = model.train(
        data=data_yaml,
        epochs=train_params['epochs'],
        imgsz=train_params['imgsz'],
        batch=train_params['batch_size'],
        patience=train_params['patience'],
        device=train_params['device'],
        workers=train_params['workers'],
        optimizer=train_params['optimizer'],
        cos_lr=train_params['cos_lr'],
        lr0=train_params['lr0'],
        amp=train_params['amp'],
        cache=train_params['cache'],
        # Augmentations for improving recall and handling variations
        hsv_h=aug_params['hsv_h'],
        hsv_s=aug_params['hsv_s'],
        hsv_v=aug_params['hsv_v'],
        degrees=aug_params['degrees'],
        translate=aug_params['translate'],
        scale=aug_params['scale'],
        shear=aug_params['shear'],
        perspective=aug_params['perspective'],
        flipud=aug_params['flipud'],
        fliplr=aug_params['fliplr'],
        mosaic=aug_params['mosaic'],
        mixup=aug_params['mixup'],
        erasing=aug_params['erasing'],
        project="runs/detect",
        name="train_optimized",
        exist_ok=True # Overwrite existing experiment with same name to keep folder clean
    )
    
    logger.info("Training completed successfully.")
    # The results.save_dir points to the run directory (e.g., runs/detect/train_optimized)
    save_dir = getattr(results, 'save_dir', 'runs/detect/train_optimized')
    logger.info(f"Best model saved to: {save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()
