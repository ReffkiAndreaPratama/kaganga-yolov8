import os
import cv2
import yaml
from pathlib import Path
from tqdm import tqdm

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def process_yolo_dataset(yaml_path, output_dir):
    data_config = load_yaml(yaml_path)
    classes = data_config['names']
    
    # Create base output directories
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        # Create class subdirectories
        for cls_name in classes:
            # Handle special characters in class name (like 'no')
            clean_cls_name = str(cls_name).strip("'")
            os.makedirs(os.path.join(split_dir, clean_cls_name), exist_ok=True)

    base_dir = os.path.dirname(yaml_path)

    for split, yolo_split in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        img_dir = data_config.get(yolo_split)
        if not img_dir:
            continue
            
        # Handle absolute or relative paths
        if not os.path.isabs(img_dir):
            img_dir = os.path.join(base_dir, img_dir)
            
        lbl_dir = img_dir.replace('images', 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            print(f"Skipping {split} - directory not found: {img_dir}")
            continue

        print(f"\nProcessing {split} dataset...")
        images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        crop_count = 0
        for img_name in tqdm(images):
            img_path = os.path.join(img_dir, img_name)
            lbl_path = os.path.join(lbl_dir, img_name.rsplit('.', 1)[0] + '.txt')
            
            if not os.path.exists(lbl_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            
            with open(lbl_path, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                    
                cls_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert YOLO format to pixel coordinates
                x1 = int((x_center - width/2) * w)
                y1 = int((y_center - height/2) * h)
                x2 = int((x_center + width/2) * w)
                y2 = int((y_center + height/2) * h)
                
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop_img = img[y1:y2, x1:x2]
                
                cls_name = str(classes[cls_id]).strip("'")
                save_path = os.path.join(output_dir, split, cls_name, f"{img_name.split('.')[0]}_crop_{i}.jpg")
                cv2.imwrite(save_path, crop_img)
                crop_count += 1
                
        print(f"Saved {crop_count} cropped images for {split}.")

if __name__ == "__main__":
    yolo_yaml = r"d:\laragon\www\deteksi-aksara-kaganga-main\Aksara-Ulu-Rejang-4\data.yaml"
    output_directory = r"d:\laragon\www\deteksi-aksara-kaganga-main\classifier_dataset"
    print("Starting to crop YOLO dataset into Classification format...")
    process_yolo_dataset(yolo_yaml, output_directory)
    print("\nDataset preparation complete! You can now run train_classifier.py")
