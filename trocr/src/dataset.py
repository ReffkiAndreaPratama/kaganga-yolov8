import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
import numpy as np

class TrOCRDataset(Dataset):
    """
    Custom PyTorch Dataset for TrOCR.
    Memuat gambar langsung dari struktur folder kelas:
    dataset/train/<nama_kelas>/<gambar>.jpg
    (Sesuai hasil keluaran convert_dataset.py)
    """
    def __init__(self, root_dir, processor, max_target_length=32, transform=None):
        self.root_dir = root_dir
        self.processor = processor
        self.max_target_length = max_target_length
        self.transform = transform
        
        self.samples = []
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root_dir}\nPastikan Anda sudah mengunduh/mengekstrak dataset Anda ke folder tersebut.")
            
        # Telusuri subfolder (setiap nama folder adalah label teks aksara, misal: 'ka', 'ga', 'ba')
        for class_name in sorted(os.listdir(root_dir)):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                        img_path = os.path.join(class_dir, file_name)
                        self.samples.append((img_path, class_name))
                        
        if len(self.samples) == 0:
            raise RuntimeError(f"Tidak ada gambar yang ditemukan di {root_dir}. Pastikan folder dataset memiliki subfolder kelas.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        # Terapkan augmentasi Albumentations jika ada
        if self.transform:
            image_np = np.array(image)
            transformed = self.transform(image=image_np)
            image_np = transformed["image"]
            image = Image.fromarray(image_np)

        # Proses gambar menggunakan ekstraktor fitur TrOCR
        pixel_values = self.processor(image, return_tensors="pt").pixel_values

        # Tokenisasi label teks aksara
        labels = self.processor.tokenizer(
            text, 
            padding="max_length", 
            max_length=self.max_target_length, 
            truncation=True
        ).input_ids

        # Penting: Ubah padding token menjadi -100 agar CrossEntropyLoss mengabaikannya
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        return {
            "pixel_values": pixel_values.squeeze(), 
            "labels": torch.tensor(labels)
        }

def get_train_transforms(config):
    """Mengembalikan pipeline augmentasi Albumentations berdasarkan config."""
    aug_cfg = config['augmentations']
    return A.Compose([
        A.Rotate(limit=aug_cfg['rotation_limit'], p=0.5, border_mode=0),
        A.RandomBrightnessContrast(p=aug_cfg['brightness_contrast_p']),
        A.GaussNoise(p=aug_cfg['noise_p']),
        A.MotionBlur(blur_limit=aug_cfg['blur_limit'], p=0.2),
        A.Perspective(scale=(0.05, 0.1), p=aug_cfg['perspective_p']),
        A.CoarseDropout(num_holes_range=(1, 1), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.2), # Random erasing
    ])
