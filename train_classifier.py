import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
DATA_DIR = r"d:\laragon\www\deteksi-aksara-kaganga-main\classifier_dataset"
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 0.001
IMAGE_SIZE = 128 # Increased from 64 to 128 to capture fine strokes, dots, and accents
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================

def get_data_loaders():
    # Advanced augmentations for handwritten Kaganga script
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(15), # Random rotation
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Translation and scaling
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Light/contrast changes
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_dataset.classes

def build_model(num_classes):
    # Using ResNet34 instead of ResNet18 for much better capacity to learn 253 classes
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    
    # Modify the final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model.to(DEVICE)

def train():
    print(f"Using device: {DEVICE}")
    train_loader, val_loader, classes = get_data_loaders()
    num_classes = len(classes)
    print(f"Found {num_classes} classes.")
    
    # Save class names mapping
    import json
    with open('classifier_classes.json', 'w') as f:
        json.dump(classes, f)

    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Cosine learning rate scheduler for smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        print(f"\nEpoch {epoch+1}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f}")
        pbar = tqdm(train_loader, desc="Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})

        # Step scheduler
        scheduler.step()

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:
            print(f"Validation accuracy improved from {best_acc:.2f}% to {val_acc:.2f}%. Saving model...")
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_classifier.pth')

    print("Training complete!")

if __name__ == "__main__":
    train()
