import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent

# ==================== Loss Functions ====================

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1.0
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return self.dice_loss(pred, target) + self.bce_loss(pred, target)

# ==================== Dataset Class ====================

class PeanutDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.jpg")))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / (img_path.stem + '.png')
        
        # Load image and mask
        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'))
        
        # Normalize mask to 0-1
        mask = (mask > 127).astype(np.float32)
        
        # Apply augmentations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Add channel dimension to mask if needed
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        
        return image, mask

# ==================== Metric Functions ====================

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_dice(pred_mask, true_mask):
    """
    Calculate Dice coefficient.
    
    Args:
        pred_mask: Binary prediction (bool or 0/1)
        true_mask: Ground truth binary mask (bool or 0/1)
    
    Returns:
        Dice score (0.0 to 1.0)
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()
    
    if pred_sum + true_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (pred_sum + true_sum)
    return float(dice)

# ==================== Training Function ====================

def train_unet(
    train_images_dir,
    train_masks_dir,
    val_images_dir,
    val_masks_dir,
    experiment_dir,
    epochs=100,
    batch_size=16,
    learning_rate=1e-4,
    device=None,
    patience=20
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Print device info
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        print("Using CPU (CUDA not available)")
    
    experiment_dir = Path(experiment_dir)
    weights_dir = experiment_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Data augmentation for training
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, p=0.7),
        A.RandomBrightnessContrast(p=0.8),
        A.HueSaturationValue(p=0.7),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(p=0.3),
        A.ElasticTransform(p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Validation transform (no augmentation)
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    train_dataset = PeanutDataset(train_images_dir, train_masks_dir, transform=train_transform)
    val_dataset = PeanutDataset(val_images_dir, val_masks_dir, transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = CombinedLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_dice_scores = []
        val_iou_scores = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate Dice/IoU per batch
                preds = torch.sigmoid(outputs) > 0.5
                for pred, mask in zip(preds, masks):
                    pred_np = pred.cpu().numpy()[0]
                    mask_np = mask.cpu().numpy()[0]
                    
                    pred_binary = pred_np > 0.5
                    mask_binary = mask_np > 0.5
                    
                    dice = calculate_dice(pred_binary, mask_binary)
                    iou = calculate_iou(pred_binary, mask_binary)
                    
                    val_dice_scores.append(dice)
                    val_iou_scores.append(iou)
        
        val_loss /= len(val_loader)
        avg_val_dice = np.mean(val_dice_scores) if val_dice_scores else 0.0
        avg_val_iou = np.mean(val_iou_scores) if val_iou_scores else 0.0
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Dice: {avg_val_dice:.4f}, "
              f"Val IoU: {avg_val_iou:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), weights_dir / "best.pth")
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model


if __name__ == "__main__":
    # Dataset paths (U-Net format)
    dataset_dir = ROOT_DIR / "assets" / "peanuts" / "datasets" / "separated" / "for-training-arch-mask"
    
    train_images_dir = dataset_dir / "train" / "images"
    train_masks_dir = dataset_dir / "train" / "masks"
    val_images_dir = dataset_dir / "val" / "images"
    val_masks_dir = dataset_dir / "val" / "masks"
    
    # Experiment folder (set manually)
    experiment_dir = CURRENT_DIR / "experiment1"
    
    # Training
    print("Starting U-Net training...")
    model = train_unet(
        train_images_dir=train_images_dir,
        train_masks_dir=train_masks_dir,
        val_images_dir=val_images_dir,
        val_masks_dir=val_masks_dir,
        experiment_dir=experiment_dir,
        epochs=100,
        batch_size=16,
        learning_rate=1e-4,
        patience=20
    )
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {experiment_dir / 'weights' / 'best.pth'}")

