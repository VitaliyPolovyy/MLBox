import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import shutil

from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent

def convert_yolo_to_mask(yolo_label_path: Path, img_width: int, img_height: int) -> np.ndarray:
    """
    Convert YOLO polygon annotation to binary mask.
    
    Args:
        yolo_label_path: Path to YOLO .txt label file
        img_width: Width of the image
        img_height: Height of the image
    
    Returns:
        Binary mask as numpy array (0 = background, 255 = foreground)
    """
    # Create a PIL Image and ImageDraw object for polygon filling
    mask_img = Image.new('L', (img_width, img_height), 0)
    draw = ImageDraw.Draw(mask_img)
    
    if not yolo_label_path.exists():
        return np.array(mask_img)
    
    with open(yolo_label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            # Skip class_id (first value), get normalized coordinates
            coords = list(map(float, parts[1:]))
            points = np.array(coords).reshape(-1, 2)
            
            # Convert normalized coordinates to pixel coordinates
            points[:, 0] *= img_width   # Scale x coordinates
            points[:, 1] *= img_height  # Scale y coordinates
            points = points.astype(np.int32)
            
            # Convert to tuple list for PIL ImageDraw
            polygon_points = [(int(p[0]), int(p[1])) for p in points]
            
            # Fill polygon in mask
            draw.polygon(polygon_points, fill=255)
    
    return np.array(mask_img)


def convert_dataset_split(
    input_split_dir: Path,
    output_split_dir: Path,
    split_name: str
) -> None:
    """
    Convert one split (train/val/test) from YOLO to U-Net format.
    
    Args:
        input_split_dir: Path to input split directory (e.g., for-training-arch/train)
        output_split_dir: Path to output split directory (e.g., for-training-arch-mask/train)
        split_name: Name of the split for logging
    """
    input_images_dir = input_split_dir / "images"
    input_labels_dir = input_split_dir / "labels"
    
    output_images_dir = output_split_dir / "images"
    output_masks_dir = output_split_dir / "masks"
    
    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_masks_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image files
    image_files = list(input_images_dir.glob("*.jpg"))
    
    print(f"Processing {split_name}: {len(image_files)} images")
    
    for img_path in image_files:
        # Load image to get dimensions
        img = np.array(Image.open(img_path))
        img_height, img_width = img.shape[:2]
        
        # Copy image to output
        output_img_path = output_images_dir / img_path.name
        shutil.copy2(img_path, output_img_path)
        
        # Convert YOLO label to mask
        label_path = input_labels_dir / (img_path.stem + '.txt')
        mask = convert_yolo_to_mask(label_path, img_width, img_height)
        
        # Save mask as PNG
        mask_path = output_masks_dir / (img_path.stem + '.png')
        Image.fromarray(mask).save(mask_path)
    
    print(f"Completed {split_name}: {len(image_files)} images processed")


def convert_yolo_to_unet_dataset(
    input_dataset_dir: Path,
    output_dataset_dir: Path
) -> None:
    """
    Convert entire YOLO dataset to U-Net format.
    
    Args:
        input_dataset_dir: Path to input dataset (e.g., for-training-arch)
        output_dataset_dir: Path to output dataset (e.g., for-training-arch-mask)
    """
    output_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        input_split_dir = input_dataset_dir / split_name
        output_split_dir = output_dataset_dir / split_name
        
        if not input_split_dir.exists():
            print(f"Warning: {split_name} split not found, skipping...")
            continue
        
        convert_dataset_split(input_split_dir, output_split_dir, split_name)
    
    print(f"\nDataset conversion complete!")
    print(f"Output saved to: {output_dataset_dir}")


if __name__ == "__main__":
    input_dataset = ROOT_DIR / "assets" / "peanuts" / "datasets" / "separated" / "for-training-arch"
    output_dataset = ROOT_DIR / "assets" / "peanuts" / "datasets" / "separated" / "for-training-arch-mask"
    
    convert_yolo_to_unet_dataset(input_dataset, output_dataset)

