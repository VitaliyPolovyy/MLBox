from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import json

from unstructured_inference.inference.layout import DocumentLayout
from unstructured_inference.models.base import get_model
from mlbox.settings import ROOT_DIR
from mlbox.utils.logger import get_logger

# Configuration
INPUT_DIR = Path("/home/polovyi/projects/mlbox/assets/labelguard/datasets/2025-10-16-complex-layout")
OUTPUT_DIR = INPUT_DIR / "output_ms"

# Logger
app_logger = get_logger(ROOT_DIR)

def calculate_intersection_area(box1, box2):
    """Calculate intersection area between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    return (x_right - x_left) * (y_bottom - y_top)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    intersection = calculate_intersection_area(box1, box2)
    
    if intersection == 0:
        return 0.0
    
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def draw_text_blocks_on_image(image: Image.Image, text_blocks) -> Image.Image:
    """Draw rectangles and labels on detected text blocks"""
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for i, block in enumerate(text_blocks):
        x_min, y_min, x_max, y_max = block['bbox']
        score = block.get('score', 0)
        
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
        label = f"{i}:{score:.2f}" if score > 0 else f"{i}"
        cv2.putText(img_cv, label,
                   (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def main():
    """Process all JPEG images using Microsoft's Layout Analysis"""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    app_logger.info("layoutguard", f"Output directory: {OUTPUT_DIR}")
    
    app_logger.info("layoutguard", "Initializing YOLOX Tiny model for layout detection")
    detection_model = get_model("yolox_tiny")
    app_logger.info("layoutguard", "YOLOX Tiny model loaded successfully")
    
    # Get all JPEG files
    jpeg_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    jpeg_files = []
    for pattern in jpeg_patterns:
        jpeg_files.extend(INPUT_DIR.glob(pattern))
    
    app_logger.info("layoutguard", f"Found {len(jpeg_files)} images to process")
    
    # Process each image
    for img_path in jpeg_files:
        app_logger.info("layoutguard", f"\nProcessing: {img_path.name}")
        
        # Read image as PIL
        image = Image.open(img_path)
        
        # Detect layout using YOLOX Tiny model
        app_logger.info("layoutguard", "Running layout detection + OCR...")
        layout = DocumentLayout.from_image_file(str(img_path), detection_model=detection_model)
        
        # Extract text blocks
        text_blocks = []
        blocks_info = []
        
        for idx, element in enumerate(layout.pages[0].elements):
            # Get bounding box
            bbox = element.bbox
            x_min, y_min, x_max, y_max = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
            
            # Crop the block
            cropped = image.crop((x_min, y_min, x_max, y_max))
            
            # Get confidence if available
            score = getattr(element, 'prob', 0)
            
            # Get text content
            text = getattr(element, 'text', '') or ''
            
            # Get category
            category = getattr(element, 'type', 'unknown')
            
            block_data = {
                'bbox': (x_min, y_min, x_max, y_max),
                'score': score,
                'type': category,
                'text': text,
                'image_crop': cropped
            }
            
            text_blocks.append(block_data)
            
            # Log block info
            text_preview = text[:50] + "..." if text and len(text) > 50 else text
            app_logger.info("layoutguard", 
                f"  Block {idx} [{category}]: \"{text_preview}\" (score: {score:.2f})")
            
            # Store for JSON
            blocks_info.append({
                'index': idx,
                'category': category,
                'text': text,
                'bbox': [x_min, y_min, x_max, y_max],
                'score': float(score)
            })
        
        app_logger.info("layoutguard", f"\nDetected {len(text_blocks)} text blocks")
        
        # Save JSON with block information
        img_stem = img_path.stem
        json_path = OUTPUT_DIR / f"{img_stem}_blocks.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'image': img_path.name,
                'total_blocks': len(blocks_info),
                'blocks': blocks_info
            }, f, indent=2, ensure_ascii=False)
        
        app_logger.info("layoutguard", f"Saved block data to {json_path.name}")
        
        # Save individual blocks with full image size
        for i, block in enumerate(text_blocks):
            # Create white background with original image size
            block_img = Image.new('RGB', image.size, (255, 255, 255))
            
            # Paste the cropped block at its original position
            x_min, y_min, x_max, y_max = block['bbox']
            block_img.paste(block['image_crop'], (x_min, y_min))
            
            # Create filename
            score = block.get('score', 0)
            block_filename = f"{img_stem}_block{i}_{score:.2f}.jpg"
            block_path = OUTPUT_DIR / block_filename
            
            block_img.save(block_path)
        
        app_logger.info("layoutguard", f"Saved {len(text_blocks)} blocks with full image context")
        
        # Draw text blocks on image
        output_image = draw_text_blocks_on_image(image, text_blocks)
        
        # Save output visualization
        output_path = OUTPUT_DIR / img_path.name
        output_image.save(output_path)
        
        app_logger.info("layoutguard", 
                       f"Saved visualization to {output_path.name}\n")

if __name__ == "__main__":
    main()

