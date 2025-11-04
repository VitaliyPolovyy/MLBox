from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from mlbox.services.LabelGuard.layout_detector import LayoutDetector
from mlbox.settings import ROOT_DIR
from mlbox.utils.logger import get_logger

# Configuration
INPUT_DIR = Path("/home/polovyi/projects/mlbox/assets/labelguard/datasets/2025-10-16-complex-layout")
OUTPUT_DIR = INPUT_DIR / "output_paddle"

# Parameters
MAX_LOOPS = 20
SCORE_THRESH = 0.4


# Logger
app_logger = get_logger(ROOT_DIR)
    
def calculate_intersection_area(box1, box2):
    """Calculate intersection area between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection coordinates
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    # Check if there's an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    return (x_right - x_left) * (y_bottom - y_top)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes"""
    intersection = calculate_intersection_area(box1, box2)
    
    if intersection == 0:
        return 0.0
    
    # Calculate areas of both boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calculate union
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def draw_text_blocks_on_image(image: Image.Image, text_blocks) -> Image.Image:
    """Draw rectangles and labels on detected text blocks"""
    # Convert PIL to cv2 for drawing
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Draw each text block
    for i, block in enumerate(text_blocks):
        x_min, y_min, x_max, y_max = block.bbox
        score = block.score
        
        # Draw rectangle
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
        # Draw label with score
        cv2.putText(img_cv, f"{i}:{score:.2f}",
                   (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Convert back to PIL
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def main():
    """Process all JPEG images and create visualizations with detected text blocks"""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    app_logger.info("layoutguard", f"Output directory: {OUTPUT_DIR}")
    
    # Initialize layout detector
    app_logger.info("layoutguard", "Initializing LayoutDetector with PP-DocLayout_plus-L")
    detector = LayoutDetector(model_name="PP-DocLayout_plus-L")
    
    # Get all JPEG files
    jpeg_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    jpeg_files = []
    for pattern in jpeg_patterns:
        jpeg_files.extend(INPUT_DIR.glob(pattern))
    
    app_logger.info("layoutguard", f"Found {len(jpeg_files)} images to process")
    
    # Process each image
    for img_path in jpeg_files:
        app_logger.info("layoutguard", f"Processing: {img_path.name}")
        
        # Read image as PIL
        image = Image.open(img_path)
        
        # Detect text blocks
        text_blocks = detector.extract_blocks(
            image, 
            max_loops=MAX_LOOPS, 
            score_thresh=SCORE_THRESH
        )
        
        # Save individual blocks with their bounding box size
        img_stem = img_path.stem  # filename without extension
        for i, block in enumerate(text_blocks):
            # Create filename: originalname_block{i}_{score}.jpg
            block_filename = f"{img_stem}_block{i}_{block.score:.2f}.jpg"
            block_path = OUTPUT_DIR / block_filename
            
            # Save cropped block image (size matches bounding box dimensions)
            block.image_crop.save(block_path)
        
        app_logger.info("layoutguard", f"Saved {len(text_blocks)} cropped blocks")
        
        # Draw text blocks on image
        output_image = draw_text_blocks_on_image(image, text_blocks)
        
        # Save output visualization
        output_path = OUTPUT_DIR / img_path.name
        output_image.save(output_path)
        
        app_logger.info("layoutguard", 
                       f"Detected {len(text_blocks)} text blocks -> saved to {output_path.name}")

if __name__ == "__main__":
    main()

