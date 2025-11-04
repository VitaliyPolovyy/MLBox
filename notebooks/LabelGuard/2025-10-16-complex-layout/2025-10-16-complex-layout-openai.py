"""
OpenAI GPT-4 Vision Layout Detection and OCR

This script uses OpenAI's GPT-4 Vision API for OCR and layout detection.
Requires OPENAI_API_KEY environment variable.
"""
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
import json
import os
import base64
import io
from openai import OpenAI

from mlbox.settings import ROOT_DIR
from mlbox.utils.logger import get_logger

# Configuration
INPUT_DIR = Path("/home/polovyi/projects/mlbox/assets/labelguard/datasets/2025-10-16-complex-layout")
OUTPUT_DIR = INPUT_DIR / "output_openai"

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

def parse_layout_response(response_text, image_width, image_height):
    """
    Parse the OpenAI response to extract text blocks with bounding boxes.
    Expected format could be JSON or structured text with coordinates.
    """
    text_blocks = []
    
    try:
        # Strip markdown code blocks if present
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0].strip()
        elif '```' in response_text:
            response_text = response_text.split('```')[1].split('```')[0].strip()
        
        # Try parsing as JSON first
        if '{' in response_text and '}' in response_text:
            # Extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            json_str = response_text[start_idx:end_idx]
            data = json.loads(json_str)
            
            # Handle different JSON structures
            if 'blocks' in data:
                blocks = data['blocks']
            elif 'regions' in data:
                blocks = data['regions']
            elif isinstance(data, list):
                blocks = data
            else:
                blocks = [data]
            
            for idx, block in enumerate(blocks):
                # Extract bounding box (could be in various formats)
                bbox = None
                if 'bbox' in block:
                    bbox = block['bbox']
                elif 'box' in block:
                    bbox = block['box']
                elif 'coordinates' in block:
                    coords = block['coordinates']
                    if isinstance(coords, list) and len(coords) == 4:
                        bbox = coords
                
                # Normalize bbox if needed (from 0-1000 to pixel coordinates)
                if bbox:
                    if all(isinstance(x, (int, float)) and x <= 1000 for x in bbox):
                        # Assuming normalized coordinates (0-1000)
                        x_min = int(bbox[0] * image_width / 1000)
                        y_min = int(bbox[1] * image_height / 1000)
                        x_max = int(bbox[2] * image_width / 1000)
                        y_max = int(bbox[3] * image_height / 1000)
                    else:
                        x_min, y_min, x_max, y_max = map(int, bbox)
                    
                    # Get text content
                    text = block.get('text', '') or block.get('content', '') or ''
                    category = block.get('type', 'text') or block.get('category', 'text')
                    score = block.get('confidence', 1.0)
                    
                    text_blocks.append({
                        'bbox': (x_min, y_min, x_max, y_max),
                        'text': text,
                        'type': category,
                        'score': score
                    })
        
        # If JSON parsing failed or no blocks found, try regex patterns
        if not text_blocks:
            import re
            # Pattern for coordinates like: [x1, y1, x2, y2] or (x1, y1, x2, y2)
            pattern = r'[\[\(](\d+),\s*(\d+),\s*(\d+),\s*(\d+)[\]\)]'
            matches = re.findall(pattern, response_text)
            
            for idx, match in enumerate(matches):
                x_min, y_min, x_max, y_max = map(int, match)
                text_blocks.append({
                    'bbox': (x_min, y_min, x_max, y_max),
                    'text': f'Block {idx}',
                    'type': 'text',
                    'score': 1.0
                })
    
    except Exception as e:
        app_logger.warning("layoutguard", f"Failed to parse layout response: {e}")
    
    return text_blocks

def draw_text_blocks_on_image(image: Image.Image, text_blocks) -> Image.Image:
    """Draw rectangles and labels on detected text blocks"""
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    for i, block in enumerate(text_blocks):
        x_min, y_min, x_max, y_max = block['bbox']
        score = block.get('score', 0)
        
        cv2.rectangle(img_cv, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
        label = f"{i}:{score:.2f}" if score < 1.0 else f"{i}"
        cv2.putText(img_cv, label,
                   (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

def detect_layout_with_openai(client, image_path):
    """
    Use OpenAI GPT-4 Vision for OCR and layout detection
    """
    # Load original image
    original_image = Image.open(image_path).convert('RGB')
    
    # Resize image to OpenAI's processing size (max dimension = 2048)
    max_dimension = 2048
    width, height = original_image.size
    
    if max(width, height) > max_dimension:
        scale_factor = max_dimension / max(width, height)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        app_logger.info("layoutguard", f"Resized image: {width}x{height} -> {new_width}x{new_height}")
    else:
        image = original_image
        new_width, new_height = width, height
    
    # Encode resized image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=95)
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Prepare the prompt
    prompt = f"""Analyze this document image and identify all text regions/blocks. For each text block, provide:
1. The bounding box coordinates (x_min, y_min, x_max, y_max) in pixels
2. The text content (OCR)
3. The type (e.g., title, paragraph, table, list, etc.)

Return ONLY a valid JSON object (no markdown, no extra text) in this exact format:
{{
  "blocks": [
    {{
      "bbox": [x_min, y_min, x_max, y_max],
      "text": "extracted text",
      "type": "paragraph"
    }}
  ]
}}

Image dimensions are {image.width}x{image.height} pixels."""
    
    # Call OpenAI Vision API
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4-vision-preview"
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=4096
        )
        
        response_text = response.choices[0].message.content
        app_logger.info("layoutguard", f"OpenAI response received (length: {len(response_text)})")
        
    except Exception as e:
        app_logger.error("layoutguard", f"OpenAI API call failed: {e}")
        response_text = ""
    
    # Parse the response (using resized image dimensions)
    text_blocks = parse_layout_response(response_text, image.width, image.height)
    
    # Return with both original and resized image info
    return text_blocks, response_text, original_image, image

def main():
    """Process all JPEG images using OpenAI GPT-4 Vision for layout detection and OCR"""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    app_logger.info("layoutguard", f"Output directory: {OUTPUT_DIR}")
    
    # Get OpenAI API key
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        app_logger.error("layoutguard", "OPENAI_API_KEY not found! Set it with: export OPENAI_API_KEY='your_key'")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)
    app_logger.info("layoutguard", "OpenAI client initialized successfully")
    app_logger.info("layoutguard", f"Using model: gpt-4o")
    
    # Get all JPEG files
    jpeg_patterns = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]
    jpeg_files = []
    for pattern in jpeg_patterns:
        jpeg_files.extend(INPUT_DIR.glob(pattern))
    
    app_logger.info("layoutguard", f"Found {len(jpeg_files)} images to process")
    
    # Process each image
    for img_path in jpeg_files:
        app_logger.info("layoutguard", f"\n{'='*80}")
        app_logger.info("layoutguard", f"Processing: {img_path.name}")
        
        try:
            # Detect layout and extract text
            text_blocks, raw_response, original_image, resized_image = detect_layout_with_openai(client, img_path)
            
            # Save raw response
            img_stem = img_path.stem
            response_path = OUTPUT_DIR / f"{img_stem}_response.txt"
            with open(response_path, 'w', encoding='utf-8') as f:
                f.write(raw_response)
            
            app_logger.info("layoutguard", f"\nRaw response saved to {response_path.name}")
            
            # Save resized image that was sent to OpenAI
            resized_image_path = OUTPUT_DIR / f"{img_stem}_resized.jpg"
            resized_image.save(resized_image_path, quality=95)
            app_logger.info("layoutguard", f"Resized image saved to {resized_image_path.name}")
            app_logger.info("layoutguard", f"\nDetected {len(text_blocks)} text blocks")
            
            # Log and save block information
            blocks_info = []
            for idx, block in enumerate(text_blocks):
                text_preview = block['text'][:80] + "..." if len(block['text']) > 80 else block['text']
                app_logger.info("layoutguard", 
                    f"  Block {idx} [{block['type']}]: \"{text_preview}\"")
                
                blocks_info.append({
                    'index': idx,
                    'category': block['type'],
                    'text': block['text'],
                    'bbox': list(block['bbox']),
                    'score': float(block['score'])
                })
            
            # Save JSON with block information
            json_path = OUTPUT_DIR / f"{img_stem}_blocks.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'image': img_path.name,
                    'total_blocks': len(blocks_info),
                    'blocks': blocks_info
                }, f, indent=2, ensure_ascii=False)
            
            app_logger.info("layoutguard", f"Saved block data to {json_path.name}")
            
            # Save individual blocks with full image size (using resized image)
            for i, block in enumerate(text_blocks):
                # Create white background with resized image size
                block_img = Image.new('RGB', resized_image.size, (255, 255, 255))
                
                # Crop and paste the block at its original position
                x_min, y_min, x_max, y_max = block['bbox']
                
                # Ensure coordinates are within resized image bounds
                x_min = max(0, min(x_min, resized_image.width))
                y_min = max(0, min(y_min, resized_image.height))
                x_max = max(0, min(x_max, resized_image.width))
                y_max = max(0, min(y_max, resized_image.height))
                
                if x_max > x_min and y_max > y_min:
                    cropped = resized_image.crop((x_min, y_min, x_max, y_max))
                    block_img.paste(cropped, (x_min, y_min))
                    
                    # Create filename
                    score = block.get('score', 1.0)
                    block_filename = f"{img_stem}_block{i}_{score:.2f}.jpg"
                    block_path = OUTPUT_DIR / block_filename
                    
                    block_img.save(block_path)
            
            app_logger.info("layoutguard", f"Saved {len(text_blocks)} blocks with full image context")
            
            # Draw text blocks on image (using resized image)
            if text_blocks:
                output_image = draw_text_blocks_on_image(resized_image, text_blocks)
                
                # Save output visualization
                output_path = OUTPUT_DIR / img_path.name
                output_image.save(output_path)
                
                app_logger.info("layoutguard", 
                               f"Saved visualization to {output_path.name}")
            else:
                app_logger.warning("layoutguard", "No blocks detected, skipping visualization")
        
        except Exception as e:
            app_logger.error("layoutguard", f"Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    app_logger.info("layoutguard", f"\n{'='*80}")
    app_logger.info("layoutguard", "Processing complete!")

if __name__ == "__main__":
    main()

