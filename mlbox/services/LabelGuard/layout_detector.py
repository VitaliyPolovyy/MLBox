from dataclasses import dataclass
from typing import List, Union, Optional
from PIL import Image
import cv2
import numpy as np
from paddleocr import LayoutDetection
from mlbox.settings import ROOT_DIR
from mlbox.utils.logger import get_logger, get_artifact_service
from pathlib import Path


CURRENT_DIR = Path(__file__).parent
SERVICE_NAME = "labelguard"
app_logger = get_logger(ROOT_DIR)
artifact_service = get_artifact_service(ROOT_DIR)

@dataclass
class LayoutTextBlock:
    bbox: tuple  # (x_min, y_min, x_max, y_max)
    image_crop: Image.Image
    score: float = 0.0
    label: str = "text"
    index: int = 0


class LayoutDetector:
    def __init__(self, model_name="PP-DocLayout_plus-L"):
        """Initialize layout detection model using PaddleOCR only"""
        self.layout_model = LayoutDetection(model_name=model_name)
        
    
    def extract_blocks(self, image_input: Image.Image, 
                      max_loops: int = 5, 
                      score_thresh: float = 0.4) -> List[LayoutTextBlock]:
        """
        Extract text blocks from image using iterative detection and removal.
        
        Args:
            image_input: Either image path (str) or PIL Image
            max_loops: Maximum iterations for text block detection
            score_thresh: Minimum confidence score for detections
            
        Returns:
            List of LayoutTextBlock objects with bounding boxes and cropped images
        """
        # Convert PIL Image to OpenCV format if needed
        if isinstance(image_input, Image.Image):
            # Convert PIL Image to RGB if it's not already
            if image_input.mode != 'RGB':
                image_input = image_input.convert('RGB')
            # Convert to numpy array and then to BGR for OpenCV
            img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(image_input)
            if img is None:
                raise ValueError(f"Cannot read image: {image_input}")

        all_blocks = []

        for loop in range(max_loops):
            predictions = self.layout_model.predict(
                img, 
                batch_size=1,
                layout_nms=True,
                layout_unclip_ratio=1.01,
                threshold=score_thresh
            )
            
            if not predictions:
                break

            page = predictions[0]
            blocks = [b for b in page['boxes'] if b['label'] == 'text']

            if not blocks:
                break

            for i, block in enumerate(blocks):
                x_min, y_min, x_max, y_max = map(int, block['coordinate'])
                
                # Crop the text block from original image
                cropped_cv = img[y_min:y_max, x_min:x_max]
                
                # Convert back to PIL Image
                cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
                
                # Create LayoutTextBlock object
                text_block = LayoutTextBlock(
                    bbox=(x_min, y_min, x_max, y_max),
                    image_crop=cropped_pil,
                    index=i,
                    score=block['score'],
                    label=block['label']
                )
                
                all_blocks.append(text_block)

                # Paint white rectangle on original image (for stripping)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)

        if all_blocks:
            app_logger.info("layoutguard", f"Detected {len(all_blocks)} text blocks")
        
        return all_blocks
    
    def strip_text_blocks(self, image_path: str, max_loops: int = 5, score_thresh: float = 0.4):
        """
        Iteratively detect and remove text blocks from an image.
        This method matches the Google Colab implementation exactly.
        
        Args:
            image_path: Path to the image file
            max_loops: Maximum iterations for text block detection
            score_thresh: Minimum confidence score for detections
            
        Returns:
            tuple: (all_blocks, stripped_img) where all_blocks is a list of detected blocks
                   and stripped_img is the image with text blocks painted white
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        all_blocks = []

        for loop in range(max_loops):
            predictions = self.layout_model.predict(
                img, batch_size=1,
                layout_nms=True,
                layout_unclip_ratio=1.01,
                threshold=score_thresh
            )
            if not predictions:
                break

            page = predictions[0]
            blocks = [b for b in page['boxes'] if b['label'] == 'text']

            if not blocks:
                break

            # Copy image for visualization
            vis_img = img.copy()
            for i, block in enumerate(blocks):
                x_min, y_min, x_max, y_max = map(int, block['coordinate'])
                cropped = img[y_min:y_max, x_min:x_max]

                all_blocks.append({
                    "coords": (x_min, y_min, x_max, y_max),
                    "label": block['label'],
                    "score": block['score'],
                    "image": cropped
                })

                # draw rectangle
                cv2.rectangle(vis_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3)
                cv2.putText(vis_img, f"{i}:{block['score']:.2f}",
                            (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # paint white rectangle on original (for stripping)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)

        return all_blocks, img    