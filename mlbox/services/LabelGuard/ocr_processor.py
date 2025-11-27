from dataclasses import dataclass
from typing import List
from abc import ABC, abstractmethod
import os
import io
import json
from pathlib import Path
from google.cloud import vision
from PIL import Image
import numpy as np
import regex
from skimage import color
from skimage.morphology import disk, opening
from mlbox.settings import ROOT_DIR
from mlbox.utils.logger import get_logger, get_artifact_service




CURRENT_DIR = Path(__file__).parent
SERVICE_NAME = "labelguard"
app_logger = get_logger(ROOT_DIR)
artifact_service = get_artifact_service(ROOT_DIR)

@dataclass
class OCRWord:
    text: str
    bold: bool
    bbox: List[tuple]

@dataclass  
class OCRResult:
    text: str
    confidence: float
    language: str
    words: List[OCRWord]
    
    @classmethod
    def from_api_response(cls, response):
        """Convert Google Vision API response to our format"""
        if not response.text_annotations:
            return cls(text="", confidence=0.0, language="unknown", words=[])
            
        text = response.text_annotations[0].description
        language = response.text_annotations[0].locale or "unknown"
        confidence = getattr(response.text_annotations[0], 'score', 0.0)
        
        words = []
        for annotation in response.text_annotations[1:]:
            vertices = annotation.bounding_poly.vertices
            x_coords = [v.x for v in vertices]
            y_coords = [v.y for v in vertices]
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            
            words.append(OCRWord(
                text=annotation.description,
                bbox=[bbox],
                bold=False  # Will be determined later
            ))
        
        return cls(text=text, confidence=confidence, language=language, words=words)
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            "full_text": self.text,
            "language": self.language,
            "confidence": self.confidence,
            "words": [
                {"text": word.text, "bbox": word.bbox, "bold": word.bold}
                for word in self.words
            ]
        }
    
    @classmethod
    def from_dict(cls, data):
        """Create from dictionary (JSON deserialization)"""
        words = [
            OCRWord(text=w["text"], 
                    bbox=[tuple(bbox_item) for bbox_item in w["bbox"]], 
                    bold=w.get("bold", False))
            for w in data.get("words", [])
        ]
        return cls(
            text=data.get("full_text", ""),
            language=data.get("language", "unknown"),
            confidence=data.get("confidence", 0.0),
            words=words
        )


class BaseOCRProcessor(ABC):
    def __init__(self):
        pass
        
    @abstractmethod
    def process(self, input_image: Image.Image, confidence_threshold: float = 0.5, input_filename: str = None, json_vision_filename: str = None) -> OCRResult:
        """Process single image and return OCR result"""
        pass
    
    def _image_to_bytes(self, pil_image: Image.Image) -> bytes:
        """Convert PIL Image to bytes"""
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()
    
    def _create_error_result(self) -> OCRResult:
        """Create error result for failed processing"""
        return OCRResult(
            text="",
            confidence=0.0,
            language="error",
            words=[]
        )


class VisionOCRProcessor(BaseOCRProcessor):
    def __init__(self):
        """Initialize Vision API client using environment variables"""
        self.input_filename_stem = None
       
    
    def _save_vision_result(self, vision_result: OCRResult, json_vision_filename: str):
        """Save VisionResult as JSON file"""
        try:
            # json_vision_filename can be either:
            # - Full path (if passed from analyze function with cache/ subdirectory)
            # - Relative path (legacy usage)
            if Path(json_vision_filename).is_absolute() or '/' in json_vision_filename or '\\' in json_vision_filename:
                # It's already a full path or contains path separators
                filepath = Path(json_vision_filename)
            else:
                # Legacy: treat as relative to service directory
                filepath = artifact_service.get_service_dir(SERVICE_NAME) / json_vision_filename

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(vision_result.to_dict(), f, indent=2, ensure_ascii=False)

            app_logger.debug("ocr_processor", f"Saved Vision result: {filepath}")
                
        except Exception as e:
            app_logger.error("ocr_processor", f"Error saving Vision result: {str(e)}")
    
    def enrich_words_with_spaces_and_newlines(self, full_text : str, words : List[OCRWord]) -> List[OCRWord]:

        processed_words = []
        full_text_pos = 0
        current_word_index = 0
        
        if not words:
            # If there are no words, just return an empty list
            return []

        while full_text_pos < len(full_text):
            if current_word_index < len(words) and full_text[full_text_pos:].startswith(words[current_word_index].text):
                # If the current position in full_text matches the beginning of the current word
                word = words[current_word_index]
                processed_words.append(word)
                full_text_pos += len(word.text)
                current_word_index += 1
            elif full_text[full_text_pos] in [' ', '\n']:
                # If the current character is a space or newline
                char = full_text[full_text_pos]
                processed_words.append(OCRWord(text=char, bbox=[(0, 0, 0, 0)], bold=word.bold))
                full_text_pos += 1
            else:
                # If the character is not part of a word, space, or newline, skip it
                # (This might happen with some special characters or if there are inaccuracies in word detection)
                full_text_pos += 1

        return processed_words

    
    def _join_wrapped_words(self, words: List[OCRWord]) -> List[OCRWord]:
        new_words = []
        i = 0
        
        while i < len(words) - 1:
            word = words[i]
            
            if i < len(words) - 1 and word.text.endswith('-') and words[i + 1].text.startswith('\n'):
                # Join the words: remove hyphen and concatenate
                # Combine bboxes, ensuring they are tuples
                combined_bboxes = []
                for bbox in word.bbox + words[i + 2].bbox:
                    if isinstance(bbox, list):
                        combined_bboxes.append(tuple(bbox))
                    else:
                        combined_bboxes.append(bbox)
                
                joined_word = OCRWord(
                    text=word.text[:-1] + words[i+2].text, 
                    bbox=combined_bboxes,  # Combine bboxes since word spans multiple lines
                    bold=word.bold or words[i + 2].bold  # Bold if either part is bold
                )
                new_words.append(joined_word)
                i += 3  # Skip  current and next word (delimeter and next word)
            else:
                new_words.append(word)
                i += 1
        
        return new_words

    def process(self, input_image: Image.Image, confidence_threshold: float = 0.5, input_filename: str = None, json_vision_filename: str = None, language_hints: list = None) -> OCRResult:
        """Process image using Google Vision API OCR with bold detection and optional caching
        
        Args:
            input_image: PIL Image to process
            confidence_threshold: Minimum confidence threshold (not currently used)
            input_filename: Original filename for artifact naming
            json_vision_filename: Path to cached Vision API result (if exists)
            language_hints: List of BCP-47 language codes that might appear in image (e.g., ["en", "uk", "ro"])
        """
        results = []
        # Store filename stem for artifact naming
        if input_filename:
            self.input_filename_stem = Path(input_filename).stem
        
        # Check for existing Vision result from provided JSON file
        if json_vision_filename and Path(json_vision_filename).exists():
            os.utime(json_vision_filename, None)  # sets mtime to now
            app_logger.debug("ocr_processor", f"Loading cached Vision result from {json_vision_filename}")
            with open(json_vision_filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            vision_result = OCRResult.from_dict(data)
        else:
            app_logger.debug("ocr_processor", "Calling Google Vision API")
            self.client = vision.ImageAnnotatorClient()
            
            # Create Vision API image object
            vision_api_image = vision.Image(content=self._image_to_bytes(input_image))
            
            # Create image context with language hints if provided
            image_context = None
            if language_hints:
                image_context = vision.ImageContext(language_hints=language_hints)
                app_logger.debug("ocr_processor", f"Using language hints: {language_hints}")
            
            # Perform text detection with language hints
            response = self.client.text_detection(image=vision_api_image, image_context=image_context)
            
            # Handle API errors
            if response.error.message:
                app_logger.error("ocr_processor", f"Vision API error: {response.error.message}")
                return self._create_error_result()
            
            # Convert to our OCRResult format
            vision_result = OCRResult.from_api_response(response)
            
            # Save for future use
            if json_vision_filename:
                self._save_vision_result(vision_result, json_vision_filename)
            
            app_logger.debug("ocr_processor", f"Saved Vision result: {json_vision_filename}")
            
        self._extract_font_style_info(input_image, vision_result.words)
        words = self.enrich_words_with_spaces_and_newlines(vision_result.text, vision_result.words)

        #join wrapped words
        words = self._join_wrapped_words(words)

        ocr_result = OCRResult(
            text=''.join(word.text for word in words),
            confidence=vision_result.confidence,
            language=vision_result.language.lower(),  # Keep BCP47 format (lowercase)
            words=words
        )
            
        return ocr_result
    
    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR text by removing specific line break patterns"""
        if not text:
            return text
        
        # Remove "-\n" pattern (hyphenated line breaks)
        text = text.replace('-\n', '')
        text = text.replace(',\n', ', ')
        
        # Replace "\n" with space if lowercase letter before and after (any language)
        text = regex.sub(r'(\p{Ll})\n(\p{Ll})', r'\1 \2', text)
        
        return text.strip()
    
    def _extract_font_style_info (self, text_image: Image.Image, words: List[OCRWord]):
        """Extract style information from text image using full text with bold positions"""
        
        for word in words:
            if not word.bbox:
                continue  # Skip words with no bounding boxes
            
            # Handle word image - simple case (single bbox) or complex case (multiple bboxes)
            if len(word.bbox) == 1:
                # Simple case - just crop the single bounding box
                bbox = word.bbox[0]
                if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    word_image = text_image.crop(bbox)
                else:
                    continue  # Skip words with invalid bounding boxes
            else:
                # Complex case - word is split across multiple bounding boxes
                word_parts = []
                for bbox in word.bbox:
                    if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        word_parts.append(text_image.crop(bbox))
                    # Skip invalid bounding boxes
                
                # Combine all parts horizontally into one image
                if word_parts:
                    total_width = sum(img.width for img in word_parts)
                    max_height = max(img.height for img in word_parts)
                    word_image = Image.new('RGB', (total_width, max_height), 'white')
                    
                    x_offset = 0
                    for img in word_parts:
                        # Create a copy to avoid reference issues
                        img_copy = img.copy()
                        word_image.paste(img_copy, (x_offset, 0))
                        x_offset += img.width
                else:
                    continue  # Skip if no parts

            # Check if current word is bold
            is_bold = False
            if word.text not in ['«', '(', '+', '-', '°', '/', '№', '"', '[', ']', '»', ',', '.', ':', ';', '!', '?', ')', ' '] and len(word.text) > 1:
                is_bold = self._is_word_bold(word_image, word.text)
            
            # Add word with bold formatting
            word.bold = is_bold
        return 
    

    def _is_word_bold(self, word_image: Image.Image, word_text: str = "") -> bool:
        """ImageJ-compatible word analysis with proper bold detection using skimage morphology"""
        try:
            """
            if LOG_LEVEL == "DEBUG":
                # Add debugging to see what word is being processed
                print(f"DEBUG: Processing word '{word_text}' with image size {word_image.size}")
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=f"{self.input_filename_stem}_Word_{word_text}.jpg",
                    data=word_image
                )
            """


            
            # Convert to ImageJ-compatible grayscale using skimage
            img_array = np.array(word_image)
            if len(img_array.shape) == 3:
                gray_img = color.rgb2gray(img_array)
            else:
                gray_img = img_array.astype(float) / 255.0  # normalize to 0-1
            
            # Boolean mask: True = black text (ImageJ-compatible threshold)
            bw_bool_orig = gray_img < 0.5
            
            # Structuring element - disk with radius 1 (ImageJ-compatible)
            se = disk(1)
            iterations = 1
            if word_image.height <= 24:
                # Smaller fonts
                ratio_min, ratio_max = 1.0, 1.9
            else:
                # Larger fonts
                ratio_min, ratio_max = 1.0, 1.5
            
            # Apply morphological opening (erosion followed by dilation)
            processed = bw_bool_orig.copy()
            for _ in range(iterations):
                processed = opening(processed, se)
            
            # Calculate bold ratio
            black_orig = np.sum(bw_bool_orig)
            black_proc = np.sum(processed)
            
            # Bold ratio: original / processed (higher = bolder text survives better)
            bold_ratio = black_orig / black_proc if black_proc > 0 else 0
            
            # Bold detection: ratio between 1.0 and 1.5 indicates bold text
            is_bold = ratio_min <= bold_ratio < ratio_max
            """
            if LOG_LEVEL == "DEBUG":
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=f"{self.input_filename_stem}_Word_{word_text}_black_proc.jpg",
                    data=Image.fromarray(processed.astype(np.uint8) * 255)
                )
            
            if LOG_LEVEL == "DEBUG":
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=f"{self.input_filename_stem}_Word_{word_text}_black_orig.jpg",
                    data=Image.fromarray(bw_bool_orig.astype(np.uint8) * 255)
                )
            
            app_logger.debug("ocr_processor", 
                    f"Word: '{word_text}' - Black orig: {black_orig}, height: {word_image.height} "
                    f"ratio_min: {ratio_min}, ratio_max: {ratio_max} "
                    f"Black proc: {black_proc}, Bold ratio: {bold_ratio:.4f}, Bold: {is_bold}")
            """
            return is_bold
            
        except Exception as e:
            app_logger.error("ocr_processor", f"Error in ImageJ-compatible bold detection: {str(e)}")
            return False
