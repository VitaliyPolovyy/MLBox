from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import datetime
import json
import ast
import html
from PIL import Image, ImageDraw
from enum import Enum
from mlbox.utils.logger import get_logger, get_artifact_service
from mlbox.utils.llm_cache import LLMCache
from mlbox.utils.lcs import all_common_substrings_by_words, Match
from openai import OpenAI
import os
import re
import unicodedata
import glob
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from mlbox.services.LabelGuard.layout_detector import LayoutTextBlock, LayoutDetector
from mlbox.services.LabelGuard.ocr_processor import OCRResult, OCRWord, VisionOCRProcessor

from mlbox.services.LabelGuard.html_reporter import generate_html_report, Match, highlight_matches_html 
from mlbox.settings import ROOT_DIR, LOG_LEVEL


CURRENT_DIR = Path(__file__).parent
SERVICE_NAME = "labelguard"
app_logger = get_logger(ROOT_DIR)
artifact_service = get_artifact_service(ROOT_DIR)
llm_cache = LLMCache(artifact_service.get_service_dir(SERVICE_NAME) / "llm_cache")

@dataclass
class LabelInput:
    kmat: str
    version: str
    label_image: Image.Image
    label_image_path: str

@dataclass
class Sentence:
    text: str
    category: str #enmarates: ingdidients, product_name, allergen_phrase, contact_info, other
    words: List[OCRWord]
    index: int = 0

@dataclass
class TextBlock:
    bbox: tuple
    sentences: List[Sentence]
    index: int
    text: str

    type: str #enmarates: ingdidients, other

    allergens: List[OCRWord] #list of allergens if type = ingredients
    languages: str
    lcs_results: Optional[List[Match]] = None
    etalon_text: Optional[str] = None

class ErrorType(Enum):
    COMPARASION_WITH_ETALON = "comparasion_with_etalon"  # text not found in etalon
    ALLERGEN_ERROR = "allergen_error"  # count of allergen doesn't match
    NUMBERS_ERROR = "numbers_error"  # count of numbers doesn't match

@dataclass
class LabelError:
    error_type: ErrorType
    html_details: str  # Detailed HTML description for popup
    words: List[OCRWord]  # Words that have the error
    bounding_boxes : List[tuple]  # Bounding boxes to highlight
    text_block: Optional[TextBlock] = None  # Link to the TextBlock where error occurred

@dataclass
class VisualOverlay:
    overlay_type: str  # "background", "rectangle", "underline"
    color: tuple  # (R, G, B, A) for transparency
    error_area: List[tuple]  # Bounding boxes to highlight
    error: LabelError  # Associated error for click handling

@dataclass
class LabelProcessingResult:
    text_blocks: List[TextBlock] = field(default_factory=list)
    html_report: str = ""
    errors: List[LabelError] = field(default_factory=list)  # Validation errors
    # None if success
    error_message: Optional[str] = None
    kmat: Optional[str] = None
    version: Optional[str] = None
    original_filename: Optional[str] = None
    success: bool = True

def find_etalon_text(block_type : str, language : str, etalon_text_blocks : List[dict]) -> str:
    etalon_text = ""
    for etalon_text_block in etalon_text_blocks:
        # Normalize languages for comparison
        etalon_lang = etalon_text_block['LANGUAGES'].strip().upper() if etalon_text_block['LANGUAGES'] else None
        # Handle language code mappings
        lang_mapping = {
            'KA': 'GE',  # Georgian: ka -> GE
            'KK': 'KZ',  # Kazakh: kk -> KZ
        }
        lang_mapped = lang_mapping.get(language, language)
        
        if etalon_text_block['type_'] == block_type and (etalon_lang == lang_mapped or etalon_lang is None):
            etalon_text = clean_html_text(etalon_text_block['text'])
            break

    return etalon_text

def LoadLayoutBlocks(label_image_path: str) -> List[LayoutTextBlock]:
    layout_blocks = []
    image = Image.open(label_image_path)
    #read all files with this mask
    image_path = artifact_service.get_service_dir(SERVICE_NAME)
    json_vision_filenames = glob.glob(f"{image_path}/{label_image_path.stem}_*_vision.json")

    for json_vision_filename in json_vision_filenames:

        app_logger.debug("ocr_processor", f"Loading cached Vision result from {json_vision_filename}")
        with open(json_vision_filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        vision_result = OCRResult.from_dict(data)
        if not vision_result.words:
            continue
        #take all box from vision_result and make largest
        # For a group of OCRWords that should form one LayoutTextBlock:
        # Since each word.bbox is now a list of tuples, we need to flatten them
        all_x1 = [bbox[0] for word in vision_result.words for bbox in word.bbox]
        all_y1 = [bbox[1] for word in vision_result.words for bbox in word.bbox]  
        all_x2 = [bbox[2] for word in vision_result.words for bbox in word.bbox]
        all_y2 = [bbox[3] for word in vision_result.words for bbox in word.bbox]

        layout_bbox = (min(all_x1), min(all_y1), max(all_x2), max(all_y2))

        
        image_crop = image.crop(layout_bbox)

        layout_block = LayoutTextBlock(
            bbox=layout_bbox,
            image_crop=image_crop,
            score=1,
            index=int(json_vision_filename.split("_")[-2])
        )
        layout_blocks.append(layout_block)

    return layout_blocks

def split_words_into_sentences(words: List[OCRWord]):
    """
    Splits a list of enriched words into sentences based on delimiters .!?

    Args:
        words_enriched: A list of dictionaries, where each dictionary
                        represents a word or character and contains 'text' and 'bbox'.

    Returns:
        A list of lists, where each inner list represents a sentence and contains
        the word dictionaries belonging to that sentence.
    """
    sentences = []
    current_sentence : Sentence = Sentence(text="", category="", words=[], index=0)
    delimiters = ['.', '!', '?']
    sentence_index = 0

    for word in words:
        current_sentence.words.append(word)
        if word.text in delimiters:
            current_sentence.index = sentence_index
            sentence_index += 1
            sentences.append(current_sentence)
            current_sentence : Sentence = Sentence(text="", category="", words=[], index=sentence_index)

    # Add the last sentence if it doesn't end with a delimiter
    if current_sentence:
        current_sentence.index = sentence_index 
        sentences.append(current_sentence)

    for sentence in sentences:
        sentence.text = ''.join(word.text for word in sentence.words)

    return sentences

def get_unmatched_positions(text: str, matches: List[Match]) -> List[tuple]:
    """Extract character positions of text that doesn't match etalon"""
    if not matches:
        # No matches - entire text is unmatched
        return [(0, len(text) - 1)]
    
    # Create a set of all matched character positions
    matched_positions = set()
    for match in matches:
        for pos in range(match.start_a, match.start_a + match.len_a):
            matched_positions.add(pos)
    
    # Find ranges of consecutive unmatched positions
    unmatched_ranges = []
    start_unmatched = None
    
    for pos in range(len(text)):
        if pos not in matched_positions:
            if start_unmatched is None:
                start_unmatched = pos
        else:
            if start_unmatched is not None:
                unmatched_ranges.append((start_unmatched, pos - 1))
                start_unmatched = None
    
    # Handle case where text ends with unmatched characters
    if start_unmatched is not None:
        unmatched_ranges.append((start_unmatched, len(text) - 1))
    
    return unmatched_ranges

def detect_comparasiomn_with_etalon_text_errors(label_processing_result: LabelProcessingResult) -> List[LabelError]:
    """Detect text that appears in OCR but not in etalon text"""
    errors = []
    app_logger.info(SERVICE_NAME, f"Starting error detection for {len(label_processing_result.text_blocks)} text blocks")
   
    for text_block in label_processing_result.text_blocks:
       
        # Find words that are not covered by LCS matches
        words = []
        for sentence in text_block.sentences:
            words.extend(sentence.words)

        lcs_results = []
        text = text_block.text.replace("\n", " ")
        if text_block.etalon_text and text_block.text:
            lcs_results = all_common_substrings_by_words(text, text_block.etalon_text, min_length_words=2, maximal_only=True)

        # Get unmatched text positions (text that doesn't match etalon)
        position_ranges = get_unmatched_positions(text, lcs_results)
        
        error_words = get_words_by_char_position(words, position_ranges)
        # to do: i want to delete error words like: space, period, comma, etc.
        error_words = [word for word in error_words if word.text not in [" ", ".", ",", "!", "?", ":", ";", "\n"]]

        if error_words:
            # Convert word bboxes from text block coordinates to full image coordinates
            full_image_bboxes = []
            for word in error_words:
                for bbox in word.bbox:
                    # Add text block offset to word coordinates
                    full_bbox = (
                        text_block.bbox[0] + bbox[0],  # x1
                        text_block.bbox[1] + bbox[1],  # y1
                        text_block.bbox[0] + bbox[2],  # x2
                        text_block.bbox[1] + bbox[3]   # y2
                    )
                    full_image_bboxes.append(full_bbox)
            
            # Get highlighted versions of both texts using existing function
            etalon_highlighted = highlight_matches_html(text_block.etalon_text or "", text_block.lcs_results or [], use_start_a=True)
            
            # Generate detailed HTML content for this error
            error_html_details = f"""
            <div class="text-comparison">
                <div class="text-block">
                    <h4>Не співпадає текст з еталоном (червоні слова - відсутні на етикетці)</h4>
                    <div class="text-content">{etalon_highlighted}</div>
                </div>
            </div>
            """
            
            errors.append(LabelError(
                error_type=ErrorType.COMPARASION_WITH_ETALON,
                html_details=error_html_details,
                words=error_words,
                bounding_boxes = full_image_bboxes,
                text_block=text_block
            ))
    
    return errors

def detect_allergen_errors(label_processing_result: LabelProcessingResult) -> List[LabelError]:
    """
    Detect allergen count mismatches between text blocks.
    
    Logic:
    1. Find maximum count of allergens among text blocks (etalon count)
    2. Compare each block's allergen count with etalon count
    3. Add errors when counts don't match
    4. Generate HTML details showing allergens with "(має бути X)" when count is incorrect
    """
    errors = []
    
    # Get all text blocks that have allergens
    text_blocks_with_allergens = [
        block for block in label_processing_result.text_blocks 
        if block.allergens and len(block.allergens) > 0
    ]
    
    if not text_blocks_with_allergens:
        return errors
    
    # 1. Find maximum count of allergens (etalon count)
    allergen_counts = [len(block.allergens) for block in text_blocks_with_allergens]
    etalon_count = max(allergen_counts)
    
    # 2. Compare each block's allergen count with etalon count
    for text_block in text_blocks_with_allergens:
        current_count = len(text_block.allergens)
        
        #if current_count != etalon_count:
        if 1==1:
            # 3. Generate HTML details for allergen error
            allergen_names = [allergen.text for allergen in text_block.allergens]
            allergens_display = ", ".join(allergen_names)
            
            #to do: I want to add list of allergens in html details
            allergen_html_details = allergens_display + f"""
            <div class="error-section">
                <h3>Проблеми з аліргенами</h3>
                <p><strong>Аліргени:</strong> {allergens_display} (має бути {etalon_count})</p>
            </div>
            """
            
            # Get bounding boxes for all allergens in this block
            allergen_bboxes = [allergen.bbox for allergen in text_block.allergens]
            
            errors.append(LabelError(
                error_type=ErrorType.ALLERGEN_ERROR,
                html_details=allergen_html_details,
                words=text_block.allergens,
                bounding_boxes=allergen_bboxes,
                text_block=text_block
            ))
    
    return errors

def detect_numbers_errors(text_block: TextBlock) -> List[LabelError]:
    """Detect number count mismatches"""
    errors = []
    
    # Extract numbers from etalon text
    etalon_numbers = set()
    if text_block.etalon_text:
        etalon_number_matches = re.findall(r'\b\d+(?:\.\d+)?(?:[%°]|\s*(?:mg|g|ml|kg|lb|oz))?\b', text_block.etalon_text)
        etalon_numbers = set(match.lower().strip() for match in etalon_number_matches)
    
    # Extract numbers from OCR text
    ocr_numbers = set()
    for sentence in text_block.sentences:
        for word in sentence.words:
            number_matches = re.findall(r'\b\d+(?:\.\d+)?(?:[%°]|\s*(?:mg|g|ml|kg|lb|oz))?\b', word.text)
            ocr_numbers.update(match.lower().strip() for match in number_matches)
    
    # Check for mismatches
    missing_numbers = etalon_numbers - ocr_numbers
    extra_numbers = ocr_numbers - etalon_numbers
    
    if missing_numbers or extra_numbers:
        error_details = "<h3>Number Count Mismatch</h3>"
        if missing_numbers:
            error_details += f"<p>Missing numbers: <strong>{', '.join(missing_numbers)}</strong></p>"
        if extra_numbers:
            error_details += f"<p>Extra numbers: <strong>{', '.join(extra_numbers)}</strong></p>"
        
        # Get word objects for numbers
        number_words = []
        for sentence in text_block.sentences:
            for word in sentence.words:
                if re.search(r'\b\d+(?:\.\d+)?(?:[%°]|\s*(?:mg|g|ml|kg|lb|oz))?\b', word.text):
                    number_words.append(word)
        
        error = LabelError(
            error_type=ErrorType.NUMBERS_ERROR,
            html_details=error_details,
            words=number_words,
            text_block=text_block
        )
        errors.append(error)
    
    return errors

def get_words_by_char_position(words, positions : List[tuple]):
    """
    Retrieves words from the words_enriched list based on character positions.

    Args:
        words_enriched: A list of dictionaries, where each dictionary
                        represents a word or character and contains 'text' and 'bbox'.
        start_pos: The starting character position (inclusive).
        end_pos: The ending character position (inclusive).

    Returns:
        A list of word dictionaries that intersect with the specified character range.
    """
    result_words = []

    for start_pos, end_pos in positions:
        current_char_pos = 0
        for word in words:
            word_length = len(word.text)
            word_end_pos = current_char_pos + word_length - 1

            # Check for intersection with the specified range
            if max(current_char_pos, start_pos) <= min(word_end_pos, end_pos):
                if word not in result_words:
                    result_words.append(word)

            current_char_pos += word_length

    return result_words


def detect_errors(label_processing_result: LabelProcessingResult) -> List[LabelError]:
    """Detect all errors in a LabelProcessingResult"""
    all_errors = []

    all_errors.extend(detect_comparasiomn_with_etalon_text_errors(label_processing_result))
    all_errors.extend(detect_allergen_errors(label_processing_result))
        
    
    return all_errors

def get_word_bounding_boxes(word: OCRWord) -> List[tuple]:
    """Get all bounding boxes for a word (handles wrapped words)"""
    if not word.bbox:
        return []
    
    # Ensure word.bbox is a list of tuples
    if isinstance(word.bbox, tuple):
        return [word.bbox]
    else:
        return word.bbox

def generate_comparison_error_report(result: LabelProcessingResult) -> str:
    """Generate HTML report for COMPARASION_WITH_ETALON errors without table"""
    
    # Filter only COMPARASION_WITH_ETALON errors
    comparison_errors = [error for error in result.errors if error.error_type == ErrorType.COMPARASION_WITH_ETALON]
    
    if not comparison_errors:
        return ""
    
    # Create report title
    title = f"Text Comparison Error Report - {result.kmat} v{result.version}"
    
    # Generate HTML content with same foundation as generate_html_report but without table
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(title)}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        
        .report-title {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 24px;
        }}
        
        .error-container {{
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-radius: 8px;
            overflow: hidden;
        }}
        
        .error-header {{
            background-color: #ff4444;
            color: white;
            padding: 15px;
            font-weight: bold;
            font-size: 16px;
        }}
        
        .error-content {{
            padding: 20px;
        }}
        
        .error-details {{
            margin-bottom: 15px;
        }}
        
        .highlighted-text {{
            background-color: #ffcccc;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }}
        
        .matched-text {{
            background-color: #ccffcc;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        
        .text-comparison {{
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }}
        
        .text-block {{
            flex: 1;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #fafafa;
        }}
        
        .text-block h4 {{
            margin-top: 0;
            color: #333;
        }}
        
        .text-content {{
            white-space: pre-wrap;
            word-break: break-word;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.4;
        }}
    </style>
</head>
<body>
    <h1 class="report-title">{html.escape(title)}</h1>
"""
    
    # Add each comparison error
    for i, error in enumerate(comparison_errors):
        html_content += f"""
    <div class="error-container">
        <div class="error-header">
            Error {i+1}: Extra Text Found
        </div>
        <div class="error-content">
            <div class="error-details">
                {error.html_details}
            </div>
            
            <div class="text-comparison">
                <div class="text-block">
                    <h4>Label Text (with errors highlighted)</h4>
                    <div class="text-content">{html.escape(error.text_block.text if error.text_block else '')}</div>
                </div>
                <div class="text-block">
                    <h4>Template Text</h4>
                    <div class="text-content">{html.escape(error.text_block.etalon_text if error.text_block and error.text_block.etalon_text else 'No template available')}</div>
                </div>
            </div>
        </div>
    </div>
"""
    
    html_content += """
</body>
</html>"""
    
    return html_content

def generate_interactive_error_viewer(
    result: LabelProcessingResult,
    error_overlay_image_path: str
) -> str:
    """Generate interactive HTML viewer for error visualization"""
    
    # Group errors by text block
    text_block_errors = {}
    for error in result.errors:
        if error.text_block:
            block_id = id(error.text_block)
            if block_id not in text_block_errors:
                text_block_errors[block_id] = {
                    'text_block': error.text_block,
                    'errors': []
                }
            text_block_errors[block_id]['errors'].append(error)
    
    # Create highlights data for JavaScript
    highlights = []
    for block_id, block_data in text_block_errors.items():
        text_block = block_data['text_block']
        errors = block_data['errors']
        
        # Join all HTML details for this text block
        combined_details = []
        for error in errors:
            combined_details.append(error.html_details)

        #to do: add Enter before block text
        combined_details.append("<br>")
        combined_details.append(f"{text_block.type}")
        combined_details.append(f"{text_block.text}")

        
        highlights.append({
            'x1': text_block.bbox[0],
            'y1': text_block.bbox[1], 
            'x2': text_block.bbox[2],
            'y2': text_block.bbox[3],
            'type': 'error',
            'message': '<br><br>'.join(combined_details)
        })
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body {{ display: flex; height: 100vh; margin: 0; overflow: hidden; }}
  #imagePanel {{ flex: 2; position: relative; overflow: hidden; border-right: 1px solid #ccc; cursor: grab; }}
  #messagePanel {{ flex: 1; padding: 10px; overflow-y: auto; }}
  canvas {{ display: block; }}
</style>
</head>
<body>

<div id="imagePanel">
  <canvas id="labelCanvas"></canvas>
</div>
<div id="messagePanel">
  <div id="msgContent">Click on a red rectangle to see error details</div>
</div>

<script>
const canvas = document.getElementById('labelCanvas');
const ctx = canvas.getContext('2d');
const msgDiv = document.getElementById('msgContent');

const img = new Image();
img.src = '{error_overlay_image_path}';

// Error highlighting data
const highlights = {json.dumps(highlights, indent=2)};

let scale = 1;
let offsetX = 0, offsetY = 0;
let isDragging = false, startX = 0, startY = 0;

img.onload = () => {{
  canvas.width = img.width;
  canvas.height = img.height;
  drawCanvas();
}};

function drawCanvas() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);
  ctx.drawImage(img, 0, 0);
  highlights.forEach(h => {{
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2 / scale;
    ctx.strokeRect(h.x1, h.y1, h.x2-h.x1, h.y2-h.y1);
  }});
  ctx.restore();
}}

// Click to show message
canvas.addEventListener('click', e => {{
  const x = (e.offsetX - offsetX) / scale;
  const y = (e.offsetY - offsetY) / scale;
  const hit = highlights.find(h => x>=h.x1 && x<=h.x2 && y>=h.y1 && y<=h.y2);
  if(hit) msgDiv.innerHTML = `
  <div style="margin-bottom:10px;">
    <span style="font-weight:bold; color:#c00;">${{hit.type}}</span><br>
    <span>${{hit.message}}</span>
  </div>
`;
}});

// Zoom with Ctrl + wheel
canvas.addEventListener('wheel', e => {{
  if(e.ctrlKey) {{
    e.preventDefault();
    const zoom = e.deltaY < 0 ? 1.1 : 0.9;
    const mx = (e.offsetX - offsetX) / scale;
    const my = (e.offsetY - offsetY) / scale;

    scale *= zoom;
    offsetX -= mx * (zoom - 1) * scale;
    offsetY -= my * (zoom - 1) * scale;
    drawCanvas();
  }}
}});

// Pan with mouse drag
canvas.addEventListener('mousedown', e => {{
  isDragging = true;
  startX = e.clientX - offsetX;
  startY = e.clientY - offsetY;
  canvas.style.cursor = 'grabbing';
}});

canvas.addEventListener('mousemove', e => {{
  if(isDragging){{
    offsetX = e.clientX - startX;
    offsetY = e.clientY - startY;
    drawCanvas();
  }}
}});

canvas.addEventListener('mouseup', () => {{ isDragging = false; canvas.style.cursor = 'grab'; }});
canvas.addEventListener('mouseleave', () => {{ isDragging = false; canvas.style.cursor = 'grab'; }});
</script>

</body>
</html>"""
    
    return html_content

def generate_error_overlay_image(
    original_image: Image.Image,
    result: LabelProcessingResult,
    errors: List[LabelError]
) -> Image.Image:
    """
    Creates overlay image with error highlights
    Returns: Original image with error overlays drawn on top
    """
    # Create a copy of the original image to draw on
    overlay_image = original_image.copy()
    # Convert to RGBA if not already
    if overlay_image.mode != 'RGBA':
        overlay_image = overlay_image.convert('RGBA')
    draw = ImageDraw.Draw(overlay_image)
    
    # Get unique text blocks that have errors
    error_text_blocks = []
    seen_text_blocks = set()
    for error in errors:
        if error.text_block and id(error.text_block) not in seen_text_blocks:
            error_text_blocks.append(error.text_block)
            seen_text_blocks.add(id(error.text_block))
    
    # Step 1: Draw red rectangle over text blocks that have errors
    for text_block in error_text_blocks:
        x1, y1, x2, y2 = text_block.bbox
        # Draw red rectangle outline over the text block
        draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        
        # Draw text block index on the rectangle
        try:
            # Try to use a default font, fallback to basic if not available
            from PIL import ImageFont
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
        except ImportError:
            font = None
            
        # Draw the text block index
        text = f"{text_block.index}"
        if font:
            # Get text size to position it properly
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # Approximate size for default font
            text_width = len(text) * 6
            text_height = 12
            
        # Position text at top-left corner of the rectangle
        text_x = x1 + 5
        text_y = y1 + 5
        
        # Draw a small background rectangle for better text visibility
        draw.rectangle([text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2], 
                      fill=(255, 255, 255, 200), outline=(255, 0, 0))
        
        # Draw the text
        if font:
            draw.text((text_x, text_y), text, fill=(255, 0, 0), font=font)
        else:
            draw.text((text_x, text_y), text, fill=(255, 0, 0))
    
    # Step 2: Draw light red background for words with errors
    for error in errors:
        for word in error.words:
            # word.bbox is a list of tuples, so we need to iterate through each bbox
            for bbox in word.bbox:
                # Calculate absolute coordinates
                word_x1 = bbox[0] + error.text_block.bbox[0]
                word_y1 = bbox[1] + error.text_block.bbox[1]
                word_x2 = bbox[2] + error.text_block.bbox[0]
                word_y2 = bbox[3] + error.text_block.bbox[1]
                
                # Create a semi-transparent light grey overlay for error words
                # Extract the word region
                word_region = overlay_image.crop((word_x1, word_y1, word_x2, word_y2))
                # Convert to RGB if needed
                if word_region.mode != 'RGB':
                    word_region = word_region.convert('RGB')
                
                # Create a light grey overlay
                overlay = Image.new('RGB', word_region.size, (255, 0, 0))
                # Blend with low opacity to keep text visible
                blended = Image.blend(word_region, overlay, 0.4)  # 20% opacity
                
                # Paste back the blended region
                overlay_image.paste(blended, (word_x1, word_y1))
    
    return overlay_image

def refine_text (ocr_result: OCRResult):
    # to do: i want to repace °С to °C in ocr_result.text and ocr_result.words
    #  ocr_result.words keep °С as two words, for example: words[145] = "°" and words[146] = "С"
    ocr_result.text = ocr_result.text.replace('°С', '°C')
    for i, word in enumerate(ocr_result.words):
        if i < len(ocr_result.words) - 1:
            next_word = ocr_result.words[i + 1]
            if word.text == '°' and next_word.text == 'С':
                ocr_result.words[i+1].text = "C"
                break
        
    
    return ocr_result

def process_labels(labels: List[LabelInput]) -> List[LabelProcessingResult]:
    """
    Process multiple labels through the complete pipeline according to architecture:
    1. Layout detection
    2. OCR processing per layout block
    3. Text analysis and classification
    4. HTML report generation
    
    Args:
        labels: List of LabelInput objects containing kmat, version, and image
        
    Returns:
        List of LabelResult objects with processed text blocks
    """
    # Initialize processors
    layout_detector = LayoutDetector()
    ocr_processor = VisionOCRProcessor()
    
    results = []
    
    for label in labels:
        app_logger.info(SERVICE_NAME, f"Processing label: {label.kmat} v{label.version}")

        result = LabelProcessingResult()
        result.kmat = label.kmat
        result.version = label.version
        result.original_filename = Path(label.label_image_path).stem  # Store original filename without extension
        
        etalon_text_blocks = _get_etalon_text_blocks(label.kmat, label.version, label.label_image_path )
        
        # Step 1: Layout detection
        app_logger.info(SERVICE_NAME, f"Starting layout detection for {label.kmat}")
        layout_blocks = layout_detector.extract_blocks(label.label_image_path)
        #layout_blocks = LoadLayoutBlocks(label.label_image_path)
        
        # Save detected blocks as artifacts
        if LOG_LEVEL == "DEBUG":
            for i, layout_block in enumerate(layout_blocks):
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=f"{Path(label.label_image_path).stem}_block_{i}.jpg",
                    data=layout_block.image_crop
                )
            
        
        # Step 2: OCR processing
        app_logger.info(SERVICE_NAME, f"Starting OCR processing for {label.kmat}")
        for layout_block in layout_blocks:
            json_vision_filename = f"{artifact_service.get_service_dir(SERVICE_NAME) / Path(label.label_image_path).stem}_{layout_block.index}_vision.json"
            ocr_result = ocr_processor.process(layout_block.image_crop, input_filename=str(label.label_image_path), json_vision_filename=json_vision_filename)
            
            # Save OCR results as text artifacts in DEBUG mode
            if LOG_LEVEL == "DEBUG":
                input_name = Path(label.label_image_path).stem
            
                text_filename = f"{input_name}_block_{i}.txt"
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=text_filename,
                    data=f"Confidence: {ocr_result.confidence:.2f}\nLanguage: {ocr_result.language}\nExtracted Text:\n{ocr_result.text}"
                )
            # Step 3: fill in LabelProcessingResult.text_blocks
            block_type = detect_text_block_type(ocr_result.text)
            allergens = ""
            refine_text (ocr_result)
            
            sentences = split_words_into_sentences (ocr_result.words)
            if block_type == "ingredients":
                # classified_sentences = [sentence, type]
                classify_ingredients_sentences(sentences)
                try:
                    ingridients_sentence_index = [sentence.index for sentence in sentences if sentence.category == "INGRIDIENTS"][0]

                    #bolded_words = [word for word in sentences[ingridients_sentence_index].words if word.bold and word.text != " "]
                    #allergens = detect_allergens(bolded_words, ocr_result.language)

                    # Find the first colon
                    for i, word in enumerate(sentences[ingridients_sentence_index].words):
                        if word.text == ':':
                            allergens = [word for word in sentences[ingridients_sentence_index].words[i + 1:] if word.bold and word.text != " "]
                            break

                except IndexError:
                    allergens = []
            
            etalon_text = find_etalon_text(block_type, ocr_result.language, etalon_text_blocks)
            lcs_results = []
            if etalon_text and ocr_result.text:
                lcs_results = all_common_substrings_by_words(etalon_text, ocr_result.text, min_length_words=2, maximal_only=True)
            
            #find etalon text block by type and language
            text_block = TextBlock(
                bbox=layout_block.bbox,
                index=layout_block.index,
                sentences=sentences,
                text=ocr_result.text,
                type=block_type,
                allergens=allergens,
                languages=[ocr_result.language],
                etalon_text=etalon_text,
                lcs_results=lcs_results
            )
            result.text_blocks.append(text_block)

            # Step 3: Text comparison
            app_logger.info(SERVICE_NAME, f"Starting text comparison for {label.kmat}")

            # Step 4: Error detection
            app_logger.info(SERVICE_NAME, f"Detecting errors for {label.kmat}")
            result.errors = detect_errors(result)
            app_logger.info(SERVICE_NAME, f"Found {len(result.errors)} errors for {label.kmat}")

            # Step 5: Generate HTML report (disabled - only keeping interactive viewer)
            # result.html_report = generate_html_report(result)
            
            # Generate and save error overlay image if errors found
            if result.errors:
                app_logger.info(SERVICE_NAME, f"Generating error overlay image for {label.kmat}")
                error_overlay_image = generate_error_overlay_image(
                    label.label_image, 
                    result, 
                    result.errors
                )
                # Convert RGBA to RGB for JPEG compatibility
                if error_overlay_image.mode == 'RGBA':
                    # Create a white background
                    background = Image.new('RGB', error_overlay_image.size, (255, 255, 255))
                    background.paste(error_overlay_image, mask=error_overlay_image.split()[-1])  # Use alpha channel as mask
                    error_overlay_image = background
                
                # Save error overlay image
                overlay_filename = f"{result.original_filename}_errors_overlay.jpg"
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=overlay_filename,
                    data=error_overlay_image
                )
                
                # Generate interactive error viewer
                app_logger.info(SERVICE_NAME, f"Generating interactive error viewer for {label.kmat}")
                # Use just the filename since HTML and image are in same directory
                interactive_viewer_html = generate_interactive_error_viewer(
                    result,
                    overlay_filename
                )
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=f"{result.original_filename}_interactive_viewer.html",
                    data=interactive_viewer_html
                )
            
            results.append(result)
    
    return results


def detect_allergens(words: List[str], language: str = "en") -> List[str]:
    """Detect allergens from bold text using LLM"""
    if not words:
        return []

    try:
        # Get OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            app_logger.error("labelguard", "OPENAI_API_KEY environment variable not set")
            return []
        
        client = OpenAI(api_key=openai_api_key)
        
        words_for_prompt = ", ".join([f"{index}: {word.text}" for index, word in enumerate(words)])
        
        prompt = f"""
        You are given a list of words: {words_for_prompt}

        Task: Identify which words are actual food ingredients.

        Instructions:
        - Consider only the words in this list.
        - Preserve the order of ingredients.
        - **EXCLUDE label/header words** such as:
        - 'Ingrediente', 'Ingredients', 'Ingrédients', 'ინგრედიენტები' (and variations in any language)
        - 'Құрамы', 'Склад', 'Состав', 'Composition', 'Contains', 'Made with'
        - Any word that introduces or describes the list itself
        - **ONLY include actual food substances** (e.g., sugar, salt, wheat, milk, sulfit, orz)
        - Return ONLY a Python-style list of integers (e.g., [1, 2]). Zero-based indexing.
        - Do NOT return JSON, explanations, or any extra text.
        - If no ingredients found, return [].
        """
        model = "gpt-4o-mini"
        indexes_str = ""
        cached_response = llm_cache.get(prompt, model)
        if cached_response:
            indexes_str = cached_response
            app_logger.info("labelguard", f"Detecting allergens LLM prompt: {prompt} cashed response: {cached_response}")

        else:
      
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            
            indexes_str = response.choices[0].message.content.strip()
            llm_cache.set(prompt, model, indexes_str)
            
            app_logger.info("labelguard", f"Detecting allergens LLM prompt: {prompt} response: {response.choices[0].message.content.strip()}")
            
        
        # Convert string representation of list into a Python list
        try:
            indexes = ast.literal_eval(indexes_str)
            if not isinstance(indexes, list):
                indexes = []
        except:
            indexes = []

        # Parse the response into a list
        if len(indexes) > 0:
            allergens = [words[index] for index in indexes]
        else:
            allergens = []

        return allergens
            
    except Exception as e:
        app_logger.error("labelguard", f"Error calling LLM for allergen detection: {str(e)}")
        return []

def detect_allergens_with_llm (words: List[str], language: str = "en") -> List[str]:
    """Detect allergens from bold text using LLM"""
    if not words:
        return []

    try:
        # Get OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            app_logger.error("labelguard", "OPENAI_API_KEY environment variable not set")
            return []
        
        client = OpenAI(api_key=openai_api_key)
        
        words_for_prompt = ", ".join([f"{index}: {word.text}" for index, word in enumerate(words)])
        
        prompt = f"""
        You are given a list of words: {words_for_prompt}

        Task: Identify which words are actual food ingredients.

        Instructions:
        - Consider only the words in this list.
        - Preserve the order of ingredients.
        - **EXCLUDE label/header words** such as:
        - 'Ingrediente', 'Ingredients', 'Ingrédients', 'ინგრედიენტები' (and variations in any language)
        - 'Құрамы', 'Склад', 'Состав', 'Composition', 'Contains', 'Made with'
        - Any word that introduces or describes the list itself
        - **ONLY include actual food substances** (e.g., sugar, salt, wheat, milk, sulfit, orz)
        - Return ONLY a Python-style list of integers (e.g., [1, 2]). Zero-based indexing.
        - Do NOT return JSON, explanations, or any extra text.
        - If no ingredients found, return [].
        """
        model = "gpt-4o-mini"
        indexes_str = ""
        cached_response = llm_cache.get(prompt, model)
        if cached_response:
            indexes_str = cached_response
            app_logger.info("labelguard", f"Detecting allergens LLM prompt: {prompt} cashed response: {cached_response}")

        else:
      
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            
            indexes_str = response.choices[0].message.content.strip()
            llm_cache.set(prompt, model, indexes_str)
            
            app_logger.info("labelguard", f"Detecting allergens LLM prompt: {prompt} response: {response.choices[0].message.content.strip()}")
            
        
        # Convert string representation of list into a Python list
        try:
            indexes = ast.literal_eval(indexes_str)
            if not isinstance(indexes, list):
                indexes = []
        except:
            indexes = []

        # Parse the response into a list
        if len(indexes) > 0:
            allergens = [words[index] for index in indexes]
        else:
            allergens = []

        return allergens
            
    except Exception as e:
        app_logger.error("labelguard", f"Error calling LLM for allergen detection: {str(e)}")
        return []


def classify_ingredients_sentences(sentences: List[Sentence]) -> None:  # Fixed typo in function name
    """Classify sentences into different categories using LLM"""
    if not sentences:
        return
    
    # Define categories with descriptions
    categories = {
        "PRODUCT_NAME": "Product name",
        "ALLERGEN_PHRASE": "Sentences that list possible allergens",
        "INGRIDIENTS": "Sentences that list ingredients or composition",
        "CONTACT_INFO": "contact details, addres, phone, address, email",
        "UNKNOWN": "Unclassified or ambiguous text"
    }
    
    try:
        # Get OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            app_logger.error("labelguard", "OPENAI_API_KEY environment variable not set")
            return
        
        client = OpenAI(api_key=openai_api_key)
        
        # Create numbered sentences for the prompt
        numbered_sentences = [
            f"{sentence.index}: {sentence.text}" 
            for sentence in sorted(sentences, key=lambda s: s.index)
        ]
        sentences_text = "\n".join(numbered_sentences)
        
        # Build categories description
        categories_desc = "\n".join([
            f"- {key}: {description}" 
            for key, description in categories.items()
        ])
        
        prompt = f"""You are given a list of sentences from a product label. Classify each sentence into these categories: {', '.join(categories.keys())}.

Categories:
{categories_desc}

Instructions:
1. For each sentence, determine which category it belongs to.
2. Return a **valid JSON object only** with category names as keys and lists of sentence indices as values.
3. Do not include any extra text, explanations, or comments.
4. If a category has no sentences, return an empty list for that key.
5. The JSON must be strictly parseable by `json.loads()`.
6. DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.

Sentences:
{sentences_text}

Return format example: {{"PRODUCT_NAME": [0, 1], "ALLERGEN_PHRASE": [2], "INGRIDIENTS": [3], "CONTACT_INFO": [4, 5], , "UNKNOWN": []}}
"""

        # Check cache first
        response = llm_cache.get(prompt, "gpt-4o-mini")
        if response:
            app_logger.info("labelguard", f"Using cached response for classification")
        else:
            # Call OpenAI API
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0
            )
            response = completion.choices[0].message.content.strip()
            
            # Cache the response
            llm_cache.set(prompt, "gpt-4o-mini", response)
            app_logger.info("labelguard", f"Received new response from LLM")

        # Parse JSON response
        # Handle potential markdown code blocks
        if response.startswith("```json"):
            response = response.replace("```json\n", "").replace("```\n", "").replace("```", "").strip()
        
        result = json.loads(response)

        # Transform the result from {category: [indices]} to {index: category}
        index_to_category = {}
        for category, indices in result.items():
            for index in indices:
                index_to_category[index] = category
        
        # Set category for each sentence
        for sentence in sentences:
            sentence.category = index_to_category.get(sentence.index, "UNKNOWN")
        
        app_logger.info("labelguard", f"Successfully classified {len(sentences)} sentences")
            
    except json.JSONDecodeError as e:
        app_logger.error("labelguard", f"Failed to parse LLM response as JSON: {str(e)}, Response: {response}")
        # Set all sentences to UNKNOWN on error
        for sentence in sentences:
            sentence.category = "UNKNOWN"
    except Exception as e:
        app_logger.error("labelguard", f"Error calling LLM for sentence classification: {str(e)}")
        # Set all sentences to UNKNOWN on error
        for sentence in sentences:
            sentence.category = "UNKNOWN"
    """Classify sentences into different categories using LLM"""
    if not sentences:
        return {category: [] for category in categories}    

    model = "gpt-4o-mini"
    #to do: add descriptoin in categories
    categories = {
        "PRODUCT_NAME": "Product name, brand, or title",
        "ALLERGEN_PHRASE": "Sentences that list possible allergens",
        "INGRIDIENTS": "Sentences that list ingredients or composition",
        "CONTACT_INFO": "Storage instructions, contact info, or other details",
        "UNKNOWN": "Unclassified or ambiguous text"
    }
    
    try:
        # Get OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            app_logger.error("labelguard", "OPENAI_API_KEY environment variable not set")
            return {category: [] for category in categories.keys()}
        
        client = OpenAI(api_key=openai_api_key)
        
        # Create numbered sentences for the prompt
        numbered_sentences = [f"{sentence.index}: {sentence.text}" for sentence in sorted(sentences, key=lambda s: s.index)]
        sentences_text = "\n".join(numbered_sentences)
        
        prompt = f"""You are given a list of sentences from a product label. Classify each sentence into these categories: {', '.join(categories.keys())}.

        Categories:
        {chr(10).join([f"        - {key}: {description}" for key, description in categories.items()])}

        Instructions:
        1. For each sentence, determine which category it belongs to.
        2. Return a **valid JSON object only** with category names as keys and lists of sentence indices as values.
        3. Do not include any extra text, explanations, or comments.
        4. If a category has no sentences, return an empty list for that key.
        5. The JSON must be strictly parseable by `json.loads()`.

        Sentences:
        {sentences_text}

        Return format: {{"PRODUCT_NAME": [0, 1], "ALLERGEN_PHRASE": [2], "INGRIDIENTS": [3], "CONTACT_INFO": [4, 5]}}
        """

        response = llm_cache.get(prompt, model)
        if response:
            app_logger.info("labelguard", f"Classifying sentences LLM prompt: {prompt} cashed response: {response}")
        else:

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0
            )
            response = response.choices[0].message.content.strip()
            llm_cache.set(prompt, model, response)
            app_logger.info("labelguard", f"Classifying sentences LLM prompt: {prompt} response: {response}")

        result = json.loads(response)

        # Transform the result from {category: [indices]} to {index: category}
        index_to_category = {}
        for category, indices in result.items():
            for index in indices:
                index_to_category[index] = category
        
        #set category for sentences
        for sentence in sentences:
            sentence.category = index_to_category.get(sentence.index, "UNKNOWN")

        return 
            
    except Exception as e:
        app_logger.error("labelguard", f"Error calling LLM for sentence classification: {str(e)}")
        return {category: [] for category in categories}
    


def detect_text_block_type(text: str, language: str = "en") -> str:
    """Detect the type of text block - ingredients or other using LLM"""
    if not text:
        return "other"

    model = "gpt-4o-mini"
    
    try:
        # Get OpenAI API key from environment variable
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            app_logger.error("labelguard", "OPENAI_API_KEY environment variable not set")
            return "other"
        
        client = OpenAI(api_key=openai_api_key)

        categories = {
            "ingredients": " Text contains ingredient listings/composition. Keywords: 'Склад', 'Съставки', 'Ingrediente', 'Składniki', 'Құрамы', 'ინგრედიენტები', 'Состав', 'Ingredients', 'Composition'",
            "manufacturing_date": "Text contains production date, expiry date, or best before date.  Keywords: 'Дата виготовлення', 'Дата изготовления', 'Жасалған күні', 'Data fabricatiei', 'Data wyprodukowania', 'Датата на производство', 'Краще спожити до', 'Годен до', 'A se consuma, de preferință, înainte de', 'Дейін қолдану', 'Най-добър до', 'Najlepiej spożyć przed', 'Best before', 'Use by', 'Expiry date', 'Production date'",
            "nutrition" : "Text contains nutritional information, nutritional values, or nutritional declaration. Keywords: 'Поживна цінність', 'Пищевая ценность', 'тағамдық құндылығы', 'Declaratie nutritionala', 'Хранителни стойности', 'Wartość odżywcza', 'Енергетична цінність', 'Энергетическая ценность', 'Қуаттық құндылығы', 'Valoare energetica', 'Енергийна стойност', 'Wartość energetyczna', 'ენერგეტიკული ღირებულება', 'Жири', 'Жиры', 'Майлар', 'Grăsimi', 'Мазнини', 'Tłuszcz', 'Вуглеводи', 'Углеводы', 'Көмірсулар', 'Glucide', 'Węglowodany', 'Білки', 'Белки', 'Нәруыздар', 'Proteine', 'Белтъци', 'Białko', 'Сіль', 'Соль', 'Тұз', 'Sare', 'Сол', 'Sól', 'kcal', 'kJ', 'Nutrition facts', 'Nutritional information'",
            "other": "Text does not fit the above categories"
        }

        # Build categories description
        categories_desc = "\n".join([
            f"- {key}: {description}" 
            for key, description in categories.items()
        ])
        
        prompt = f"""Classify this text into one of three categories for language - {language}.

        Text: "{text}"

        Categories:
        {categories_desc}

        The text may contain multiple languages, not only the ones listed here.

        Respond with only one word: "ingredients", "manufacturing_date", or "other"."""


        result = llm_cache.get(prompt, model)
        if result:
            app_logger.info("labelguard", f"Type detection LLM prompt: {prompt} cashed response: {result}")
        else:
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            
            result = response.choices[0].message.content.strip().lower()
            llm_cache.set(prompt, model, result)

            app_logger.info("labelguard", f"LLM response: {result}")
        
        # Validate response
        return result
        
            
    except Exception as e:
        app_logger.error("labelguard", f"Error calling LLM for type detection: {str(e)}")
        return "other"

def _match_to_dict(match: Match) -> dict:
    """Convert Match object to dictionary for JSON serialization"""
    return {
        "text": match.text,
        "len_a": match.len_a,
        "len_b": match.len_b,
        "start_a": match.start_a,
        "start_b": match.start_b
    }


def _dict_to_match(match_dict: dict) -> Match:
    """Convert dictionary back to Match object"""
    return Match(
        text=match_dict["text"],
        len_a=match_dict["len_a"],
        len_b=match_dict["len_b"],
        start_a=match_dict["start_a"],
        start_b=match_dict["start_b"]
    )


def clean_html_text(text: str) -> str:
    """Clean HTML text using regex + built-in html.unescape"""
    import html
    # First decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Clean up whitespace and special characters
    text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
    text = re.sub(r'\s+([,.;:()])', r'\1', text)  # Remove spaces before punctuation
    text = text.replace(' ', ' ')  # Non-breaking spaces
    text = text.replace('–', '-')  # En-dash to hyphen
    
    return text.strip()


def _get_etalon_text_blocks(kmat: str, version: str, label_image_path: str) -> List[dict]:
    # Read and clean JSON file to handle control characters
    
    etalon_path = label_image_path.parent / f'{label_image_path.stem}_etalon.json'
    with open(etalon_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove control characters that cause JSON parsing issues
    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
    
    content = content.replace('°С', '°C')
    etalon_bank = json.loads(content)

    # Convert empty string values in LANGUAGES field to None and strip whitespace
    for item in etalon_bank:
        if 'LANGUAGES' in item:
            lang = item['LANGUAGES']
            # Convert empty string to None, otherwise strip whitespace
            item['LANGUAGES'] = None if lang == '' else lang.strip()

    return etalon_bank

if __name__ == "__main__":
    # Process all JPG files in the LabelGuard dataset directory
    dataset_dir = ROOT_DIR / "assets" / "datasets" / "LabelGuard" 
    jpg_files = list(dataset_dir.glob("Yummi-Gummi_Cheesecakes_70g_VKF_v271124C__Text.jpg"))

    
    test_labels = []
    for i, image_path in enumerate(jpg_files):
        kmat = f"TEST{i+1:03d}_{image_path.stem.split('_')[0]}"
        test_labels.append(
            LabelInput(
                kmat=kmat,
                version="v1.0", 
                label_image=Image.open(image_path),
                label_image_path=image_path
            )
        )

    results = process_labels(test_labels)

def strip_diacritics(s):
    # 1) decompose
    s2 = unicodedata.normalize('NFD', s)
    # 2) remove all non‐spacing marks
    return ''.join(ch for ch in s2 if unicodedata.category(ch) != 'Mn')


    """
1 вирбник
2 страна походження
3 двтв виготовлення
4 енергетична цінність
5 аліргени (жирні, однакова кількість як в еталоні)
6 аліргенна фраза (жирна)
7 числа маюсть спіпадати в кожній мові(і кількість також)
    """
    """
    Task: Implement Three-Level Progressive Disclosure UI for Label Validation Results
Overview
Create a three-level error reporting interface that allows designers to progressively explore validation results from high-level summary to detailed error analysis.
Level 1: Check Summary (Landing Page)
Purpose: Instant health check - pass/fail decision at a glance
Display:

Total critical errors count (🔴)
Total warnings count (🟡)
Total blocks validated successfully (✅)
Overall score percentage with threshold indicator
Single CTA button to view detailed report

Example:
╔═══════════════════════════════════════════════╗
║  Label Validation Report                     ║
║  Yummi-Gummi Cheesecakes v271124C            ║
╠═══════════════════════════════════════════════╣
║                                               ║
║     🔴 8 Critical Errors                      ║
║     🟡 0 Warnings                             ║
║     ✅ 24 Blocks OK                           ║
║                                               ║
║     Overall Score: 75% ⚠️                     ║
║     (Minimum required: 90%)                   ║
║                                               ║
║         [View Detailed Report →]              ║
╚═══════════════════════════════════════════════╝
Technical Requirements:

Single HTML page generation
Calculate overall score from all block scores
Color coding: green (>90%), yellow (70-90%), red (<70%)


Level 2: Check Details (Error List)
Purpose: Prioritized list of all errors with scores and thresholds
Display:

Numbered list of all errors
Each error shows:

Block number and type (e.g., "ingredients", "nutrition_table")
Language code (e.g., UA, BG, RO)
Error type (text matching, number matching, missing allergen, etc.)
Score vs. threshold (e.g., "87% (required: >90%)")
Clickable "View Details" button



Example:
║  1. Block #3 "ingredients" (BG)                           ║
║     ❌ Text matching score: 87% (required: >90%)          ║
║     [View Details →]                                      ║
║                                                           ║
║  2. Block #5 "ingredients" (UA)                           ║
║     ❌ Text matching score: 57% (required: >90%)          ║
║     [View Details →]                                      ║
Technical Requirements:

Generate from List[LabelError]
Group errors by block number
Sort by severity (critical first) then by score (worst first)
Each error links to Level 3 with specific error ID


Level 3: Super Details (Two-Column Interactive View)
Purpose: Deep dive into specific error with visual + textual comparison
Layout:
┌─────────────────────────────────┬────────────────────────────────────┐
│  IMAGE COLUMN (50%)             │  TEXT COLUMN (50%)                 │
│  - Zoomable/pannable canvas     │  - Error description               │
│  - Red rectangles on blocks     │  - Template text                   │
│  - Light red highlight on words │  - Extracted text                  │
│  - Block navigation [1][3][5]   │  - Difference highlights           │
│                                 │  - Navigation: prev/next/back      │
└─────────────────────────────────┴────────────────────────────────────┘
Image Column Features:

Display overlay image (from generate_error_overlay_image())
Canvas with zoom/pan controls (mouse wheel + drag)
Current error block highlighted with darker/pulsing border
Clickable block numbers to jump between errors
Error words have light red background overlay
    """