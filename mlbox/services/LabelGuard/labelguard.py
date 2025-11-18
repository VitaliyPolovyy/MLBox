from dataclasses import dataclass
from pathlib import Path
from typing import List
from collections import deque
import json
from jinja2 import Template
import ast
from PIL import Image
from mlbox.utils.logger import get_logger, get_artifact_service
from mlbox.utils.llm_cache import LLMCache
from mlbox.utils.lcs import all_common_substrings_by_words, Match, highlight_matches_by_words_html
from openai import OpenAI
import os
import re
import unicodedata
import glob
import base64
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

#print(f"DEBUG: GOOGLE_APPLICATION_CREDENTIALS = {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")


from mlbox.services.LabelGuard.layout_detector import LayoutTextBlock, LayoutDetector
from mlbox.services.LabelGuard.ocr_processor import OCRResult, OCRWord, VisionOCRProcessor
from mlbox.services.LabelGuard.datatypes import (
    LabelInput,
    Sentence,
    TextBlock,
    RulesName,
    VisualMarker,
    RuleCheckResult,
    LabelProcessingResult,
    CategoryNumberResult,
    NumbersCheckResult
)
from mlbox.settings import ROOT_DIR, LOG_LEVEL


from pydantic import BaseModel

class TextBlockDetection(BaseModel):
    text: str
    category: str
    language: str

class TextBlockDetectionList(BaseModel):
    blocks: list[TextBlockDetection]


CURRENT_DIR = Path(__file__).parent
LLM_MODEL = "qwen/qwen-2.5-72b-instruct"
SERVICE_NAME = "labelguard"
CATEGORIES_TO_CHECK = ["A"]

app_logger = get_logger(ROOT_DIR)
artifact_service = get_artifact_service(ROOT_DIR)
# LLM cache goes in cache/ subdirectory
llm_cache = LLMCache(artifact_service.get_service_dir(SERVICE_NAME) / "cache")


# Language mapping helpers
def _load_language_mapping() -> dict:
    """Load language mapping from JSON file (all codes in lowercase)"""
    mapping_path = ROOT_DIR / "assets" / "labelguard" / "json" / "language-mapping.json"
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get("etalon_to_bcp47", {})
    except Exception as e:
        app_logger.warning(SERVICE_NAME, f"Failed to load language mapping: {e}. Using empty mapping.")
        return {}

def _get_reverse_mapping(etalon_to_bcp47: dict) -> dict:
    """Generate reverse mapping (bcp47_to_etalon) from etalon_to_bcp47"""
    return {v: k for k, v in etalon_to_bcp47.items()}

# Cache the mapping at module level
# _load_language_mapping() returns etalon_to_bcp47 from JSON
LANGUAGE_MAPPING_ETALON_TO_BCP47 = _load_language_mapping()
# Reverse mapping: bcp47_to_etalon
LANGUAGE_MAPPING_BCP47_TO_ETALON = _get_reverse_mapping(LANGUAGE_MAPPING_ETALON_TO_BCP47)


# Text block classification categories
def load_text_block_categories():
    """Load text block categories from JSON file"""
    categories_file = ROOT_DIR / "assets" / "labelguard" / "json" / "data-categories.json"
    with open(categories_file, 'r', encoding='utf-8') as f:
        return json.load(f)

TEXT_BLOCK_CATEGORIES = load_text_block_categories()

def find_etalon_text(block_type : str, language : str, etalon_text_blocks : List[dict]) -> str:
    """
    Find etalon text matching block type and language.
    
    Args:
        block_type: Type of text block
        language: BCP47 language code from OCR (lowercase, e.g., "uk", "ru", "bs")
        etalon_text_blocks: List of etalon text blocks with LANGUAGES in uppercase (e.g., "UK", "RU", "BA")
    
    Returns:
        Matching etalon text or empty string
    """
    etalon_text = ""
    
    for etalon_text_block in etalon_text_blocks:
        # Etalon codes are stored in uppercase (e.g., "UK", "RU", "BA")
        etalon_lang = etalon_text_block['LANGUAGES'].strip().lower() if etalon_text_block['LANGUAGES'] else None
        etalon_lang_bcp47 = LANGUAGE_MAPPING_ETALON_TO_BCP47.get(etalon_lang)
        
        if etalon_text_block['type_'] == block_type and (etalon_lang_bcp47 == language or etalon_lang is None):
            etalon_text = clean_html_text(etalon_text_block['text'])
            break

    return etalon_text if etalon_text else ""

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
            index=str(int(json_vision_filename.split("_")[-2]))
        )
        layout_blocks.append(layout_block)

    return layout_blocks

def split_words_into_sentences(words: List[OCRWord], language: str, delimiters : List[str] = ['.', '!', '?']):
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
        sentence.text = refine_text_symbols(sentence.text)

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

def rule_check_text_matches_etalon(label_processing_result: LabelProcessingResult) -> List[RuleCheckResult]:
    """Detect text that appears in OCR but not in etalon text"""
    rule_check_results = []
    app_logger.info(SERVICE_NAME, f"Starting error detection for {len(label_processing_result.text_blocks)} text blocks")
   
   # to do: check only blocks which category in list category to check (CATEGORIES_TO_CHECK)
    for text_block in label_processing_result.text_blocks:
       
        if len(CATEGORIES_TO_CHECK) > 0 and text_block.type not in CATEGORIES_TO_CHECK:
            continue
       
        # Find words that are not covered by LCS matches
        words = []
        for sentence in text_block.sentences:
            words.extend(sentence.words)

        lcs_results = []
        text = text_block.text.replace("\n", " ")

        if text_block.etalon_text and text_block.text:
            lcs_results = all_common_substrings_by_words(
                text, 
                text_block.etalon_text, 
                min_length_words=2, 
                maximal_only=True,
                ignorable_symbols=",.()!:;- "
            )

        # Get unmatched text positions (text that doesn't match etalon)
        unmatched_ranges = get_unmatched_positions(text, lcs_results)

        error_words = get_words_by_char_position(words, unmatched_ranges)
        error_words = [word for word in error_words if word.text not in [",", ".", "(", ")", "[", "]", "{", "}", " ", ".", ",", "!", "?", ":", ";", "\n"]]

        # Calculate score based on word matching
        etalon_words = text_block.etalon_text.split() if text_block.etalon_text else []
        total_words = len(etalon_words)
        missing_words = len(error_words)
        
        # Calculate score: 100 - (missing_words / total_words * 100)
        if total_words > 0:
            score = 100 - (missing_words / total_words * 100)
        else:
            score = 100.0
        
        score = max(0, min(100, score))  # Clamp between 0 and 100
        threshold = 90.0
        passed = (score >= threshold)

        # Create score expression for visualization
        score_expression = f"100 - {missing_words} / {total_words} * 100" if total_words > 0 else "100"

        # Create visual markers (only for specific items, not text block outline)
        visual_markers = []
        
        if error_words:
            # Add light red highlight on each error word
            for word in error_words:
                for bbox in word.bbox:
                    # Convert from text block coordinates to full image coordinates
                    full_bbox = (
                        text_block.bbox[0] + bbox[0],  # x1
                        text_block.bbox[1] + bbox[1],  # y1
                        text_block.bbox[0] + bbox[2],  # x2
                        text_block.bbox[1] + bbox[3]   # y2
                    )
                    visual_markers.append(VisualMarker(
                        type="highlight",
                        bbox=full_bbox,
                        color=(255, 100, 100),
                        opacity=0.4
                    ))
        
        # Always create result (even when no errors)
        result = RuleCheckResult(
            rule_name=RulesName.ETALON_MATCHING,
            scope="block",
            text_block=text_block,
            affected_words=error_words,
            visual_markers=visual_markers,
            passed=passed,
            score=score,
            threshold=threshold,
            score_expression=score_expression
        )
        
        # Generate HTML details only if there are errors
        if error_words:
            html_parts = []
            html_parts.append("<hr style='margin: 4px 0; border: 0; border-top: 1px solid #ddd;'>")
            etalon = text_block.etalon_text or ''
            status_emoji = "✅" if passed else "❌"
            etalon_highlighted = highlight_matches_by_words_html(etalon, lcs_results, use_start_a=True) if etalon else ''
            
            html_parts.append(f"<div style='margin-top:1px;'><strong>{status_emoji} Перевірка відповідності тексту ({score:.0f}/100):</strong></div>")
            
            if total_words > 0:
                if error_words and len(error_words) > 0:
                    missing_words_text = ', '.join([word.text for word in error_words])
                    html_parts.append(f"<div style='font-size:0.9em; color:#666;'><strong>відсутні слова:</strong> {missing_words_text}</div>")
            
            html_parts.append(f"<div style='margin-top:10px; font-size:12px;'><strong>Еталон:</strong><br>{etalon_highlighted}</div>")
            
            result.html_details = ''.join(html_parts)
        
        # Always append result
        rule_check_results.append(result)
    
    return rule_check_results

def rule_check_allergens(label_processing_result: LabelProcessingResult) -> List[RuleCheckResult]:
    """
    Detect allergen count mismatches between text blocks and etalon.
    
    Logic:
    1. Get etalon allergen count from etalon_allergen_words
    2. Compare each block's allergen count with etalon count
    3. Add errors when counts don't match
    """
    errors = []
    
    # Get etalon allergen count
    uk_blocks = [block for block in label_processing_result.text_blocks if block.languages == ["uk"] and block.type == "A"]
    if not uk_blocks:
        return errors
    
    uk_block = uk_blocks[0]
    etalon_allergen_words = uk_block.allergens
    etalon_count = len(etalon_allergen_words)
    
    # If no etalon, skip validation
    if etalon_count == 0:
        return errors
    
    # Get all text blocks that have allergens
    text_blocks_with_allergens = [
        block for block in label_processing_result.text_blocks 
        if block.allergens and len(block.allergens) > 0
    ]
    
    if not text_blocks_with_allergens:
        return errors
    
    # Compare each block's allergen count with etalon count
    for text_block in text_blocks_with_allergens:
        current_count = len(text_block.allergens)
        
        passed = (current_count == etalon_count)
        score = (current_count / etalon_count * 100) if etalon_count > 0 else 100.0
        
        # Create visual markers for allergens
        visual_markers = []
        allergen_names = [allergen.text for allergen in text_block.allergens]
        
        # Highlight each allergen word
        for allergen in text_block.allergens:
            for bbox in allergen.bbox:
                # Convert from text block coordinates to full image coordinates
                full_bbox = (
                    text_block.bbox[0] + bbox[0],
                    text_block.bbox[1] + bbox[1],
                    text_block.bbox[0] + bbox[2],
                    text_block.bbox[1] + bbox[3]
                )
                visual_markers.append(VisualMarker(
                    type="highlight",
                    bbox=full_bbox,
                    color=(255, 255, 100),
                    opacity=0.3
                ))
        
        result = RuleCheckResult(
            rule_name=RulesName.ALLERGENS,
            scope="block",
            text_block=text_block,
            passed=passed,
            score=score,
            threshold=100.0,
            affected_words=text_block.allergens,
            visual_markers=visual_markers,
            metadata={
                'expected_count': etalon_count,
                'actual_count': current_count,
                'allergen_names': allergen_names
            }
        )
        
        # Generate HTML details
        html_parts = []
        html_parts.append("<hr style='margin: 4px 0; border: 0; border-top: 1px solid #ccc;'>")
        status_emoji = "✅" if passed else "❌"
        
        html_parts.append(f"<div style='margin-top:1px;'><strong>{status_emoji} Алергени</strong></div>")
        html_parts.append(f"<div>{', '.join(allergen_names) if allergen_names else 'немає'}</div>")
        
        if not passed:
            html_parts.append(f"<div style='color:#d00;'>в наявності {current_count} очікую {etalon_count}</div>")
        
        result.html_details = ''.join(html_parts)
        errors.append(result)
    
    return errors


def extract_numbers_from_text(text: str) -> List[str]:
    """
    Extract numbers from text using specific patterns.
    
    Patterns (in priority order):
    1. Temperature ranges: 18±3, 18.5±3.2
    2. Percentages: 75%, 4.5%
    3. Decimals: 4.5, 1,5
    4. Integers: 51
    
    Excludes: E-numbers like E331
    
    Args:
        text: Text to extract numbers from
        
    Returns:
        List of extracted numbers as strings
    """
    numbers = []
    
    # Pattern 1: Temperature ranges with ± (e.g., 18±3, 18.5±3.2)
    # Must match digit±digit pattern, allowing optional spaces around ±
    temp_pattern = r'\d+(?:[.,]\d+)?\s*±\s*\d+(?:[.,]\d+)?'
    temp_matches = re.findall(temp_pattern, text)
    
    # Remove matched temperature ranges from text to avoid duplicate extraction
    for match in temp_matches:
        text = text.replace(match, ' ')
    
    # Normalize matches by removing spaces for consistency
    temp_matches = [m.replace(' ', '') for m in temp_matches]
    numbers.extend(temp_matches)
    
    # Pattern 2: Percentages (e.g., 75%, 4.5%)
    percent_pattern = r'\d+(?:[.,]\d+)?%'
    percent_matches = re.findall(percent_pattern, text)
    numbers.extend(percent_matches)
    
    # Remove matched percentages
    for match in percent_matches:
        text = text.replace(match, ' ')
    
    # Pattern 3: Decimals (e.g., 4.5, 1,5)
    decimal_pattern = r'\d+[.,]\d+'
    decimal_matches = re.findall(decimal_pattern, text)
    numbers.extend(decimal_matches)
    
    # Remove matched decimals
    for match in decimal_matches:
        text = text.replace(match, ' ')
    
    # Pattern 4: Integers, but exclude E-numbers (e.g., 51, but not 331 from E331)
    # Remove E-numbers first (both Latin E and Cyrillic Е)
    clean_text = re.sub(r'[EЕ]\d+', '', text, flags=re.IGNORECASE)
    clean_text = re.sub(r'(?i)v\d+.', ' ', clean_text)


    integer_pattern = r'\d+'
    numbers.extend(re.findall(integer_pattern, clean_text))
    
    return numbers


def rule_check_numbers(label_processing_result: LabelProcessingResult) -> List[NumbersCheckResult]:
    """
    Validate numbers in INGRIDIENTS and STORAGE_CONDITIONS sentences.
    
    Logic:
    1. Filter text blocks with type='ingredients'
    2. Extract numbers from sentences by category (INGRIDIENTS, STORAGE_CONDITIONS)
    3. Build reference sets (union of all numbers per category across all blocks)
    4. Validate each block's numbers against reference sets
    5. If block has no sentences for a category, actual_numbers = []
    
    Args:
        label_processing_result: Result containing text blocks
        
    Returns:
        List of NumbersCheckResult, one per block that needs validation
    """
    errors = []
    
    # Step 1: Filter blocks with type='ingredients'
    ingredients_blocks = [
        block for block in label_processing_result.text_blocks 
        if block.type == "A"
    ]
    
    if not ingredients_blocks:
        return errors
    
    # Step 2: Extract numbers by category from all blocks (parse only once!)
    categories = ['INGRIDIENTS', 'STORAGE_CONDITIONS', 'PRODUCT_NAME']
    block_numbers = {}  # {block_id: {category: [numbers]}}
    
    reference_sets = {category: [] for category in categories}

    for block in ingredients_blocks:
        block_id = id(block)
        block_numbers[block_id] = {cat: [] for cat in categories}
        
        for sentence in block.sentences:
            if sentence.category in categories:
                extracted = extract_numbers_from_text(sentence.text)
                if sentence.category == "PRODUCT_NAME":
                    block_numbers[block_id]["INGRIDIENTS"].extend(extracted)
                else:
                    block_numbers[block_id][sentence.category].extend(extracted)
        
        if block.languages == ["uk"]:
            reference_sets = {
                category: sorted(list(set(block_numbers[block_id][category])))
                for category in categories
            }


    
    # Step 4: Validate each block
    for block in ingredients_blocks:
        block_id = id(block)
        category_results = []
        passed = True
        
        for category in categories:
            actual = sorted(list(set(block_numbers[block_id][category])))
            reference = reference_sets[category]
            
            # Check if numbers match (skip validation if reference is empty)
            if reference:  # Only validate if reference set has numbers
                if set(actual) != set(reference):
                    passed = False
            
            category_results.append(CategoryNumberResult(
                category=category,
                actual_numbers=actual,
                reference_numbers=reference
            ))
        
        # Create result
        result = NumbersCheckResult(
            rule_name=RulesName.NUMBERS_IN_INGRIDIENTS,
            scope="block",
            text_block=block,
            passed=passed,
            category_results=category_results
        )
        
        # Generate HTML details
        html_parts = []
        html_parts.append("<hr style='margin: 4px 0; border: 0; border-top: 1px solid #ccc;'>")
        status_emoji = "✅" if passed else "❌"
        html_parts.append(f"<div style='margin-top:1px;'><strong>{status_emoji} Числа:</strong></div>")
        
        # Collect all numbers across categories
        all_actual = []
        all_missing = []
        all_extra = []
        
        for cat_result in category_results:
            all_actual.extend(cat_result.actual_numbers)
            missing = set(cat_result.reference_numbers) - set(cat_result.actual_numbers)
            extra = set(cat_result.actual_numbers) - set(cat_result.reference_numbers)
            all_missing.extend(missing)
            all_extra.extend(extra)
        
        # Show what's present
        actual_display = ', '.join(all_actual) if all_actual else '-'
        html_parts.append(f"<div>є: {actual_display}</div>")
        
        # Show what's missing (only if there are missing numbers)
        if all_missing:
            html_parts.append(f"<div style='color:#d00;'>відсутні: {', '.join(sorted(all_missing))}</div>")
        
        # Show extra numbers if any
        if all_extra:
            html_parts.append(f"<div style='color:#d00;'>зайві: {', '.join(sorted(all_extra))}</div>")
        
        result.html_details = ''.join(html_parts)
        errors.append(result)
    
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


def validate_label(label_processing_result: LabelProcessingResult) -> List[RuleCheckResult]:
    """Detect all errors in a LabelProcessingResult"""
    rule_check_results = []

    rule_check_results.extend(rule_check_text_matches_etalon(label_processing_result))
    
    rule_check_results.extend(rule_check_allergens(label_processing_result))
    
    rule_check_results.extend(rule_check_numbers(label_processing_result))
        
    
    return rule_check_results

def get_word_bounding_boxes(word: OCRWord) -> List[tuple]:
    """Get all bounding boxes for a word (handles wrapped words)"""
    if not word.bbox:
        return []
    
    # Ensure word.bbox is a list of tuples
    if isinstance(word.bbox, tuple):
        return [word.bbox]
    else:
        return word.bbox

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
    
    label_processing_results: List[LabelProcessingResult] = []
    
    for label in labels:
        app_logger.info(SERVICE_NAME, f"Processing label: {label.kmat} v{label.version}")

        label_processing_result = LabelProcessingResult()
        label_processing_result.kmat = label.kmat
        label_processing_result.version = label.version
        label_processing_result.original_filename = Path(label.label_image_path).stem  # Store original filename without extension
        
        etalon_text_blocks = _get_etalon_text_blocks(label.kmat, label.version, label.label_image_path )
        
        # Step 1: Layout detection
        app_logger.info(SERVICE_NAME, f"Starting layout detection for {label.kmat}")
        layout_blocks = layout_detector.extract_blocks(label.label_image_path)
        #layout_blocks = LoadLayoutBlocks(label.label_image_path)
        
        # Save detected blocks as artifacts (debug files go to temp/)
        if LOG_LEVEL == "DEBUG":
            for i, layout_block in enumerate(layout_blocks):
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=f"temp/{Path(label.label_image_path).stem}_{i}.jpg",
                    data=layout_block.image_crop
                )
            
        
        # Step 2: OCR processing
        app_logger.info(SERVICE_NAME, f"Starting OCR processing for {label.kmat}")
        
        # Extract language hints from etalon files for better OCR accuracy
        language_hints = extract_languages_from_etalon_files(label.label_image_path)
        app_logger.info(SERVICE_NAME, f"Language hints: {language_hints}")

        processing_queue = deque(layout_blocks)
        splitted_text_block_indicies = []
        
        while processing_queue:
            layout_block = processing_queue.popleft()

            json_vision_filename = f"{artifact_service.get_service_dir(SERVICE_NAME) / Path(label.label_image_path).stem}_{layout_block.index}_vision.json"
            ocr_result = ocr_processor.process(
                layout_block.image_crop, 
                input_filename=str(label.label_image_path), 
                json_vision_filename=json_vision_filename,
                language_hints=language_hints
            )
            
            ocr_result.text = refine_text_symbols(ocr_result.text)
            for word in ocr_result.words:
                word.text = refine_text_symbols(word.text)


            #detect text block types and split if needed
            block_type = classify_text_block ( ocr_result)

            if block_type == "Z" and layout_block.index not in splitted_text_block_indicies:
                layout_blocks_splitted = split_text_block_1 (layout_block, ocr_result)
                
                if len(layout_blocks_splitted) > 1:
                    splitted_text_block_indicies.extend([split_block.index for split_block in layout_blocks_splitted])
                    for split_block in layout_blocks_splitted:
                        # Save split block images to temp/ (debug files)
                        artifact_service.save_artifact(
                            service=SERVICE_NAME,
                            file_name=f"temp/{Path(label.label_image_path).stem}_{split_block.index}.jpg",
                            data=split_block.image_crop
                        )
                        processing_queue.appendleft(split_block)
                    continue;
            
            # Save OCR results as text artifacts in DEBUG mode (debug files go to temp/)
            if LOG_LEVEL == "DEBUG":
                input_name = Path(label.label_image_path).stem
            
                text_filename = f"temp/{input_name}_block_{i}.txt"
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=text_filename,
                    data=f"Confidence: {ocr_result.confidence:.2f}\nLanguage: {ocr_result.language}\nExtracted Text:\n{ocr_result.text}"
                )
            # Step 3: fill in LabelProcessingResult.text_blocks
            allergens = ""
           
            if ocr_result.language == "hy": # armenian
                delimiters = [':', '!', '?',]
            else:
                delimiters = ['.', '!', '?']
            sentences = split_words_into_sentences (ocr_result.words, ocr_result.language, delimiters)
            if block_type == "A":
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
                lcs_results = all_common_substrings_by_words(
                    etalon_text, 
                    ocr_result.text, 
                    min_length_words=2, 
                    maximal_only=True,
                    ignorable_symbols=",.()!:;- "
                )
            
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
        
            label_processing_result.text_blocks.append(text_block)
       
        label_processing_result.rule_check_results = validate_label(label_processing_result)
        
        app_logger.info(SERVICE_NAME, f"Found {len(label_processing_result.rule_check_results)} errors for {label.kmat}")

        
        # Step 5: Generate validation report
        from mlbox.services.LabelGuard import visualization
        
        validation_artifacts = visualization.generate_validation_report(
            label_processing_result=label_processing_result,
            label_image=label.label_image,
            output_format="interactive_viewer"
        )
        
        # Step 6: Save artifacts to disk using filenames from visualization
        for filename, image in validation_artifacts.images:
            artifact_service.save_artifact(
                service=SERVICE_NAME,
                file_name=filename,
                data=image
            )
        
        if validation_artifacts.html_report:
            artifact_service.save_artifact(
                service=SERVICE_NAME,
                file_name=validation_artifacts.html_filename,
                data=validation_artifacts.html_report
            )
            # Assign HTML report to the result so it can be returned via API
            label_processing_result.html_report = validation_artifacts.html_report
        
        label_processing_results.append(label_processing_result)

    
    return label_processing_results


def mask_bboxes_in_image(image: Image.Image, bboxes: List[tuple]) -> Image.Image:
    """
    Mask provided bbox areas with white rectangles.
    
    Args:
        image: PIL Image to mask
        bboxes: List of bbox tuples (x1, y1, x2, y2)
        
    Returns:
        New PIL Image with white rectangles drawn over bbox areas
    """
    import cv2
    import numpy as np
    
    # Convert PIL to OpenCV format
    img_array = np.array(image.convert('RGB'))
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # Draw white rectangles over each bbox
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
    # Convert back to PIL
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


def calculate_bbox_overlap(bbox1: tuple, bbox2: tuple) -> float:
    """
    Calculate overlap percentage between two bboxes.
    
    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)
        
    Returns:
        Overlap percentage (0.0 to 1.0)
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    if bbox2_area == 0:
        return 0.0
    
    return intersection_area / bbox2_area


def analyze(request_json: dict) -> LabelProcessingResult:
    """
    Unified analyze function that handles both initial detection and re-analysis with corrections.
    
    According to spec:
    - If blocks is empty → normal layout detection
    - If blocks provided → mask provided bbox areas (white rectangles), run layout detection on masked image,
      filter detected bboxes (overlap > 10% with provided), merge provided + filtered detected bboxes
    - Always run full pipeline (OCR, text parsing, category detection, validation)
    
    Args:
        request_json: Dict with:
            - 'image_path' (str): Path to image file (e.g., '/artifacts/labelguard/filename.jpg')
            - 'etalon_path' (str, optional): Path to etalon image file (will be OCR'd to extract text blocks)
            - 'blocks' (list): List of block dicts or empty list
                Block format: [{"id": str, "bbox": [x1,y1,x2,y2], "category": str, "text": str, "modified": bool}, ...]
            - 'kmat' (str, optional): Product code
            - 'version' (str, optional): Version string
            - Note: If etalon_path not provided, will try to load from {image_stem}_etalon.json (legacy)
            
    Returns:
        LabelProcessingResult with enriched blocks and validation results
    """
    image_path_str = request_json["image_path"]
    etalon_path_str = request_json.get("etalon_path")  # Optional: path to etalon image
    blocks_data = request_json.get("blocks", [])
    kmat = request_json.get("kmat", "UNKNOWN")
    version = request_json.get("version", "v1.0")
    
    # Load image
    # image_path_str can be:
    # - First call: path to uploaded image (e.g., "assets/labelguard/datasets/...")
    # - Subsequent calls: path to artifact (e.g., "/artifacts/labelguard/{filename}.jpg")
    # The image must exist at this path (ERP/uploads should save it before calling analyze)
    image_path = ROOT_DIR / image_path_str.lstrip('/')  # Remove leading slash if present
    if not image_path.exists():
        app_logger.error(SERVICE_NAME, f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    label_image = Image.open(image_path)
    original_filename = Path(image_path_str).stem
    
    app_logger.info(SERVICE_NAME, f"Analyzing: kmat={kmat}, version={version}, blocks_count={len(blocks_data)}")
    
    # Initialize processors
    layout_detector = LayoutDetector()
    ocr_processor = VisionOCRProcessor()
    
    # Step 1: Collect bboxes
    provided_blocks = []
    layout_blocks_input : List[LayoutTextBlock] = []
    image = label_image.copy()
    
    if blocks_data:
        # Create lightweight TextBlocks from provided blocks
        for block_dict in blocks_data:
            block_bbox = tuple(block_dict.get("bbox", []))
            layout_blocks_input.append(LayoutTextBlock(
                index=block_dict.get("id", ""),
                bbox=block_bbox,
                image_crop=label_image.crop(block_bbox),
                label=block_dict.get("category", ""),
                score=1
            ))

        # to do: куьщму all input bboxes from label_image leave blank space on their places
        for layout_block in layout_blocks_input:
            x1, y1, x2, y2 = layout_block.bbox
            image.paste(Image.new('RGB', (x2 - x1, y2 - y1), (255, 255, 255)), (x1, y1))
        
        
    detected_layout_blocks = layout_detector.extract_blocks(image)
    detected_layout_blocks.extend(layout_blocks_input)
    
    # Convert to TextBlocks
    layout_blocks_to_process = []
    for layout_block in detected_layout_blocks:
        text_block = TextBlock(
            bbox=layout_block.bbox,
            index=layout_block.index,
            sentences=[],
            text="",
            type="",
            allergens=[],
            languages="",
            modified=False
        )
        layout_blocks_to_process.append(text_block)
    
    # Step 2: Get etalon data
    if etalon_path_str:
        # Process etalon image: OCR to extract text blocks
        app_logger.info(SERVICE_NAME, "No etalon_path provided, trying to load from JSON file")
        etalon_text_blocks = _get_etalon_text_blocks(kmat, version, etalon_path_str)
    
    # Step 3: OCR processing and enrichment
    app_logger.info(SERVICE_NAME, f"Starting OCR processing for {len(layout_blocks_to_process)} blocks")
    
    # Extract language hints from etalon files
    language_hints = extract_languages_from_etalon_files(image_path)
    app_logger.info(SERVICE_NAME, f"Language hints: {language_hints}")
    
    label_processing_result = LabelProcessingResult()
    label_processing_result.kmat = kmat
    label_processing_result.version = version
    label_processing_result.original_filename = original_filename
    
    # Process each block
    for text_block in layout_blocks_to_process:
        # Crop image by bbox
        x1, y1, x2, y2 = text_block.bbox
        cropped_image = label_image.crop((x1, y1, x2, y2))
        
        # Run OCR (or use provided text if exists and not empty)
        if text_block.text and text_block.text.strip():
            # Use provided text, but still need to run OCR for word-level data
            app_logger.info(SERVICE_NAME, f"Block {text_block.index}: Using provided text, running OCR for word-level data")
        
        # Vision cache files go in cache/ subdirectory
        cache_dir = artifact_service.get_service_dir(SERVICE_NAME) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        json_vision_filename = str(cache_dir / f"{original_filename}_{text_block.index}_vision.json")
        ocr_result = ocr_processor.process(
            cropped_image,
            input_filename=str(image_path),
            json_vision_filename=json_vision_filename,
            language_hints=language_hints
        )
        
        # Refine text symbols
        ocr_result.text = refine_text_symbols(ocr_result.text)
        for word in ocr_result.words:
            word.text = refine_text_symbols(word.text)
        
        # Use provided text if available, otherwise use OCR text
        if text_block.text and text_block.text.strip():
            # Keep provided text but update OCR words
            final_text = text_block.text
        else:
            final_text = ocr_result.text
            text_block.text = final_text
        
        # Detect category if empty
        if not text_block.type or text_block.type.strip() == "":
            text_block.type = classify_text_block(ocr_result)
        
        # Parse sentences
        if ocr_result.language == "hy":  # armenian
            delimiters = [':', '!', '?']
        else:
            delimiters = ['.', '!', '?']
        sentences = split_words_into_sentences(ocr_result.words, ocr_result.language, delimiters)
        
        # Classify sentences if type is A (ingredients)
        if text_block.type == "A":
            classify_ingredients_sentences(sentences)
            # Detect allergens
            try:
                ingredients_sentence_index = [sentence.index for sentence in sentences if sentence.category == "INGRIDIENTS"][0]
                for i, word in enumerate(sentences[ingredients_sentence_index].words):
                    if word.text == ':':
                        allergens = [word for word in sentences[ingredients_sentence_index].words[i + 1:] if word.bold and word.text != " "]
                        break
                else:
                    allergens = []
            except IndexError:
                allergens = []
        else:
            allergens = []
        
        # Find etalon text
        etalon_text = find_etalon_text(text_block.type, ocr_result.language, etalon_text_blocks)
        lcs_results = []
        if etalon_text and final_text:
            lcs_results = all_common_substrings_by_words(
                etalon_text,
                final_text,
                min_length_words=2,
                maximal_only=True,
                ignorable_symbols=",.()!:;- "
            )
        
        # Update text block with enriched data
        text_block.sentences = sentences
        text_block.text = final_text
        text_block.allergens = allergens
        text_block.languages = [ocr_result.language]
        text_block.etalon_text = etalon_text
        text_block.lcs_results = lcs_results
        
        label_processing_result.text_blocks.append(text_block)
    
    # Step 4: Run validation
    app_logger.info(SERVICE_NAME, f"Running validation on {len(label_processing_result.text_blocks)} blocks")
    label_processing_result.rule_check_results = validate_label(label_processing_result)
    
    app_logger.info(SERVICE_NAME, f"Analysis complete: {len(label_processing_result.rule_check_results)} validation results")
    
    # Step 5: Save JSON to artifacts directory
    artifacts_dir = artifact_service.get_service_dir(SERVICE_NAME)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    result_json = label_processing_result.to_json()
    
    # Validate that result_json is fully serializable before saving
    try:
        json.dumps(result_json)  # Test serialization
    except (TypeError, ValueError) as e:
        app_logger.error(SERVICE_NAME, f"Error: result_json contains non-serializable objects: {e}")
        # Try to find and fix the issue
        import json as json_lib
        def find_non_serializable(obj, path=""):
            """Recursively find non-serializable objects"""
            if isinstance(obj, dict):
                for k, v in obj.items():
                    find_non_serializable(v, f"{path}.{k}" if path else k)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    find_non_serializable(v, f"{path}[{i}]" if path else f"[{i}]")
            else:
                try:
                    json_lib.dumps(obj)
                except (TypeError, ValueError):
                    app_logger.error(SERVICE_NAME, f"Non-serializable object found at {path}: {type(obj)} = {obj}")
        find_non_serializable(result_json)
        raise
    
    output_data = {
        "image_path": f"/artifacts/labelguard/{original_filename}.jpg",
        "labelProcessingResult": result_json
    }
    
    json_file = artifacts_dir / f"{original_filename}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    app_logger.info(SERVICE_NAME, f"Saved JSON to: {json_file}")
    
    # Smart image copying: only copy if source is different from target
    image_file = artifacts_dir / f"{original_filename}.jpg"
    
    # Check if image is already in artifacts at the correct location
    if image_file.exists() and image_path.resolve() == image_file.resolve():
        # Image is already in the right place, skip copy
        app_logger.debug(SERVICE_NAME, f"Image already in artifacts: {image_file}")
    elif not image_file.exists() or image_path.resolve() != image_file.resolve():
        # Image needs to be copied (either doesn't exist in artifacts, or source is different)
        from shutil import copyfile
        copyfile(str(image_path), str(image_file))
        app_logger.info(SERVICE_NAME, f"Copied image to: {image_file}")
    
    return label_processing_result


def analyze_corrected_blocks(request_json: dict) -> LabelProcessingResult:
    """
    Re-analyze label with user corrections.
    Skips layout detection, runs OCR on each corrected bbox.
    
    According to spec:
    1. SKIP layout detection
    2. SKIP block splitting
    3. For each block: Run OCR on bbox → extract text (or use provided text if exists)
    4. Parse text → create sentences → create words with bboxes and bold flags
    5. Build complete TextBlock structure (enrichment), preserving modified flag
    6. Run category detection if category is empty/None
    7. Reconstruct LabelProcessingResult from enriched blocks
    8. Run validation using validate_label()
    9. Generate updated validation results
    
    Args:
        request_json: Dict with 'image_path' (str) and 'blocks' (list of dicts)
            blocks format: [{"id": str, "bbox": [x,y,w,h], "category": str, "text": str, "modified": bool}, ...]
        
    Returns:
        LabelProcessingResult with enriched blocks and updated validation results
    """
    def create_label_processing_result_from_json(request_json: dict) -> LabelProcessingResult:
        """
        Create initial LabelProcessingResult from JSON request.
        Creates lightweight TextBlocks (without OCR enrichment yet).
        """
        image_path_str = request_json["image_path"]
        blocks_data = request_json["blocks"]
        
        # Create result object with metadata
        label_processing_result = LabelProcessingResult()
        label_processing_result.kmat = request_json.get("kmat", "UNKNOWN")
        label_processing_result.version = request_json.get("version", "v1.0")
        label_processing_result.original_filename = Path(image_path_str).stem
        
        # Create lightweight TextBlocks from JSON blocks
        for block_dict in blocks_data:
            block_id = block_dict.get("id", "")
            block_bbox = tuple(block_dict.get("bbox", []))
            block_category = block_dict.get("category", "")
            block_text = block_dict.get("text", "")
            block_modified = block_dict.get("modified", False)
            
            # Create lightweight TextBlock (will be enriched later with OCR)
            text_block = TextBlock(
                bbox=block_bbox,
                index=block_id,
                sentences=[],  # Will be populated by OCR
                text=block_text,  # May be empty, will be filled by OCR if empty
                type=block_category,  # May be empty, will be detected if empty
                allergens=[],  # Will be populated during enrichment
                languages="",  # Will be detected during OCR
                modified=block_modified
            )
            
            label_processing_result.text_blocks.append(text_block)
        
        return label_processing_result
    
    # Step 1: Create initial result from JSON
    label_processing_result = create_label_processing_result_from_json(request_json)
    
    image_path_str = request_json["image_path"]
    blocks_data = request_json["blocks"]
    
    app_logger.info(SERVICE_NAME, f"Analyzing {len(blocks_data)} corrected blocks")
    
    # Step 2: Load image from path
    image_path = ROOT_DIR / image_path_str
    if not image_path.exists():
        app_logger.error(SERVICE_NAME, f"Image not found: {image_path}")
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    label_image = Image.open(image_path)
    app_logger.info(SERVICE_NAME, f"Loaded image from {image_path}")
    
    # Initialize processors
    ocr_processor = VisionOCRProcessor()
    
    # Step 3-6: Enrich each block with OCR
    for text_block in label_processing_result.text_blocks:
        app_logger.info(SERVICE_NAME, f"Processing block {text_block.index}: bbox={text_block.bbox}, category={text_block.type}")
        
        # TODO: Implement OCR processing per block
        # - Crop image by bbox
        # - Run OCR (or use provided text if exists)
        # - Parse sentences and words
        # - Detect category if empty
        # - Update text_block with enriched data
    
    # Step 7-9: Run validation
    app_logger.info(SERVICE_NAME, f"Running validation on {len(label_processing_result.text_blocks)} blocks")
    label_processing_result.rule_check_results = validate_label(label_processing_result)
    
    app_logger.info(SERVICE_NAME, f"Analysis complete: {len(label_processing_result.rule_check_results)} validation results")
    
    return label_processing_result


def detect_allergens(words: List[str], language: str = "en") -> List[str]:
    """Detect allergens from bold text using LLM"""
    if not words:
        return []

    try:
        # Get OpenAI API key from environment variable
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_api_key:
            app_logger.error("labelguard", "OPENROUTER_API_KEY environment variable not set")
            return []
        
        client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
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
        indexes_str = ""
        llm_cache_file_prefix = f"alrg_{words_for_prompt[:12]}_llm"
        cached_response = llm_cache.get(prompt, LLM_MODEL, llm_cache_file_prefix)
        if cached_response:
            indexes_str = cached_response
            app_logger.info("labelguard", f"Detecting allergens LLM prompt: {prompt} cashed response: {cached_response}")

        else:
      
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            
            indexes_str = response.choices[0].message.content.strip()
            llm_cache.set(prompt, LLM_MODEL, indexes_str, llm_cache_file_prefix)
            
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
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_api_key:
            app_logger.error("labelguard", "OPENROUTER_API_KEY environment variable not set")
            return []
        
        client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
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
        indexes_str = ""
        llm_cache_file_prefix = f"alrg_{words_for_prompt[:12]}_llm"
        cached_response = llm_cache.get(prompt, LLM_MODEL, llm_cache_file_prefix)
        if cached_response:
            indexes_str = cached_response
            app_logger.info("labelguard", f"Detecting allergens LLM prompt: {prompt} cashed response: {cached_response}")

        else:
      
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0
            )
            
            indexes_str = response.choices[0].message.content.strip()
            llm_cache.set(prompt, LLM_MODEL, indexes_str, llm_cache_file_prefix)
            
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
            "STORAGE_CONDITIONS": "Storage instructions including temperature, humidity, and preservation conditions",
            "INGRIDIENTS": "Sentences that list ingredients or composition",
            "CONTACT_INFO": "contact info, address, phone, email",
            "UNKNOWN": "Unclassified or ambiguous text"
    }
    
    try:
        # Get OpenAI API key from environment variable
        openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
        if not openrouter_api_key:
            app_logger.error("labelguard", "OPENROUTER_API_KEY environment variable not set")
            return
        
        client = OpenAI(
            api_key=openrouter_api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
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
        
        prompt = f"""You are given a list of multilanguage sentences from a product label. Classify each sentence into these categories: {', '.join(categories.keys())}.

Categories:
{categories_desc}

Instructions:
1. For each sentence, determine which category it belongs to.
2. Translate non-English text to understand the meaning before classifying.
3. Return a **valid JSON object only** with category names as keys and lists of sentence indices as values.
4. Do not include any extra text, explanations, or comments.
5. If a category has no sentences, return an empty list for that key.
6. The JSON must be strictly parseable by `json.loads()`.
7. DO NOT OUTPUT ANYTHING OTHER THAN VALID JSON.

Sentences:
{sentences_text}

Return format example: {{"PRODUCT_NAME": [0, 1], "ALLERGEN_PHRASE": [2], "INGRIDIENTS": [3], "CONTACT_INFO": [4, 5], , "UNKNOWN": []}}
"""

        # Check cache first
        llm_cache_file_prefix = f"cls_ingr_{sentences_text[:12]}"
        response = llm_cache.get(prompt, LLM_MODEL, llm_cache_file_prefix)
        if response:
            app_logger.info("labelguard", f"Using cached response for classification")
        else:
            # Call OpenAI API
            completion = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0
            )
            response = completion.choices[0].message.content.strip()
            
            # Cache the response
            llm_cache.set(prompt, LLM_MODEL, response, llm_cache_file_prefix)
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



def split_text_block_1(layout_block: LayoutTextBlock, ocr_result: OCRResult) -> List[LayoutTextBlock]:
    """Detect text block types using word-level analysis and LLM position-based categorization"""
    # Step 1: Join all words into single text
    joined_text = "".join([word.text for word in ocr_result.words])
    
    # Get OpenRouter API key from environment variable
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_api_key:
        app_logger.error("labelguard", "OPENROUTER_API_KEY environment variable not set")
        return TextBlockDetectionList(blocks=[])
    
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # Build categories description
    categories_desc = "\n".join([
        f"- {key}: {category['description']}" 
        for key, category in TEXT_BLOCK_CATEGORIES.items()
    ])

    template = Template(Path(ROOT_DIR / "assets" / SERVICE_NAME / "prompts" / "split-and-detect-category-of-text-block.jinja").read_text())
    user_prompt = template.render(
        text=joined_text,
        categories_desc=categories_desc
    )

    # Check cache first
    llm_cache_file_prefix = f"split_{joined_text[:12]}_llm"
    cached_result = llm_cache.get(user_prompt, LLM_MODEL, llm_cache_file_prefix)
    
    if cached_result:
        app_logger.debug("labelguard", f"Type detection LLM prompt cached response: {cached_result}")
        result = TextBlockDetectionList.model_validate_json(cached_result)
    else:
        app_logger.debug("labelguard", f"LLM prompt: {user_prompt}")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5000,
            temperature=0
        )
        
        app_logger.debug("labelguard", f"LLM response: {response.choices[0].message.content}")

        result = TextBlockDetectionList.model_validate_json(response.choices[0].message.content)
        llm_cache.set(user_prompt, LLM_MODEL, result.model_dump_json(), llm_cache_file_prefix)
        

    # Step 2: Process LLM response and map to words
    blocks : List[LayoutTextBlock] = []
    base_index = layout_block.index
    index = ""

    for i, block in enumerate(result.blocks):
        if len(result.blocks) == 1:
            index = base_index  # "10"
        else:
            index = f"{base_index}_{i+1}"  # "10_1", "10_2", etc.

        # how to removeleading  spaces and newlines from block.text
        block.text = block.text.strip()
        #find
        joined_text.replace('\n', ' ')
        start_pos = joined_text.find(block.text[:50])
        end_pos = start_pos + len(block.text)-1 if start_pos != -1 else len(joined_text)
        
        category_words = get_words_by_char_position(ocr_result.words, [(start_pos, end_pos)])
            
        # Step 4: Calculate union bbox for this category
        if category_words:
            # Flatten all bbox coordinates from all words, excluding (0,0,0,0) bboxes
            all_x1 = [bbox[0] for word in category_words for bbox in word.bbox if bbox != (0, 0, 0, 0)]
            all_y1 = [bbox[1] for word in category_words for bbox in word.bbox if bbox != (0, 0, 0, 0)]
            all_x2 = [bbox[2] for word in category_words for bbox in word.bbox if bbox != (0, 0, 0, 0)]
            all_y2 = [bbox[3] for word in category_words for bbox in word.bbox if bbox != (0, 0, 0, 0)]
            
            min_x = min(all_x1)
            min_y = min(all_y1)
            max_x = max(all_x2)
            max_y = max(all_y2)
        else: 
            # Fallback bbox if no words found
            min_x = min_y = max_x = max_y = 0
        

        shift_x = layout_block.bbox[0]
        shift_y = layout_block.bbox[1]

        
        blocks.append(LayoutTextBlock(
            bbox=(min_x + shift_x, min_y + shift_y, max_x + shift_x, max_y + shift_y),
            image_crop=layout_block.image_crop.crop((min_x, min_y, max_x, max_y)),
            score=layout_block.score,
            label=block.category,
            index=index
        ))

    # If no blocks were detected, return a default "other" block
    if not blocks:
        blocks.append(LayoutTextBlock(
            bbox=(0, 0, len(joined_text), 0),
            image_crop=layout_block.image_crop,
            score=layout_block.score,
            label="other",
            index=index
        ))

    return blocks


def split_text_block (layout_block: LayoutTextBlock, ocr_result: OCRResult) -> List[LayoutTextBlock]:
    # this function try new approach: text segmentation using indexed sentences 
    
    # Local classes for the new JSON structure with line ranges
    class TextBlockClassification(BaseModel):
        """Classification result for a text block with line range, category, and language"""
        lines: str  # Line range in format "start-end" (e.g., "0-3", "4-7")r
        category: str  # Category, must match one from the provided list
        language: str  # ISO 639-1 two-letter codes (e.g., "uk", "ro", "pl", "en", "ru", "de")
                       # If text is multilingual, use "multilingual"
                       # Do not use custom tags like "ro-md" or "ua-pl"

    class TextBlockClassificationResult(BaseModel):
        """Complete classification result containing multiple text blocks"""
        blocks: List[TextBlockClassification] = [] 
    
    sentences = split_words_into_sentences(ocr_result.words, delimiters=['\n'])
    # Create numbered sentences for the prompt
    numbered_sentences = [
        f"{sentence.index}: {sentence.text}" 
        for sentence in sorted(sentences, key=lambda s: s.index)
    ]
    sentences_text = "\n".join(numbered_sentences)
    
    # Create joined text for fallback bbox calculation
    joined_text = "".join([word.text for word in ocr_result.words])

    # Get OpenRouter API key from environment variable
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_api_key:
        app_logger.error("labelguard", "OPENROUTER_API_KEY environment variable not set")
        return TextBlockDetectionList(blocks=[])
    
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # Build categories description
    categories_desc = "\n".join([
        f"- {key}: {category['description']}" 
        for key, category in TEXT_BLOCK_CATEGORIES.items()
    ])

    template = Template(Path(ROOT_DIR / "assets" / "labelguard" / "split-and-detect-category-of-text-block--by-line-index.jinja").read_text())
    user_prompt = template.render(
        text=sentences_text,
        categories_desc=categories_desc
    )

    # Check cache first
    llm_cache_file_prefix = f"split_{sentences_text[:12]}"
    cached_result = llm_cache.get(user_prompt, LLM_MODEL, llm_cache_file_prefix)
    
    if cached_result:
        app_logger.debug("labelguard", f"Type detection LLM prompt cached response: {cached_result}")
        result = TextBlockClassificationResult.model_validate_json(cached_result)
    else:
        app_logger.debug("labelguard", f"LLM prompt: {user_prompt}")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5000,
            temperature=0
        )
        
        app_logger.debug("labelguard", f"LLM response: {response.choices[0].message.content}")

        result = TextBlockClassificationResult.model_validate_json(response.choices[0].message.content)
        llm_cache.set(user_prompt, LLM_MODEL, result.model_dump_json(), llm_cache_file_prefix)
        

    # Step 2: Process LLM response and map to words
    blocks = []
    base_index = layout_block.index

    for i, block in enumerate(result.blocks):
        if len(result.blocks) == 1:
            index = base_index  # "10"
        else:
            index = f"{base_index}_{i+1}"  # "10_1", "10_2", etc.

        # Parse line range (e.g., "0-3" -> [0, 1, 2, 3])
        start_line, end_line = map(int, block.lines.split('-'))
        line_numbers = list(range(start_line, end_line + 1))

        category_words = []
        
        for line_number in line_numbers:
            if line_number < len(sentences):
                category_words.extend(sentences[line_number].words)
            
        # Step 4: Calculate union bbox for this category
        if category_words:
            # Flatten all bbox coordinates from all words, excluding (0,0,0,0) bboxes
            all_x1 = [bbox[0] for word in category_words for bbox in word.bbox if bbox != (0, 0, 0, 0)]
            all_y1 = [bbox[1] for word in category_words for bbox in word.bbox if bbox != (0, 0, 0, 0)]
            all_x2 = [bbox[2] for word in category_words for bbox in word.bbox if bbox != (0, 0, 0, 0)]
            all_y2 = [bbox[3] for word in category_words for bbox in word.bbox if bbox != (0, 0, 0, 0)]
            
            min_x = min(all_x1)
            min_y = min(all_y1)
            max_x = max(all_x2)
            max_y = max(all_y2)
        else:
            # Fallback bbox if no words found
            min_x = min_y = max_x = max_y = 0
        

        shift_x = layout_block.bbox[0]
        shift_y = layout_block.bbox[1]

        
        blocks.append(LayoutTextBlock(
            bbox=(min_x + shift_x, min_y + shift_y, max_x + shift_x, max_y + shift_y),
            image_crop=layout_block.image_crop.crop((min_x, min_y, max_x, max_y)),
            score=layout_block.score,
            label=block.category,
            index=index
        ))

    # If no blocks were detected, return a default "other" block
    if not blocks:
        blocks.append(LayoutTextBlock(
            bbox=(0, 0, len(joined_text), 0),
            image_crop=layout_block.image_crop,
            score=layout_block.score,
            label="other",
            index=index
        ))

    return blocks

def classify_text_block (ocr_result: OCRResult) -> str:

    # Create joined text for fallback bbox calculation
    joined_text = "".join([word.text for word in ocr_result.words])
    
    # Get OpenRouter API key from environment variable
    openrouter_api_key = os.getenv('OPENROUTER_API_KEY')
    if not openrouter_api_key:
        app_logger.error("labelguard", "OPENROUTER_API_KEY environment variable not set")
        return "E"  # Return category code "E" for "other" when API key is not available
    
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    # Build categories description
    categories_desc = "\n".join([
        f"- {key}: {category['description']}" 
        for key, category in TEXT_BLOCK_CATEGORIES.items()
    ])

    template = Template(Path(ROOT_DIR / "assets" / SERVICE_NAME / "prompts" / "classify-text-block.jinja").read_text())
    user_prompt = template.render(
        text=joined_text,
        categories_desc=categories_desc
    )

    # Check cache first
    llm_cache_file_prefix = f"cls_{joined_text[:12]}"
    result = llm_cache.get(user_prompt, LLM_MODEL, llm_cache_file_prefix)
    if result:
        app_logger.debug("labelguard", f"Type detection LLM prompt cached response: {result}")
    else:
        app_logger.debug("labelguard", f"LLM prompt: {user_prompt}")
        response = client.chat.completions.create(
            model=LLM_MODEL,
            response_format={"type": "text"},
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=5000,
            temperature=0
        )
        
        app_logger.debug("labelguard", f"LLM response: {response.choices[0].message.content}")

        result = response.choices[0].message.content
        llm_cache.set(user_prompt, LLM_MODEL, result, llm_cache_file_prefix)
   
    return result.strip().upper()


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

def refine_text_symbols(text : str) -> str:
    def strip_diacritics(s):
        # 1) decompose
        s2 = unicodedata.normalize('NFD', s)
        # 2) remove all non‐spacing marks
        return ''.join(ch for ch in s2 if unicodedata.category(ch) != 'Mn')
        
    text = text.replace(' ', ' ')  # Non-breaking spaces
    text = text.replace('–', '-')  # En-dash to hyphen
    text = text.replace(' ± ', '±')
    text = text.replace(' ° С', '°C')
    text = text.replace('° С', '°C')
    text = text.replace(' ° С', '°C')
    text = text.replace('° С', '°C')
    text = text.replace('°С', '°C')
    text = text.replace(' ІV ', ' IV ')
    text = text.replace(' ІѴ ', ' IV ')
    text = text.replace(' ІѴ.', ' IV ')
    text = text.replace(' ІV.', ' IV ')
    text = text.replace('"', "'")
    
    # Fix OCR mistake: (18+3)°C should be (18±3)°C (with or without spaces)
    text = re.sub(r'\((\d+)\s*\+\s*(\d+)\)°C', r'(\1±\2)°C', text)
    text = re.sub(r'(\d+)\s*\+\s*(\d+)°C', r'\1±\2°C', text)
    # Fix OCR mistake: (18+5)'S should be (18±5)°S 
    text = re.sub(r'\((\d+)\s*\+\s*(\d+)\)\'S', r'(\1±\2)°S', text)
    
    
    

    text = strip_diacritics(text)

    return text

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
    text = refine_text_symbols(text)
     
    return text.strip()


def extract_languages_from_etalon_files(label_image_path) -> List[str]:
    """
    Extract language codes from etalon files and convert them to BCP-47 format for OCR.
    
    Args:
        label_image_path: Path to the label image file (Path or str)
        
    Returns:
        List of BCP-47 language codes for OCR processing
    """
    try:
        # Handle both Path and str
        if isinstance(label_image_path, str):
            label_image_path = Path(label_image_path)
        # Get etalon text blocks
        etalon_text_blocks = _get_etalon_text_blocks("", "", label_image_path)
        
        # Extract unique language codes from etalon
        # Etalon codes are in uppercase (e.g., "UK", "RU", "BA")
        etalon_languages = set()
        for block in etalon_text_blocks:
            if block.get('LANGUAGES') and block['LANGUAGES'].strip():
                lang_code = block['LANGUAGES'].strip().upper()
                etalon_languages.add(lang_code)
        
        # Load language mapping from JSON file (all codes in lowercase)
        etalon_to_bcp47_mapping = LANGUAGE_MAPPING_ETALON_TO_BCP47
        
        # Convert etalon languages (uppercase) to BCP-47 codes (lowercase)
        bcp47_languages = []
        for etalon_lang in etalon_languages:
            # Convert etalon code (uppercase) to lowercase for lookup in JSON mapping
            etalon_lang_lower = etalon_lang.lower()
            bcp47_code = etalon_to_bcp47_mapping.get(etalon_lang_lower)
            if bcp47_code and bcp47_code not in bcp47_languages:
                bcp47_languages.append(bcp47_code)
        
        # Fallback to default languages if no etalon languages found
        if not bcp47_languages:
            return ["sq", "sr", "hy", "ar", "az", "bg", "bs", "zh", "cs", "de", "et", "en", "es", "fr", "ka", "el", "hr", "hu", "he", "it", "ky", "kk", "lt", "lv", "sr-ME", "mk", "mn", "ms", "nl", "pl", "pt", "ro", "ru", "sl", "sk", "tg", "tk", "uk", "uz"]
            
        app_logger.info(SERVICE_NAME, f"Extracted languages from etalon: {etalon_languages} -> OCR hints: {bcp47_languages}")
        return bcp47_languages
        
    except Exception as e:
        app_logger.warning(SERVICE_NAME, f"Failed to extract languages from etalon: {e}. Using default languages.")
        # Fallback to hardcoded languages
        return ["sq", "sr", "hy", "ar", "az", "bg", "bs", "zh", "cs", "de", "et", "en", "es", "fr", "ka", "el", "hr", "hu", "he", "it", "ky", "kk", "lt", "lv", "sr-ME", "mk", "mn", "ms", "nl", "pl", "pt", "ro", "ru", "sl", "sk", "tg", "tk", "uk", "uz"]


def _process_etalon_image(etalon_path_str: str) -> List[dict]:
    """
    Process etalon image: run OCR and extract text blocks in etalon format.
    
    Args:
        etalon_path_str: Path to etalon image file
        
    Returns:
        List of etalon text blocks in format: [{"type_": str, "LANGUAGES": str, "text": str}, ...]
    """
    # Load etalon image
    etalon_path = ROOT_DIR / etalon_path_str.lstrip('/')
    if not etalon_path.exists():
        app_logger.error(SERVICE_NAME, f"Etalon image not found: {etalon_path}")
        return []
    
    etalon_image = Image.open(etalon_path)
    
    # Run layout detection on etalon image
    layout_detector = LayoutDetector()
    etalon_layout_blocks = layout_detector.extract_blocks(etalon_path)
    
    app_logger.info(SERVICE_NAME, f"Detected {len(etalon_layout_blocks)} blocks in etalon image")
    
    # Run OCR on each block
    ocr_processor = VisionOCRProcessor()
    etalon_text_blocks = []
    
    for layout_block in etalon_layout_blocks:
        cropped = layout_block.image_crop
        
        # Run OCR
        cache_dir = artifact_service.get_service_dir(SERVICE_NAME) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        etalon_stem = Path(etalon_path_str).stem
        json_vision_filename = str(cache_dir / f"{etalon_stem}_{layout_block.index}_vision.json")
        
        ocr_result = ocr_processor.process(
            cropped,
            input_filename=str(etalon_path),
            json_vision_filename=json_vision_filename,
            language_hints=[]  # No hints for etalon
        )
        
        # Classify block type
        block_type = classify_text_block(ocr_result)
        
        # Create etalon text block in expected format
        etalon_block = {
            "type_": block_type,
            "LANGUAGES": ocr_result.language.upper() if ocr_result.language else None,  # Uppercase for etalon format
            "text": clean_html_text(ocr_result.text)
        }
        etalon_text_blocks.append(etalon_block)
    
    app_logger.info(SERVICE_NAME, f"Processed etalon image: {len(etalon_text_blocks)} text blocks extracted")
    return etalon_text_blocks


def _get_etalon_text_blocks(kmat: str, version: str, etalon_path_str) -> List[dict]:
    """
    Legacy function: Read etalon data from JSON file.
    Used as fallback when etalon_path is not provided.
    """
    etalon_path = ROOT_DIR / etalon_path_str.lstrip('/')
    if not etalon_path.exists():
        app_logger.error(SERVICE_NAME, f"Etalon file not found: {etalon_path}")
        return []
    with open(etalon_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove control characters that cause JSON parsing issues
    content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
    
    etalon_bank = json.loads(content)

    # Convert empty string values in LANGUAGES field to None and strip whitespace
    for item in etalon_bank:
        if 'LANGUAGES' in item:
            lang = item['LANGUAGES']
            # Convert empty string to None, otherwise strip whitespace
            item['LANGUAGES'] = None if lang == '' else lang.strip()

    return etalon_bank

if __name__ == "__main__":
    # Test analyze() function
    # Example: Initial detection (empty blocks)
    request_json = {
        'image_path': 'artifacts/labelguard/MС_Lemon_180g_UA_FP_KKF_v230925B__Text.jpg',
        "etalon_path" : "artifacts/labelguard/MС_Lemon_180g_UA_FP_KKF_v230925B__Text_etalon.json",
        'blocks': [],  # Empty = initial detection
        'kmat': 'MС_Lemon_180g',
        'version': 'v230925B'
    }
    
    print("Running analyze() function...")
    result = analyze(request_json)
    
    print(f"\n✅ Analysis complete!")
    print(f"   Found {len(result.text_blocks)} text blocks")
    print(f"   Validation results: {len(result.rule_check_results)}")
    print(f"   JSON saved to: artifacts/labelguard/{result.original_filename}.json")
    print(f"   Image saved to: artifacts/labelguard/{result.original_filename}.jpg")
    print("\n   Now refresh your HTML page to see the results!")
    print(f"   URL: viewer.html?data_id={result.original_filename}")


    """
1 вирбник
2 страна походження
3 двтв виготовлення
4 енергетична цінність
5 аліргени (жирні, однакова кількість як в еталоні)
6 аліргенна фраза (жирна)
7 числа маюсть спіпадати в кожній мові(і кількість також)
    """
