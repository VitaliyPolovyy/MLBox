from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Literal
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
from mlbox.services.LabelGuard.datatypes import (
    LabelInput,
    Sentence,
    TextBlock,
    RulesName,
    VisualMarker,
    RuleCheckResult,
    LabelProcessingResult
)
from mlbox.settings import ROOT_DIR, LOG_LEVEL


CURRENT_DIR = Path(__file__).parent
SERVICE_NAME = "labelguard"
app_logger = get_logger(ROOT_DIR)
artifact_service = get_artifact_service(ROOT_DIR)
llm_cache = LLMCache(artifact_service.get_service_dir(SERVICE_NAME) / "llm_cache")

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

def rule_check_text_matches_etalon(label_processing_result: LabelProcessingResult) -> List[RuleCheckResult]:
    """Detect text that appears in OCR but not in etalon text"""
    rule_check_results = []
    app_logger.info(SERVICE_NAME, f"Starting error detection for {len(label_processing_result.text_blocks)} text blocks")
   
    for text_block in label_processing_result.text_blocks:
       
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
        position_ranges = get_unmatched_positions(text, lcs_results)
        
        error_words = get_words_by_char_position(words, position_ranges)
        # to do: i want to delete error words like: space, period, comma, etc.
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

        if error_words:
            # Create visual markers for error words
            visual_markers = []
            
            # Add red outline around the entire text block
            visual_markers.append(VisualMarker(
                type="outline",
                bbox=text_block.bbox,
                color=(255, 0, 0),
                width=3
            ))
            
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
            
            rule_check_results.append(RuleCheckResult(
                rule_name=RulesName.ETALON_MATCHING,
                scope="block",
                text_block=text_block,
                affected_words=error_words,
                visual_markers=visual_markers,
                passed=passed,
                score=score,
                threshold=threshold,
                score_expression=score_expression
            ))
    
    return rule_check_results

def rule_check_allergens(label_processing_result: LabelProcessingResult) -> List[RuleCheckResult]:
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
        passed = (current_count == etalon_count)
        score = (current_count / etalon_count * 100) if etalon_count > 0 else 100.0
        
        # Create visual markers for allergens
        visual_markers = []
        allergen_names = [allergen.text for allergen in text_block.allergens]
        
        # Add yellow outline around text block if allergen check fails
        if not passed:
            visual_markers.append(VisualMarker(
                type="outline",
                bbox=text_block.bbox,
                color=(255, 255, 0),
                width=3
            ))
        
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
        
        errors.append(RuleCheckResult(
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
        ))
    
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
        
        label_processing_results.append(label_processing_result)

    
    return label_processing_results

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