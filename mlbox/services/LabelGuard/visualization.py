"""
Visualization module for LabelGuard validation results.

This module contains all functions for generating visual reports from rule check results.
Handles different output formats: interactive HTML viewers, comparison reports, overlay images.
"""

import json
import html as html_module
import re
import base64
from collections import defaultdict
from typing import List
from PIL import Image, ImageDraw

# Import datatypes at module level
from mlbox.services.LabelGuard.datatypes import (
    LabelProcessingResult,
    RuleCheckResult,
    RulesName,
    ValidationArtifacts
)
from mlbox.utils.lcs import Match, highlight_matches_by_words_html


# ============================================================================
# RULE NAME TRANSLATIONS
# ============================================================================

# Ukrainian rule name translations
RULE_NAMES_UK = {
    RulesName.ETALON_MATCHING: "Перевірка відповідності тексту",
    RulesName.ALLERGENS: "Алергени",
    RulesName.NUMBERS_IN_INGRIDIENTS: "Числа в інгредієнтах"
}


# ============================================================================
# TEXT HIGHLIGHTING FUNCTIONS
# ============================================================================
# Note: highlight_matches_by_words_html moved to mlbox.utils.lcs


# ============================================================================
# PLAIN TEXT FORMATTING FUNCTIONS
# ============================================================================

def format_validation_details_as_text(
    text_block,
    check_results: List[RuleCheckResult]
) -> str:
    """
    Format validation details as clean plain text without extra empty lines.
    
    Args:
        text_block: The text block being validated
        check_results: List of rule check results for this text block
        
    Returns:
        Plain text formatted validation details
    """
    lines = []
    
    # Process each check result FIRST
    for check_result in check_results:
        if check_result.rule_name == RulesName.ETALON_MATCHING:
            etalon = check_result.text_block.etalon_text if check_result.text_block else ''
            score = check_result.score
            passed = check_result.passed
            score_expression = check_result.score_expression or ''
            
            total_words = len(etalon.split()) if etalon else 0
            status_emoji = "✅" if passed else "❌"
            
            lines.append(f"{status_emoji} Перевірка відповідності тексту ({score:.0f}/100):")
            lines.append("")
            
            if total_words > 0:
                lines.append(f"score = {score:.0f} = {score_expression}")
                lines.append("(100 - відсутніх слів / слів всього * 100)")
                
                if check_result.affected_words and len(check_result.affected_words) > 0:
                    missing_words_text = ', '.join([word.text for word in check_result.affected_words])
                    lines.append(f"відсутні слова: {missing_words_text}")
            
            lines.append("")
            lines.append(f"Еталон:")
            lines.append(etalon)
            lines.append("=" * 22)
            
        elif check_result.rule_name == RulesName.ALLERGENS:
            allergens = check_result.metadata.get('allergen_names', [])
            expected = check_result.metadata.get('expected_count', 0)
            actual_count = check_result.metadata.get('actual_count', 0)
            passed = (actual_count == expected)
            
            status_emoji = "✅" if passed else "❌"
            
            lines.append(f"{status_emoji} Алергени")
            lines.append(', '.join(allergens) if allergens else 'немає')
            
            if not passed:
                lines.append(f"в наявності {actual_count} очікую {expected}")
                
        elif check_result.rule_name == RulesName.NUMBERS_IN_INGRIDIENTS:
            missing = check_result.metadata.get('missing_numbers', [])
            extra = check_result.metadata.get('extra_numbers', [])
            passed = (not missing and not extra)
            
            status_emoji = "✅" if passed else "❌"
            
            lines.append(f"{status_emoji} Числа в інгредієнтах")
            if missing:
                lines.append(f"Відсутні: {', '.join(missing)}")
            if extra:
                lines.append(f"Зайві: {', '.join(extra)}")
                
        elif check_result.rule_name == RulesName.NUMBERS_IN_INGRIDIENTS:
            # Import here to avoid circular dependency
            from mlbox.services.LabelGuard.datatypes import NumbersCheckResult
            
            if isinstance(check_result, NumbersCheckResult):
                lines.append("=" * 22)
                status_emoji = "✅" if check_result.passed else "❌"
                lines.append(f"{status_emoji} Числа:")
                
                for cat_result in check_result.category_results:
                    missing = set(cat_result.reference_numbers) - set(cat_result.actual_numbers)
                    extra = set(cat_result.actual_numbers) - set(cat_result.reference_numbers)
                    
                    # Format the category line
                    actual_str = ', '.join(cat_result.actual_numbers) if cat_result.actual_numbers else '(порожньо)'
                    
                    if missing or extra:
                        # Show errors in red
                        error_parts = []
                        if missing:
                            error_parts.append(f"відсутні: {', '.join(sorted(missing))}")
                        if extra:
                            error_parts.append(f"зайві: {', '.join(sorted(extra))}")
                        lines.append(f"{cat_result.category}: {actual_str} ({'; '.join(error_parts)})")
                    else:
                        lines.append(f"{cat_result.category}: {actual_str}")
    
    # Add separator and header
    lines.append("=" * 32)
    lines.append(f"[{text_block.type.upper()}]")
    
    # Add the extracted text
    lines.append("")
    lines.append(text_block.text)
    lines.append("")
    
    # Group sentences by category and display them
    sentences_by_category = defaultdict(list)
    if hasattr(text_block, 'sentences') and text_block.sentences:
        for sentence in text_block.sentences:
            sentences_by_category[sentence.category].append(sentence.text)
    
    # Display sentences grouped by category
    if sentences_by_category:
        for category, sentence_texts in sorted(sentences_by_category.items()):
            combined_text = '. '.join(sentence_texts)
            lines.append(f"{category}: {combined_text}")
    
    return '\n'.join(lines)


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def generate_block_details_html(text_block, check_results: List[RuleCheckResult], is_error_block: bool = True) -> str:
    """
    Generate detailed HTML for a text block by concatenating pre-generated rule HTML.
    
    HTML is now generated in labelguard.py when rules are executed,
    so this function just assembles the pieces.
    
    Args:
        text_block: The TextBlock to generate details for
        check_results: List of RuleCheckResult for this block (with html_details pre-generated)
        is_error_block: If True, details sections will be collapsed by default. If False, open by default.
        
    Returns:
        HTML string with formatted validation details
    """
    parts = []
    
    # Header with block index, type, and bounding box for debugging
    parts.append(f"<div style='background-color: #f0f0f0; padding: 8px; margin-bottom: 10px; border-radius: 4px; font-weight: bold;'>")
    parts.append(f"📋 Block #{text_block.index} | Type: {text_block.type} | Lang: {text_block.languages} | BBox: ({text_block.bbox[0]}, {text_block.bbox[1]}) - ({text_block.bbox[2]}, {text_block.bbox[3]})")
    parts.append(f"</div>")
    
    # Text preview
    text_preview = text_block.text[:50] + "..." if len(text_block.text) > 50 else text_block.text
    text_preview = text_preview.replace('\n', ' ')
    parts.append(f"<div style='margin-bottom:10px;'>\"{html_module.escape(text_preview)}\"</div>")
    
    # Add pre-generated HTML for each rule
    for check_result in check_results:
        if check_result.html_details:
            # Wrap each rule in a filterable container
            rule_type = check_result.rule_name.value
            parts.append(f'<div class="rule-error" data-rule-{rule_type}>{check_result.html_details}</div>')
    
    # Add full extracted text from text block
    # Always include full details section
    parts.append("<hr style='margin: 4px 0; border: 0; border-top: 1px solid #ddd;'>")
    # Details open by default for non-error blocks, collapsed for error blocks
    open_attr = " open" if not is_error_block else ""
    parts.append(f"<details style='margin-top:10px;'{open_attr}>")
    parts.append("<summary style='cursor: pointer; font-weight: bold;'>Розпізнаний текст:</summary>")
    parts.append(f"<div style='white-space: pre-wrap; font-family: monospace; background-color: #f5f5f5; padding: 10px; border-radius: 4px; margin-top: 5px;'>{html_module.escape(text_block.text)}</div>")
    parts.append("</details>")
    
    # Add sentences with their types for category A blocks (ingredients)
    # Check if block type is "A" (case-insensitive) and has sentences
    block_type_str = str(text_block.type) if text_block.type else ""
    block_type_match = block_type_str.upper() == "A" or block_type_str.lower() == "ingredients"
    has_sentences = hasattr(text_block, 'sentences') and text_block.sentences
    
    # Show sentences with their types for category A blocks (ingredients)
    if block_type_match and has_sentences:
        parts.append("<hr style='margin: 4px 0; border: 0; border-top: 1px solid #ddd;'>")
        # Details open by default for non-error blocks, collapsed for error blocks
        open_attr = " open" if not is_error_block else ""
        parts.append(f"<details style='margin-top:10px;'{open_attr}>")
        parts.append("<summary style='cursor: pointer; font-weight: bold;'>Речення з типами:</summary>")
        parts.append("<div style='margin-top: 5px;'>")
        
        # Group sentences by category
        sentences_by_category = defaultdict(list)
        sentences_without_category = []
        
        for sentence in text_block.sentences:
            if hasattr(sentence, 'category') and sentence.category:
                sentences_by_category[sentence.category].append(sentence)
            else:
                sentences_without_category.append(sentence)
        
        # Display sentences grouped by category - format as "CATEGORY: sentence1. sentence2. ..."
        if sentences_by_category:
            for category in sorted(sentences_by_category.keys()):
                sentences = sentences_by_category[category]
                # Join all sentences with the same category
                sentence_texts = []
                for sentence in sentences:
                    sentence_text = html_module.escape(sentence.text) if hasattr(sentence, 'text') else str(sentence)
                    sentence_texts.append(sentence_text)
                joined_text = ' '.join(sentence_texts)
                parts.append(f"<div style='margin-bottom: 8px;'>")
                parts.append(f"<span style='font-weight: bold; color: #0066cc;'>{category}:</span> ")
                parts.append(f"<span>{joined_text}</span>")
                parts.append(f"</div>")
        
        # Show sentences without category if any
        if sentences_without_category:
            # Join all sentences without category
            sentence_texts = []
            for sentence in sentences_without_category:
                sentence_text = html_module.escape(sentence.text) if hasattr(sentence, 'text') else str(sentence)
                sentence_texts.append(sentence_text)
            joined_text = ' '.join(sentence_texts)
            parts.append(f"<div style='margin-bottom: 8px;'>")
            parts.append(f"<span style='font-weight: bold; color: #0066cc;'>БЕЗ ТИПУ:</span> ")
            parts.append(f"<span>{joined_text}</span>")
            parts.append(f"</div>")
        
        if not sentences_by_category and not sentences_without_category:
            parts.append("<div style='color: #666; font-style: italic;'>Речень не знайдено</div>")
        
        parts.append("</div>")
        parts.append("</details>")
    
    return ''.join(parts)


def generate_comparison_error_report(result: LabelProcessingResult) -> str:
    """Generate HTML report for etalon matching errors without table"""
    
    # Filter only etalon matching errors
    comparison_errors = [error for error in result.rule_check_results if error.rule_name == RulesName.ETALON_MATCHING]
    
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
    <title>{html_module.escape(title)}</title>
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
    <h1 class="report-title">{html_module.escape(title)}</h1>
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
                {error.text_block.etalon_text if error.text_block and error.text_block.etalon_text else 'No details available'}
            </div>
            
            <div class="text-comparison">
                <div class="text-block">
                    <h4>Label Text (with errors highlighted)</h4>
                    <div class="text-content">{html_module.escape(error.text_block.text if error.text_block else '')}</div>
                </div>
                <div class="text-block">
                    <h4>Template Text</h4>
                    <div class="text-content">{html_module.escape(error.text_block.etalon_text if error.text_block and error.text_block.etalon_text else 'No template available')}</div>
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
    label_processing_result: LabelProcessingResult,
    error_overlay_image: Image.Image
) -> str:
    """Generate interactive HTML viewer for error visualization"""
    
    # Group errors by text block
    text_block_errors = {}
    for error in label_processing_result.rule_check_results:
        if error.text_block:
            block_id = id(error.text_block)
            if block_id not in text_block_errors:
                text_block_errors[block_id] = {
                    'text_block': error.text_block,
                    'errors': []
                }
            text_block_errors[block_id]['errors'].append(error)
    
    # Generate summary for whitespace click - FULL DETAILED HTML
    # Count critical errors (passed=False)
    critical_errors = [e for e in label_processing_result.rule_check_results if not e.passed]
    
    # STEP 1: Count errors by rule type
    from collections import Counter
    rule_counts = Counter([e.rule_name for e in critical_errors])
    
    if not critical_errors:
        summary_html = "✅ Немає критичних помилок!"
    else:
        # Group failed errors by text block
        failed_by_block = {}
        for error in critical_errors:
            if error.text_block:
                block_id = id(error.text_block)
                if block_id not in failed_by_block:
                    failed_by_block[block_id] = {
                        'text_block': error.text_block,
                        'failed_rules': []
                    }
                failed_by_block[block_id]['failed_rules'].append(error)
        
        # Generate FULL detailed HTML for each block with failures
        all_blocks_html = []
        
        for block_id, block_data in sorted(failed_by_block.items(), 
                                           key=lambda x: x[1]['text_block'].index):
            text_block = block_data['text_block']
            check_results = block_data['failed_rules']  # Only failed rules

            # Use helper function to generate HTML
            # These are all error blocks since they're in failed_by_block
            block_html = generate_block_details_html(text_block, check_results, is_error_block=True)

            # Don't add data attributes to block wrapper anymore (individual errors have them)
            block_html = f'<div class="error-block" style="margin-bottom: 20px;">{block_html}</div>'

            all_blocks_html.append(block_html)
        
        # Add error count at the top and separator line before details
        error_count = len(critical_errors)
        passed_count = len([e for e in label_processing_result.rule_check_results if e.passed])
        blocks_content = f'<hr style="margin: 10px 0; border: 0; border-top: 6px solid #000;">'.join(all_blocks_html)
        separator_hr = '<hr id="error-separator" style="margin: 10px 0; border: 0; border-top: 6px solid #000;">' if all_blocks_html else ''
        summary_html = f'<div style="font-weight: bold; font-size: 22px; margin-bottom: 5px; color: #d00;">❌ Критичних помилок - {error_count}</div><div style="font-weight: bold; font-size: 22px; margin-bottom: 15px;">✅ Всього перевірок - {passed_count}</div>{separator_hr}{blocks_content}'
    
    # Create highlights data for JavaScript (for all blocks with validation results)
    # Sort by block index to ensure consistent order
    def sort_key(item):
        """Sort function that handles mixed string indices like '0', '1', '0_1', '0_2'"""
        index = item[1]['text_block'].index
        # Convert to string first, then split by underscore and convert to integers for proper sorting
        index_str = str(index)
        parts = index_str.split('_')
        return [int(part) for part in parts]
    
    highlights = []
    for block_id, block_data in sorted(text_block_errors.items(), key=sort_key):
        text_block = block_data['text_block']
        check_results = block_data['errors']
        
        # Determine if this block is an error block (has any failed checks)
        has_error = any(not result.passed for result in check_results)
        
        # Add highlight for all blocks (both passed and failed)
        highlights.append({
            'x1': text_block.bbox[0],
            'y1': text_block.bbox[1], 
            'x2': text_block.bbox[2],
            'y2': text_block.bbox[3],
            'type': 'validation_result',
            'block_index': text_block.index,  # Add block index for debugging
            'block_type': text_block.type,    # Add block type for debugging
            'message': generate_block_details_html(text_block, check_results, is_error_block=has_error)
        })
    
    # Convert PIL Image to base64
    from io import BytesIO
    
    try:
        # Convert image to bytes
        img_buffer = BytesIO()
        
        # Ensure image is in RGB mode for JPEG
        if error_overlay_image.mode in ('RGBA', 'LA', 'P'):
            rgb_image = Image.new('RGB', error_overlay_image.size, (255, 255, 255))
            if error_overlay_image.mode == 'P':
                error_overlay_image = error_overlay_image.convert('RGBA')
            if error_overlay_image.mode in ('RGBA', 'LA'):
                alpha_channel = error_overlay_image.split()[-1] if error_overlay_image.mode == 'RGBA' else error_overlay_image.split()[1]
                rgb_image.paste(error_overlay_image, mask=alpha_channel)
            error_overlay_image = rgb_image
        
        # Save to buffer as JPEG
        error_overlay_image.save(img_buffer, format='JPEG', quality=95)
        img_data = img_buffer.getvalue()
        img_base64 = base64.b64encode(img_data).decode('utf-8')
        img_data_uri = f'data:image/jpeg;base64,{img_base64}'
    except Exception as e:
        raise RuntimeError(f"Failed to encode image: {e}")
    
    # Generate HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  body {{ display: flex; height: 100vh; margin: 0; overflow: hidden; }}
  #imagePanel {{ flex: 1; position: relative; overflow: hidden; border-right: 1px solid #ccc; cursor: grab; padding: 25px; box-sizing: border-box; }}
  #messagePanel {{ flex: 1; padding: 10px; overflow-y: auto; }}
  canvas {{ display: block; }}
  #debugInfo {{
    position: absolute;
    top: 5px;
    left: 5px;
    background: rgba(0,0,0,0.7);
    color: #0f0;
    padding: 8px;
    font-family: monospace;
    font-size: 12px;
    border-radius: 4px;
    pointer-events: none;
    z-index: 1000;
    white-space: pre;
  }}
  #filterPanel {{
    font-size: 18px;
    font-family: inherit;
    margin-bottom: 22px;
  }}
  #filterPanel label {{
    display: inline-block;
    margin-right: 18px;
    margin-bottom: 6px;
    cursor: pointer;
  }}
  #filterPanel label:hover {{
    color: #054aad;
  }}
  .rule-count {{
    font-weight: 600;
  }}
  .rule-count.has-errors {{
    color: #d00;
  }}
  .rule-count.no-errors {{
    color: #888;
  }}
</style>
</head>
<body>

<div id="imagePanel">
  <div id="debugInfo">Hover over blocks to see info</div>
  <canvas id="labelCanvas"></canvas>
</div>
<div id="messagePanel">
  <div id="filterPanel">
    {''.join([f'<label><input type="checkbox" class="rule-filter" value="{rule.value}" checked style="transform: scale(1.2); vertical-align: middle; margin-right: 6px;"> {name} <span class="rule-count {"has-errors" if rule_counts.get(rule, 0) > 0 else "no-errors"}">({rule_counts.get(rule, 0)})</span></label>' for rule, name in RULE_NAMES_UK.items()])}
  </div>
  <div id="msgContent">Click on a red rectangle to see error details</div>
</div>

<script>
const canvas = document.getElementById('labelCanvas');
const ctx = canvas.getContext('2d');
const msgDiv = document.getElementById('msgContent');
const debugInfo = document.getElementById('debugInfo');

const img = new Image();
img.src = '{img_data_uri}';

// Error highlighting data
const highlights = {json.dumps(highlights, indent=2)};

// Summary for whitespace clicks
const summary = {json.dumps(summary_html)};

// Rule display names mapping
const RULE_DISPLAY_NAMES = {json.dumps({k.value: v for k, v in RULE_NAMES_UK.items()}, ensure_ascii=False)};

// Filter state - all rule types enabled by default
let enabledRuleTypes = new Set(["etalon", "allergens", "numbers"]);

let scale = 1;
let offsetX = 0, offsetY = 0;
let isDragging = false, startX = 0, startY = 0;

img.onload = () => {{
  canvas.width = img.width;
  canvas.height = img.height;
  
  // Calculate initial scale to fit entire image in viewport
  const panel = document.getElementById('imagePanel');
  const scaleX = panel.clientWidth / img.width;
  const scaleY = panel.clientHeight / img.height;
  scale = Math.min(scaleX, scaleY, 1); // Don't scale up, only down if needed
  
  // Center the image
  offsetX = (panel.clientWidth - img.width * scale) / 2;
  offsetY = (panel.clientHeight - img.height * scale) / 2;
  
  drawCanvas();
  // Show summary by default
  msgDiv.innerHTML = summary;
  // Initialize filtering after summary is loaded
  setTimeout(initializeFiltering, 100);
}};

function drawCanvas() {{
  ctx.clearRect(0,0,canvas.width,canvas.height);
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);
  ctx.drawImage(img, 0, 0);
  ctx.restore();
}}

function initializeFiltering() {{
  const filterPanel = document.getElementById('filterPanel');
  if (!filterPanel) return;

  // Add event listeners to checkboxes
  filterPanel.addEventListener('change', function(e) {{
    if (e.target.classList.contains('rule-filter')) {{
      const ruleValue = e.target.value;
      if (e.target.checked) {{
        enabledRuleTypes.add(ruleValue);
      }} else {{
        enabledRuleTypes.delete(ruleValue);
      }}
      updateFilteredSummary();
    }}
  }});

  // Initial filtering update - all enabled by default, so all blocks show
  setTimeout(updateFilteredSummary, 500);
}}

function updateFilteredSummary() {{
  // Target individual rule errors instead of entire blocks
  const ruleErrors = document.querySelectorAll('.rule-error');
  
  ruleErrors.forEach(ruleDiv => {{
    let shouldShow = false;
    for (const ruleType of enabledRuleTypes) {{
      if (ruleDiv.hasAttribute(`data-rule-${{ruleType}}`)) {{
        shouldShow = true;
        break;
      }}
    }}
    ruleDiv.style.display = shouldShow ? '' : 'none';
  }});

  // Hide entire blocks if they have no visible errors
  const errorBlocks = document.querySelectorAll('.error-block');
  let visibleBlockCount = 0;
  errorBlocks.forEach(block => {{
    const visibleErrors = block.querySelectorAll('.rule-error:not([style*="display: none"])');
    if (visibleErrors.length === 0) {{
      block.style.display = 'none';
    }} else {{
      block.style.display = '';
      visibleBlockCount++;
    }}
  }});

  // Hide separators between blocks that are adjacent to hidden blocks
  const allBlocks = Array.from(document.querySelectorAll('.error-block'));
  const allSeparators = Array.from(document.querySelectorAll('.error-block')).flatMap(block => {{
    const prev = block.previousElementSibling;
    const next = block.nextElementSibling;
    const separators = [];
    if (prev && prev.tagName === 'HR' && !prev.id) {{  // Don't include the main separator
      separators.push(prev);
    }}
    if (next && next.tagName === 'HR') {{
      separators.push(next);
    }}
    return separators;
  }});
  
  // Remove duplicates
  const uniqueSeparators = [...new Set(allSeparators)];
  
  uniqueSeparators.forEach(separator => {{
    const prevBlock = separator.previousElementSibling;
    const nextBlock = separator.nextElementSibling;
    const prevIsHidden = prevBlock && prevBlock.classList.contains('error-block') && prevBlock.style.display === 'none';
    const nextIsHidden = nextBlock && nextBlock.classList.contains('error-block') && nextBlock.style.display === 'none';
    
    // Hide separator if either adjacent block is hidden
    if (prevIsHidden || nextIsHidden) {{
      separator.style.display = 'none';
    }} else {{
      separator.style.display = '';
    }}
  }});

  // Hide main separator if no blocks are visible
  const mainSeparator = document.getElementById('error-separator');
  if (mainSeparator) {{
    mainSeparator.style.display = visibleBlockCount > 0 ? '' : 'none';
  }}

  // Checkbox counts are static (show total counts), don't update with filtering
}}

// Click to show message or summary
canvas.addEventListener('click', e => {{
  const x = (e.offsetX - offsetX) / scale;
  const y = (e.offsetY - offsetY) / scale;
  
  // Debug: log click coordinates
  console.log('Click at canvas coords:', {{x: e.offsetX, y: e.offsetY}});
  console.log('Click at image coords:', {{x: x.toFixed(2), y: y.toFixed(2)}});
  console.log('Transform:', {{scale: scale.toFixed(3), offsetX: offsetX.toFixed(2), offsetY: offsetY.toFixed(2)}});
  
  // Find all matching blocks (in case of overlaps)
  const matches = highlights.filter(h => x>=h.x1 && x<=h.x2 && y>=h.y1 && y<=h.y2);
  
  if(matches.length > 0) {{
    // Debug: log all matches
    console.log('Found', matches.length, 'matching blocks:');
    matches.forEach(m => {{
      console.log('  Block', m.block_index, ':', m.block_type, 'bbox:', [m.x1, m.y1, m.x2, m.y2]);
    }});
    
    // Use the first match (or last if you want topmost)
    const hit = matches[0];
    console.log('Showing details for Block', hit.block_index);
    
    // Clicked on bounding box - show detailed view
    msgDiv.innerHTML = `<div style="margin-bottom:10px;"><span>${{hit.message}}</span></div>`;
  }} else {{
    console.log('No block hit - showing summary');
    // Clicked on whitespace - show summary
    msgDiv.innerHTML = summary;
  }}
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
  }} else {{
    // Update debug info with current hover position
    const x = (e.offsetX - offsetX) / scale;
    const y = (e.offsetY - offsetY) / scale;
    const matches = highlights.filter(h => x>=h.x1 && x<=h.x2 && y>=h.y1 && y<=h.y2);
    
    if(matches.length > 0) {{
      const blockInfos = matches.map(m => `Block #${{m.block_index}} [${{m.block_type}}]`).join(', ');
      debugInfo.textContent = `Hovering: ${{blockInfos}}\\nMouse: (${{x.toFixed(0)}}, ${{y.toFixed(0)}})\\nBlocks at cursor: ${{matches.length}}`;
      debugInfo.style.display = 'block';
    }} else {{
      debugInfo.textContent = `Mouse: (${{x.toFixed(0)}}, ${{y.toFixed(0)}})\\nNo block here`;
      debugInfo.style.display = 'block';
    }}
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
    rule_check_results: List[RuleCheckResult]
) -> Image.Image:
    """
    Creates overlay image with visual markers from rule check results.
    Returns: Original image with all visual markers drawn on top
    """
    # Import logger inside function to avoid circular imports
    from mlbox.services.LabelGuard.labelguard import SERVICE_NAME, app_logger
    
    # Create a copy of the original image to draw on
    overlay_image = original_image.copy()
    # Convert to RGBA if not already
    if overlay_image.mode != 'RGBA':
        overlay_image = overlay_image.convert('RGBA')
    draw = ImageDraw.Draw(overlay_image)
    
    # Load font for text block labels
    try:
        from PIL import ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
    except ImportError:
        font = None
    
    # Track which text blocks have been labeled
    labeled_blocks = set()
    
    # Process all visual markers (only highlights, outlines are drawn separately)
    for check_result in rule_check_results:
        for marker in check_result.visual_markers:
            if marker.type == "highlight":
                # Draw semi-transparent highlight
                x1, y1, x2, y2 = marker.bbox
                
                # Extract the region
                try:
                    word_region = overlay_image.crop((x1, y1, x2, y2))
                    if word_region.mode != 'RGB':
                        word_region = word_region.convert('RGB')
                    
                    # Create colored overlay
                    color_overlay = Image.new('RGB', word_region.size, marker.color)
                    # Blend with specified opacity
                    blended = Image.blend(word_region, color_overlay, marker.opacity or 0.3)
                    
                    # Paste back the blended region
                    overlay_image.paste(blended, (x1, y1))
                except Exception as e:
                    # If crop fails (bbox out of bounds), skip this marker
                    app_logger.warning(SERVICE_NAME, f"Failed to apply highlight marker: {e}")
    
    # Draw text block outlines based on pass/fail status
    from collections import defaultdict
    block_results = defaultdict(list)
    
    for check_result in rule_check_results:
        if check_result.text_block:
            block_id = id(check_result.text_block)
            block_results[block_id].append(check_result)
    
    # Draw outlines: red for any failure, green for all passed
    for block_id, results in block_results.items():
        text_block = results[0].text_block
        any_failed = any(not r.passed for r in results)
        
        # Determine outline color
        if any_failed:
            outline_color = (255, 0, 0)  # Red for any failure
        else:
            outline_color = (0, 200, 0)  # Green for all passed
        
        # Draw outline
        draw.rectangle(
            text_block.bbox,
            outline=outline_color,
            width=3
        )
        
        # Label the text block with its index
        labeled_blocks.add(block_id)
        x1, y1, x2, y2 = text_block.bbox
        text = f"{text_block.index}"
        
        if font:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            text_width = len(text) * 10
            text_height = 16
        
        # Position at bottom-right corner (with small padding from edges)
        text_x = x2 - text_width - 5
        text_y = y2 - text_height - 5
        
        # Draw text without background (transparent)
        if font:
            draw.text((text_x, text_y), text, fill=outline_color, font=font)
        else:
            draw.text((text_x, text_y), text, fill=outline_color)
    
    return overlay_image


def generate_validation_report(
    label_processing_result: LabelProcessingResult,
    label_image: Image.Image,
    output_format: str = "interactive_viewer"
) -> ValidationArtifacts:
    """
    Main orchestration function for generating validation reports.
    
    Args:
        result: LabelProcessingResult with rule_check_results populated
        label_image: Original label image
        output_format: Which visualization to generate ("interactive_viewer", "comparison_report", "both")
    
    Returns:
        ValidationArtifacts with html_report, html_filename, and images list with filenames
    """
    # Only generate if there are rule check results
    if not label_processing_result.rule_check_results:
        return ValidationArtifacts(html_report="", html_filename="", images=[])
    
    # Step 1: Generate filenames based on original_filename
    base_name = label_processing_result.original_filename or "label"
    image_filename = f"{base_name}_errors_overlay.jpg"
    
    # Step 2: Generate overlay image
    error_overlay_image = generate_error_overlay_image(
        label_image, 
        label_processing_result, 
        label_processing_result.rule_check_results
    )
    
    # Convert RGBA to RGB for JPEG compatibility
    if error_overlay_image.mode == 'RGBA':
        background = Image.new('RGB', error_overlay_image.size, (255, 255, 255))
        background.paste(error_overlay_image, mask=error_overlay_image.split()[-1])
        error_overlay_image = background
    
    # Step 3: Generate HTML viewer based on format and determine filename
    if output_format == "interactive_viewer":
        html_content = generate_interactive_error_viewer(label_processing_result, error_overlay_image)
        html_filename = f"{base_name}_interactive_viewer.html"
    elif output_format == "comparison_report":
        html_content = generate_comparison_error_report(label_processing_result)
        html_filename = f"{base_name}_comparison_report.html"
    elif output_format == "both":
        # For "both", return interactive viewer as main report
        html_content = generate_interactive_error_viewer(label_processing_result, error_overlay_image)
        html_filename = f"{base_name}_interactive_viewer.html"
        # Could add comparison as second HTML in future
    else:
        raise ValueError(f"Unknown output_format: {output_format}. Use 'interactive_viewer', 'comparison_report', or 'both'")
    
    # Step 4: Return artifacts with filenames
    return ValidationArtifacts(
        html_report=html_content if html_content else "",
        html_filename=html_filename,
        images=[(image_filename, error_overlay_image)]
    )