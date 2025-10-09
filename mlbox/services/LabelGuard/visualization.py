"""
Visualization module for LabelGuard validation results.

This module contains all functions for generating visual reports from rule check results.
Handles different output formats: interactive HTML viewers, comparison reports, overlay images.
"""

import json
import html as html_module
import re
from typing import List
from PIL import Image, ImageDraw

# Import datatypes at module level
from mlbox.services.LabelGuard.datatypes import (
    LabelProcessingResult,
    RuleCheckResult,
    RulesName,
    ValidationArtifacts
)
from mlbox.utils.lcs import Match


# ============================================================================
# TEXT HIGHLIGHTING FUNCTIONS
# ============================================================================

def highlight_matches_html(text: str, matches: List[Match], use_start_a: bool = False) -> str:
    """
    Highlight matched text segments in green and unmatched segments in red.
    
    Args:
        text: The text to highlight
        matches: List of Match objects containing position information
        use_start_a: If True, use start_a and len_a from matches, otherwise use start_b and len_b
        
    Returns:
        HTML string with highlighted text
    """
    if not matches:
        # No matches - entire text is unmatched (red)
        escaped_text = html_module.escape(text)
        return f'<span style="background-color: #ffcccc;">{escaped_text}</span>'
    
    # Sort matches by start position
    if use_start_a:
        sorted_matches = sorted(matches, key=lambda m: m.start_a)
    else:
        sorted_matches = sorted(matches, key=lambda m: m.start_b)
    
    result = []
    pos = 0
    
    for match in sorted_matches:
        if use_start_a:
            start = match.start_a
            length = match.len_a
        else:
            start = match.start_b
            length = match.len_b
            
        # Add unmatched text before this match (red background)
        if start > pos:
            unmatched_text = html_module.escape(text[pos:start])
            result.append(f'<span style="background-color: #ffcccc;">{unmatched_text}</span>')
        
        # Add matched text (green background)
        matched_text = html_module.escape(text[start:start + length])
        result.append(f'<span style="background-color: #ccffcc;">{matched_text}</span>')
        pos = start + length
    
    # Add any remaining unmatched text (red background)
    if pos < len(text):
        remaining_text = html_module.escape(text[pos:])
        result.append(f'<span style="background-color: #ffcccc;">{remaining_text}</span>')
    
    return ''.join(result)


def highlight_matches_by_words_html(text: str, matches: List[Match], use_start_a: bool = False) -> str:
    """
    Highlight matched text at WORD level (not character level).
    Entire words are either green (fully matched) or red (partially/fully unmatched).
    
    Args:
        text: The text to highlight
        matches: List of Match objects containing position information
        use_start_a: If True, use start_a and len_a from matches, otherwise use start_b and len_b
        
    Returns:
        HTML string with word-level highlighted text
    """
    if not matches:
        # No matches - entire text is unmatched (red)
        escaped_text = html_module.escape(text)
        return f'<span style="background-color: #ffcccc;">{escaped_text}</span>'
    
    result = ""
    start_pos = 0

    for match in matches:
        # Get the correct match
        #  position based on use_start_a parameter
        match_start = match.start_a if use_start_a else match.start_b
        match_len = match.len_a if use_start_a else match.len_b
        
        # Add unmatched text (red background)
        if start_pos < match_start:
            unmatched = html_module.escape(text[start_pos:match_start])
            result += f'<span style="background-color: #ffcccc;">{unmatched}</span>'
        
        # Add matched text (green background)
        matched = html_module.escape(text[match_start:match_start + match_len])
        result += f'<span style="background-color: #ccffcc;">{matched}</span>'
        start_pos = match_start + match_len

    # Add any remaining unmatched text (red background)
    if start_pos < len(text):
        remaining = html_module.escape(text[start_pos:])
        result += f'<span style="background-color: #ffcccc;">{remaining}</span>'
    
    return result


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================


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
    error_overlay_image_path: str
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
    
    # Create highlights data for JavaScript
    highlights = []
    for block_id, block_data in text_block_errors.items():
        text_block = block_data['text_block']
        check_results = block_data['errors']
        
        # Start with block info at the top
        combined_details = []
        combined_details.append(f"<div style='margin-bottom:15px;'><strong>[{text_block.type.upper()}]</strong></div>")
        combined_details.append(f"<div style='margin-bottom:20px; padding:10px; background-color:#f5f5f5; border-radius:4px;'>{html_module.escape(text_block.text)}</div>")
        combined_details.append("<hr>")
        
        # Generate formatted message from rule check results
        for check_result in check_results:
            # Add rule-specific details from metadata
            if check_result.rule_name == RulesName.ETALON_MATCHING:
                # Read from text_block (already linked in check_result)
                etalon = check_result.text_block.etalon_text if check_result.text_block else ''
                actual = check_result.text_block.text.replace("\n", " ") if check_result.text_block else ''
                lcs_matches = check_result.text_block.lcs_results if check_result.text_block else []
                
                # Get score from check_result (calculated in labelguard.py)
                score = check_result.score
                passed = check_result.passed
                score_expression = check_result.score_expression or ''
                
                # Calculate for display (can derive from existing data)
                total_words = len(etalon.split()) if etalon else 0
                missing_words = len(check_result.affected_words) if check_result.affected_words else 0
                
                status_emoji = "✅" if passed else "❌"
                
                # Highlight etalon text: green = exists in actual, red = missing from actual
                # use_start_a=False because etalon is parameter B in LCS matching
                etalon_highlighted = highlight_matches_by_words_html(etalon, lcs_matches, use_start_a=False) if etalon else ''
                
                combined_details.append(f"<div style='margin-top:15px;'><strong>{status_emoji} Перевірка відповідності тексту ({score:.0f}/100):</strong></div>")
                
                # Show score calculation formula
                if total_words > 0:
                    combined_details.append(f"<div style='margin-top:5px; font-size:0.9em; color:#666;'>score = 100 - відсутніх слів / слів всього * 100</div>")
                    combined_details.append(f"<div style='font-size:0.9em; color:#666;'>{score:.0f} = {score_expression}</div>")
                
                combined_details.append(f"<div style='margin-top:10px;'><strong>Еталон:</strong><br>{etalon_highlighted}</div>")
                
            elif check_result.rule_name == RulesName.ALLERGENS:
                allergens = check_result.metadata.get('allergen_names', [])
                expected = check_result.metadata.get('expected_count', 0)
                actual_count = check_result.metadata.get('actual_count', 0)
                passed = (actual_count == expected)
                
                status_emoji = "✅" if passed else "❌"
                
                combined_details.append(f"<div style='margin-top:15px;'><strong>{status_emoji} Алергени</strong></div>")
                combined_details.append(f"<div style='margin-top:5px;'>{', '.join(allergens) if allergens else 'немає'}</div>")
                
                if not passed:
                    combined_details.append(f"<div style='margin-top:5px; color:#d00;'>в наявності {actual_count} очікую {expected}</div>")
                    
            elif check_result.rule_name == RulesName.NUMBERS_IN_INGRIDIENTS:
                missing = check_result.metadata.get('missing_numbers', [])
                extra = check_result.metadata.get('extra_numbers', [])
                passed = (not missing and not extra)
                
                status_emoji = "✅" if passed else "❌"
                
                combined_details.append(f"<div style='margin-top:15px;'><strong>{status_emoji} Числа в інгредієнтах</strong></div>")
                if missing:
                    combined_details.append(f"<div style='margin-top:5px;'><strong>Відсутні:</strong> {', '.join(missing)}</div>")
                if extra:
                    combined_details.append(f"<div style='margin-top:5px;'><strong>Зайві:</strong> {', '.join(extra)}</div>")
        
        highlights.append({
            'x1': text_block.bbox[0],
            'y1': text_block.bbox[1], 
            'x2': text_block.bbox[2],
            'y2': text_block.bbox[3],
            'type': 'validation_result',
            'message': '<br>'.join(combined_details)
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
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None
    except ImportError:
        font = None
    
    # Track which text blocks have been labeled
    labeled_blocks = set()
    
    # Process all visual markers from all rule check results
    for check_result in rule_check_results:
        for marker in check_result.visual_markers:
            if marker.type == "outline":
                # Draw outline/border
                draw.rectangle(
                    marker.bbox,
                    outline=marker.color,
                    width=marker.width or 2
                )
                
                # Label the text block with its index (only once per block)
                if check_result.text_block and id(check_result.text_block) not in labeled_blocks:
                    labeled_blocks.add(id(check_result.text_block))
                    x1, y1, _, _ = marker.bbox
                    text = f"{check_result.text_block.index}"
                    
                    if font:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        text_width = bbox[2] - bbox[0]
                        text_height = bbox[3] - bbox[1]
                    else:
                        text_width = len(text) * 6
                        text_height = 12
                    
                    text_x, text_y = x1 + 5, y1 + 5
                    draw.rectangle(
                        [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
                        fill=(255, 255, 255, 200),
                        outline=marker.color
                    )
                    
                    if font:
                        draw.text((text_x, text_y), text, fill=marker.color, font=font)
                    else:
                        draw.text((text_x, text_y), text, fill=marker.color)
                        
            elif marker.type == "highlight":
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
        html_content = generate_interactive_error_viewer(label_processing_result, image_filename)
        html_filename = f"{base_name}_interactive_viewer.html"
    elif output_format == "comparison_report":
        html_content = generate_comparison_error_report(label_processing_result)
        html_filename = f"{base_name}_comparison_report.html"
    elif output_format == "both":
        # For "both", return interactive viewer as main report
        html_content = generate_interactive_error_viewer(label_processing_result, image_filename)
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

