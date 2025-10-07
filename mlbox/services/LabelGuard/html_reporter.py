"""
HTML Report Generator for LabelGuard text comparison results.

This module provides functionality to generate HTML reports comparing 
OCR text blocks with their etalon (template) counterparts, highlighting 
matches and differences.
"""

from dataclasses import dataclass
from typing import List, Optional
import html
from collections import defaultdict
import re

from mlbox.utils.lcs import Match


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
        escaped_text = html.escape(text)
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
            unmatched_text = html.escape(text[pos:start])
            result.append(f'<span style="background-color: #ffcccc;">{unmatched_text}</span>')
        
        # Add matched text (green background)
        matched_text = html.escape(text[start:start + length])
        result.append(f'<span style="background-color: #ccffcc;">{matched_text}</span>')
        pos = start + length
    
    # Add any remaining unmatched text (red background)
    if pos < len(text):
        remaining_text = html.escape(text[pos:])
        result.append(f'<span style="background-color: #ffcccc;">{remaining_text}</span>')
    
    return ''.join(result)


def count_alien_words(text: str, matches: List[Match]) -> int:
    """
    Count the number of "alien" words - words that don't match the template.
    
    Args:
        text: The text to analyze
        matches: List of matches to determine which words are matched
        
    Returns:
        Number of alien (unmatched) words
    """
    if not text:
        return 0
    
    # Extract all words from the text
    words = re.findall(r'\w+', text, flags=re.UNICODE)
    if not words:
        return 0
    
    if not matches:
        return len(words)
    
    # Create a set of character positions that are covered by matches
    covered_positions = set()
    for match in matches:
        for pos in range(match.start_b, match.start_b + match.len_b):
            covered_positions.add(pos)
    
    # Count words that are not fully covered by matches
    alien_count = 0
    for word_match in re.finditer(r'\w+', text, flags=re.UNICODE):
        word_start, word_end = word_match.span()
        # Check if any part of the word is not covered
        word_covered = all(pos in covered_positions for pos in range(word_start, word_end))
        if not word_covered:
            alien_count += 1
    
    return alien_count


def generate_comparison_table_html(text_blocks: List, results_title: str = "Text Comparison Report") -> str:
    """
    Generate HTML table comparing text blocks with their etalon counterparts.
    
    Args:
        text_blocks: List of TextBlock objects with etalon_text and lcs_results
        results_title: Title for the HTML report
        
    Returns:
        Complete HTML document as string
    """
    html_rows = []
    
    for text_block in text_blocks:
        if not text_block.etalon_text or not text_block.lcs_results:
            # Skip blocks without etalon text or comparison results
            continue
            
        # Get matches for this text block
        matches = text_block.lcs_results if isinstance(text_block.lcs_results, list) else []
        
        # Count alien words
        alien_word_count = count_alien_words(text_block.text, matches)
        
        # Create highlighted versions of both texts
        text_block_html = highlight_matches_html(text_block.text, matches, use_start_a=False)
        etalon_html = highlight_matches_html(text_block.etalon_text, matches, use_start_a=True)
        
        # First row: Text block (left) and Template text (right)
        first_row = f"""
        <tr>
            <td class="text-cell">
                <div class="cell-content">
                    <pre class="text-content">{text_block_html}</pre>
                </div>
            </td>
            <td class="template-cell">
                <div class="cell-content">
                    <pre class="text-content">{etalon_html}</pre>
                </div>
            </td>
        </tr>
        """
        
        # Second row: Merged cell with metadata
        language_str = ", ".join([f"({lang.upper()})" for lang in text_block.languages]) if text_block.languages else "(UNKNOWN)"
        allergen_count = len(text_block.allergens) if text_block.allergens else 0
        allergens_display = ', '.join([allergen.text for allergen in text_block.allergens]) if text_block.allergens else "None"
        
        metadata_content = f"""
        <div class="metadata-content">
            <span class="metadata-item"><strong>{language_str} {text_block.type}</strong></span>
            <span class="metadata-item"><strong>Allergens ({allergen_count}):</strong> {allergens_display}</span>
        </div>
        """
        
        second_row = f"""
        <tr>
            <td colspan="2" class="metadata-cell">
                {metadata_content}
            </td>
        </tr>
        """
        
        html_rows.append(first_row)
        html_rows.append(second_row)
        
        # Add separator row between text blocks for better visual separation
        separator_row = f"""
        <tr class="separator-row">
            <td colspan="2" style="height: 20px; background-color: #f5f5f5; border: none;"></td>
        </tr>
        """
        html_rows.append(separator_row)
    
    # CSS styles
    css_styles = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        
        .report-title {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 24px;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .text-cell, .template-cell {
            width: 50%;
            vertical-align: top;
            border: 2px solid #ccc;
            border-bottom: 1px solid #ccc;
            padding: 0;
        }
        
        .metadata-cell {
            border: 2px solid #ccc;
            border-top: 1px solid #ccc;
            border-bottom: 4px solid #666;
            padding: 15px;
            background-color: #f0f4f8;
            margin-bottom: 10px;
        }
        
        .cell-content {
            padding: 15px;
            height: 100%;
        }
        
        .text-content {
            white-space: pre-wrap;
            word-break: break-word;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.4;
            margin: 0;
            background-color: transparent;
            border: none;
        }
        
        .metadata-content {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        
        .metadata-item {
            font-size: 15px;
            color: #444;
            line-height: 1.4;
        }
        
        .metadata-item strong {
            color: #222;
        }
        
        .separator-row {
            border: none !important;
        }
        
        .separator-row td {
            border: none !important;
            padding: 0 !important;
        }
        
        /* Highlighting styles */
        span[style*="background-color: #ccffcc"] {
            background-color: #ccffcc !important;
            padding: 1px 2px;
            border-radius: 2px;
        }
        
        span[style*="background-color: #ffcccc"] {
            background-color: #ffcccc !important;
            padding: 1px 2px;
            border-radius: 2px;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .comparison-table {
                font-size: 12px;
            }
            
            .text-content {
                font-size: 12px;
            }
            
            .metadata-content {
                gap: 6px;
            }
            
            .metadata-item {
                font-size: 13px;
            }
        }
    </style>
    """
    
    # Complete HTML document
    html_document = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{html.escape(results_title)}</title>
        {css_styles}
    </head>
    <body>
        <h1 class="report-title">{html.escape(results_title)}</h1>
        <table class="comparison-table">
            {''.join(html_rows)}
        </table>
    </body>
    </html>
    """
    
    return html_document


def generate_html_report(processing_result, output_filename: Optional[str] = None) -> str:
    """
    Generate HTML report for a LabelProcessingResult object.
    
    Args:
        processing_result: LabelProcessingResult object containing text_blocks with comparison data
        output_filename: Optional filename for the report (used in title)
        
    Returns:
        Complete HTML document as string
    """
    # Create report title
    if output_filename:
        title = f"Label Analysis Report - {output_filename}"
    else:
        title = f"Label Analysis Report - {processing_result.kmat} v{processing_result.version}"
    
    # Generate HTML table
    html_content = generate_comparison_table_html(processing_result.text_blocks, title)
    
    return html_content
