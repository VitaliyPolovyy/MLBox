# HTML Report Generation for LabelGuard

This document describes the HTML report generation functionality that creates visual comparison reports between OCR text blocks and their etalon (template) counterparts.

## Overview

The HTML report generator creates a table-based comparison view where:
- **Left column**: Text from the OCR text block (formatted with green for matches, red for missing text)
- **Right column**: Template/etalon text (formatted with green for matches, red for text missing from the OCR block)
- **Merged row**: Metadata including text block type, language, and alien word count

## Key Features

### Visual Highlighting
- **Green background**: Text segments that match between OCR and template
- **Red background**: Text segments that don't match (missing or extra text)

### Metadata Display
- **Text block type**: Classification (e.g., "ingredients", "other")
- **Language**: Detected language(s) of the text block
- **Alien words**: Number of words in the OCR text that don't match the template

### Responsive Design
- Mobile-friendly layout
- Professional styling with shadows and spacing
- Monospace font for text content to preserve formatting

## Architecture

### Core Components

#### `html_reporter.py`
Main module containing HTML generation functions:

- `highlight_matches_html()`: Highlights matched/unmatched text segments
- `count_alien_words()`: Counts words not covered by matches
- `generate_comparison_table_html()`: Creates the comparison table
- `generate_html_report()`: Main entry point for report generation

#### Integration Points

The HTML reporter integrates with the main LabelGuard pipeline:

1. **Text Comparison**: Uses LCS results from `mlbox.utils.lcs`
2. **Text Blocks**: Works with `TextBlock` objects containing OCR results
3. **Artifact Storage**: Saves reports using the artifact service

## Usage

### Basic Usage

```python
from mlbox.services.LabelGuard.html_reporter import generate_html_report

# Assuming you have a ProcessingResult with text_blocks containing:
# - text: OCR extracted text
# - etalon_text: Template text for comparison
# - lcs_results: List of Match objects from LCS comparison

html_content = generate_html_report(processing_result)

# Save to file
with open("report.html", "w", encoding="utf-8") as f:
    f.write(html_content)
```

### Integration in Pipeline

The HTML report generation is automatically integrated into the main processing pipeline:

```python
# In labelguard.py process_labels() function:
result.html_report = generate_html_report(result)

# Save HTML report as artifact
artifact_service.save_artifact(
    service=SERVICE_NAME,
    file_name=f"{result.original_filename}_report.html",
    data=result.html_report
)
```

## Data Flow

1. **OCR Processing**: Extract text from label images
2. **Template Matching**: Find etalon text based on type and language
3. **LCS Comparison**: Find common substrings between OCR and template text
4. **HTML Generation**: Create visual comparison report
5. **Artifact Storage**: Save HTML file for viewing

## Example Output Structure

```html
<!DOCTYPE html>
<html>
<head>
    <title>Label Analysis Report</title>
    <style>/* Responsive CSS styles */</style>
</head>
<body>
    <h1>Label Analysis Report - filename</h1>
    <table class="comparison-table">
        <tr>
            <td class="text-cell">
                <!-- OCR text with highlighting -->
                <pre>OCR text with <span style="background: green">matches</span> 
                and <span style="background: red">differences</span></pre>
            </td>
            <td class="template-cell">
                <!-- Template text with highlighting -->
                <pre>Template text with <span style="background: green">matches</span>
                and <span style="background: red">missing parts</span></pre>
            </td>
        </tr>
        <tr>
            <td colspan="2" class="metadata-cell">
                <div class="metadata-content">
                    <span><strong>Text block type:</strong> ingredients</span>
                    <span><strong>Language:</strong> ru</span>
                    <span><strong>Alien words:</strong> 5</span>
                </div>
            </td>
        </tr>
    </table>
</body>
</html>
```

## Configuration

### Styling Customization

The CSS styles can be customized by modifying the `css_styles` string in `generate_comparison_table_html()`. Key style classes:

- `.comparison-table`: Main table styling
- `.text-cell`, `.template-cell`: Individual text cell styling
- `.metadata-cell`: Metadata row styling
- `.text-content`: Text content formatting

### Highlighting Colors

Default colors:
- **Match highlighting**: `#ccffcc` (light green)
- **Difference highlighting**: `#ffcccc` (light red)

## Error Handling

The HTML generator handles various edge cases:

- **No matches found**: Entire text highlighted as different
- **Empty text blocks**: Graceful handling with appropriate messages
- **Missing etalon text**: Skips HTML generation for that block
- **Invalid match data**: Validates match positions before highlighting

## Performance Considerations

- **Memory usage**: Large texts are processed efficiently with streaming
- **HTML escaping**: All user text is properly escaped to prevent XSS
- **CSS optimization**: Inline styles for better rendering performance

## Future Enhancements

Potential improvements:
- **Interactive features**: Click to highlight specific matches
- **Export options**: PDF generation, print-friendly views
- **Batch reporting**: Multiple labels in single report
- **Statistics dashboard**: Aggregate comparison metrics
- **Custom templates**: User-defined report layouts
