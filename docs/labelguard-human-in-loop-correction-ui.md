## Workflow

### Initial Detection Phase

**User Action:** External ERP system sends image to `POST /StartProcessing`

**Input:**
- `LabelInput`: `kmat`, `version`, `label_image`, `label_image_path`, optional `etalon_file`

**Server Processing:**
- Calls `process_labels(labels: List[LabelInput])`
- Runs full pipeline: layout detection → OCR → text block categorization → validation

**Output:**
- `LabelProcessingResult` containing:
  - `text_blocks: List[TextBlock]` - detected blocks with bbox, text, category
  - `rule_check_results: List[RuleCheckResult]` - validation results
  - `html_report: str` - self-contained HTML with correction UI and initial validation

---

### User Correction Phase

**User Actions in Browser:**
- Opens `LabelProcessingResult.html_report` in browser
- Looks at result of analyzing
- Adjusts bounding boxes (resize, move, create new, delete) if needed
- Edits extracted text (fix OCR errors) if needed
- Changes block categories (correct misclassifications) if needed
- Client-side JavaScript tracks `modified: bool` flag for each edited/new block

---

### Analysis Phase (Re-analysis with Corrections)

**User Action:** Clicks "Analyze" button

**Input to `POST /Analyze`:**
- `LabelProcessingResult` (or lighter class) with corrected blocks:
  - Modified bboxes, text, categories
  - `modified: bool` flag per block

**Server Processing:**
- Calls `process_labels()` with corrected data:
  1. **SKIP** layout detection
  2. **SKIP** OCR processing
  3. **SKIP** text block categorization
  4. Use corrected blocks directly
  5. Run validation using `validate_label()`
  6. Generate updated validation results

**Output:**
- `LabelProcessingResult` with:
  - Corrected `text_blocks`
  - Updated `rule_check_results`
  - Updated `html_report` (or just validation data for client-side update)

**Client Processing:**
- Receives `LabelProcessingResult`
- Regenerates parts of HTML: error list, block details, error overlay image
- Redraws image with updated bboxes and error visualization
- Updates results panel dynamically (no page reload)
- Allows filtering: error list view ↔ detailed block view

---

### Iteration Phase

- User reviews validation results
- Makes additional corrections
- Clicks "Analyze" again
- Process repeats until satisfied

---

## Data Structure for Saving Corrections

**Saved JSON Structure:**
```json
{
  "image_id": "string",
  "timestamp": "ISO 8601 datetime string",
  "kmat": "string",
  "version": "string",
  "original_image_path": "string",
  "blocks": [
    {
      "id": "string",
      "bbox": [x, y, w, h],
      "text": "string",
      "category": "string",
      "modified": boolean
    }
  ]
}
```

**JSON Schema:** Defined for validation and documentation

**Serialization:** Single function `label_processing_result_to_json()` converts `LabelProcessingResult` to this structure