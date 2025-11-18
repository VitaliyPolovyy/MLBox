## Workflow Overview

**Complete Flow:**
1. **ERP → Server**: ERP sends image + etalon JSON → `POST /labelguard/analyze` (empty blocks)
2. **Server → ERP**: Server processes, saves artifacts, returns URLs (`image_path`, `data_endpoint`)
3. **ERP → Browser**: ERP opens static `viewer.html` with `data_id` parameter
4. **Browser → Server**: Browser loads `viewer.html`, fetches image and JSON data from server
5. **User → Browser**: User edits bboxes in browser UI
6. **Browser → Server**: Browser sends corrections → `POST /labelguard/analyze` (with blocks)
7. **Server → Browser**: Server returns updated JSON `LabelProcessingResult`
8. **Repeat**: Steps 5-7 until user is satisfied

---

## Workflow

### Initial Detection Phase

**User Action:** External ERP system sends image + etalon JSON to `POST /labelguard/analyze` with empty blocks list

**Input:**
- Request format: `multipart/form-data` or JSON with base64 image
- Request body:
  ```json
  {
    "image": <file upload or base64 encoded image>,
    "kmat": "string" (required),
    "version": "string" (required),
    "etalon": [  // Etalon text blocks as JSON array (required)
      {
        "type_": "PRODUCT_NAME",
        "LANGUAGES": "UK",  // Language code (uppercase)
        "text": "МС Лимон 180г"  // HTML text content
      },
      {
        "type_": "INGRIDIENTS",
        "LANGUAGES": "UK",
        "text": "Склад: цукор, лимонна кислота..."
      }
      // ... more etalon blocks
    ],
    "blocks": []  // Empty list = normal detection
  }
  ```
- **Etalon format**: Same structure as `{image_stem}_etalon.json` files (list of dicts with `type_`, `LANGUAGES`, `text`)

**Server Processing:**
- Load image from upload (or decode base64)
- Extract `original_filename` from image (without extension)
- Load etalon from request (instead of from file)
- Save image to: `artifacts/labelguard/{original_filename}.jpg`
- Since `blocks` is empty, run normal layout detection (no masking)
- Runs full pipeline: layout detection → OCR → text block categorization → validation (using etalon from request)
- Calls `analyze_corrected_blocks(request_json: dict)` with empty blocks list and etalon data
- Save result JSON to: `artifacts/labelguard/{original_filename}.json`

**Output:**
- **Response format**: `application/json`
- **Response body**:
  ```json
  {
    "image_path": "/artifacts/labelguard/{original_filename}.jpg",
    "data_endpoint": "/artifacts/labelguard/{original_filename}.json",
    "original_filename": "{original_filename}"
  }
  ```
- **File storage**:
  - Image: `artifacts/labelguard/{original_filename}.jpg`
  - JSON: `artifacts/labelguard/{original_filename}.json`
- **Note**: Both files saved in root of `artifacts/labelguard/` directory (not in subdirectories)
- **Static file serving**: Server serves files from `artifacts/` directory via HTTP (handled in Ray Serve deployment)

**ERP → Browser Flow:**
- ERP receives JSON response with file paths
- ERP knows static viewer URL: `http://server:port/assets/labelguard/html/viewer.html`
- ERP opens: `viewer.html?data_id={original_filename}` (or passes filename via URL parameter)
- Browser loads static `viewer.html`
- JavaScript in browser:
  - Reads `data_id` from URL parameters
  - Loads image: `img.src = '/artifacts/labelguard/' + data_id + '.jpg'`
  - Fetches data: `fetch('/artifacts/labelguard/' + data_id + '.json')` → gets JSON with `image_path` and `labelProcessingResult`
  - Renders UI with fetched data

---

### User Correction Phase

**User Actions in Browser:**
- Browser opens static `viewer.html` with `data_id` parameter in URL
- JavaScript reads `data_id` from URL: `new URLSearchParams(window.location.search).get('data_id')`
- JavaScript loads image: `img.src = '/artifacts/labelguard/' + data_id + '.jpg'`
- JavaScript fetches data: `fetch('/artifacts/labelguard/' + data_id + '.json')` → receives JSON with `image_path` and `labelProcessingResult`
- JavaScript renders: overlayed image with bboxes, error list, block details using `labelProcessingResult`
- Looks at result of analyzing
- Adjusts bounding boxes (resize, move, create new, delete) if needed
- Creates new blocks: Client assigns sequential ID (last ID + 1)
- Changes block categories (correct misclassifications) if needed
- Text is read-only (cannot edit - preserves word structure with bboxes and bold flags)
- Client-side JavaScript tracks `modified: bool` flag for each edited/new block
- **Modified flag logic**: Any user action sets `modified: true`:
  - User moves/resizes bbox → `modified: true`
  - User changes category → `modified: true`
  - User creates new block → `modified: true`
  - Once `modified: true`, it stays `true` (even if user reverts to original position/category)
  - Initial blocks from detection → `modified: false`
  - Deleted blocks: Simply omitted (no flag needed)
- **Important**: After edits, full JSON structure becomes stale (sentences/words tied to old bbox)
- JavaScript extracts lightweight blocks (id, bbox, category, modified) and `image_path` for sending to server

---

### Analysis Phase (Unified Endpoint - Initial Detection or Re-analysis with Corrections)

**User Action:** 
- **Initial**: ERP sends request to `POST /labelguard/analyze` (empty blocks)
- **Subsequent**: User clicks "Analyze" button in browser → JavaScript sends request to `POST /labelguard/analyze` (with blocks)

**Input to `POST /labelguard/analyze`:**
- JSON request body (dict):
  ```json
  {
    "image_path": "/artifacts/labelguard/{original_filename}.jpg",
    "kmat": "string" (optional, can reuse from first call),
    "version": "string" (optional, can reuse from first call),
    "etalon": [  // Optional - if not provided, reuse from first call (stored in response)
      {
        "type_": "PRODUCT_NAME",
        "LANGUAGES": "UK",
        "text": "..."
      }
      // ... etalon blocks
    ],
    "blocks": [  // Empty list = normal detection, non-empty = use provided bboxes
      {
        "id": "string",  // Block identifier (uses TextBlock.index, e.g., "0", "1", "10", "10_1")
        "bbox": [x1, y1, x2, y2],  // Bounding box coordinates (x1, y1, x2, y2 format)
        "category": "string",  // Block category (empty = auto-detect)
        "text": "string",  // Optional text (empty = run OCR)
        "modified": boolean  // Modification flag
      }
    ]
  }
  ```
- **Note**: For subsequent calls (after initial detection), `etalon` is optional - server can reuse etalon from first call if not provided

**Server Processing:**
- Load image from path: `ROOT_DIR / image_path` (resolve relative path from project root)
- Load etalon from request (if provided) or reuse from first call (if stored/cached)
- Calls `analyze_corrected_blocks(request_json: dict)` with JSON dict:
  
  **Step 1: Bbox Collection (Layout Detection)**
  - If `blocks` is empty or not provided:
    - Run normal layout detection (no masking)
    - Use all detected bboxes
  - If `blocks` is provided and not empty:
    - Convert image to OpenCV format (BGR)
    - For each provided bbox: paint white rectangle using `cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)` to mask out provided areas
    - Run layout detection on masked image
    - Filter detected bboxes: if overlap with any provided bbox > 10%, skip it (provided bboxes have priority)
    - Merge: provided bboxes + filtered detected bboxes
    - **Overlap calculation**: Uses same logic as `layout_detector.py` (intersection area / new box area)
    - **Overlap threshold**: 10% (if detected bbox overlaps >10% with any provided bbox, skip it)
    - **Priority**: Provided bboxes always win (if overlap detected, provided is kept, detected is skipped)
  
  **Step 2: Full Pipeline (Always Runs)**
  1. For each bbox (provided + detected): Run OCR on bbox → extract text (or use provided text if exists)
  2. Parse text → create sentences → create words with bboxes and bold flags
  3. Build complete `TextBlock` structure (enrichment), preserving `modified` flag from lightweight blocks
  4. Run category detection if category is empty/None (always detect if empty, regardless of whether block is new or user cleared)
  5. Reconstruct `LabelProcessingResult` from enriched blocks
  6. Run validation using `validate_label()`
  7. Generate updated validation results

**Output:**
- **First call** (from ERP, empty blocks): HTML document with embedded JSON (same as Initial Detection Phase)
- **Subsequent calls** (from browser, with blocks): JSON `LabelProcessingResult` only:
  - `text_blocks: List[TextBlock]` - enriched blocks with full structure (bbox, text, category, sentences, words, modified flag)
  - `kmat`, `version`, `original_filename` - metadata
  - `rule_check_results: List[RuleCheckResult]` - updated validation results

**Client Processing (Browser):**
- **First call**: ERP receives file paths, opens static `viewer.html?data_id={filename}`; Browser loads image and fetches JSON from static file paths
- **Subsequent calls**: JavaScript receives fresh JSON `LabelProcessingResult` (server re-enriched with new OCR data)
- JavaScript uses same rendering logic as initial call
- Updates: overlayed image with bboxes, error list, block details
- Updates results panel dynamically (no page reload)
- Allows filtering: error list view ↔ detailed block view
- Full JSON structure is now fresh (matches current bbox positions)

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
  "image_path": "string",
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
- Extracts simplified data from `LabelProcessingResult.text_blocks` (each `TextBlock` contains `modified` flag)
- Flattens nested structure (TextBlock → simplified block with id, bbox, text, category, modified)

---

## Architecture Decisions

### Data Class Structure
- **LabelInput**: Raw input (image + metadata: kmat, version, label_image, label_image_path)
- **LabelProcessingResult**: Contains text_blocks, metadata (kmat, version, original_filename), and rule_check_results
- **Progression**: LabelInput → LabelProcessingResult (via process_labels)
- **Connection**: `RuleCheckResult.text_block` references `TextBlock` from LabelProcessingResult
- **TextBlock**: Includes `modified: bool = False` field to track user modifications
- **No separate request classes**: `/Analyze` endpoint accepts plain JSON dict (no BlockInput/AnalyzeRequest classes)

### Unified Endpoint Function
- `analyze_corrected_blocks(request_json: dict)` - Unified analysis (handles both initial detection and corrections)
  - **Input**: JSON dict with `image_path`, `blocks` (list of dicts, can be empty), optional `kmat`, `version`
  - **Internal helper**: `create_label_processing_result_from_json()` - Creates initial LabelProcessingResult from JSON
  - **Processing**:
    - **Bbox Collection**:
      - If `blocks` empty: Run normal layout detection (no masking)
      - If `blocks` provided: Mask provided bbox areas (white rectangles), run layout detection on masked image, filter overlaps >10%, merge provided + detected
    - **Full Pipeline** (always runs):
      - For each bbox: Runs OCR on bbox → extracts text (or uses provided text if exists)
      - Parses text → creates sentences → creates words with bboxes and bold flags
      - Builds complete TextBlock structure (enrichment), preserving `modified` flag
      - Runs category detection if category is empty/None (always detect if empty)
      - Runs validation using `validate_label()`
  - **Output**: LabelProcessingResult with enriched blocks and updated validation results

- `process_labels(labels: List[LabelInput])` - Legacy function (may still be used internally, but external API uses `/Analyze` only)
  - Can call `analyze_corrected_blocks()` internally with empty blocks list

### Image Handling
- **Storage structure**:
  ```
  artifacts/labelguard/
    ├── cache/                    # Cache files (LLM, Google Vision API)
    │   ├── {filename}_vision.json
    │   └── {filename}_llm.json
    ├── temp/                     # Debug files, temporary stuff
    ├── {original_filename}.jpg   # Input/Output image (root)
    └── {original_filename}.json  # Input/Output JSON (root)
  ```
- **File naming**: Uses `original_filename` (extracted from image, without extension)
- **Storage location**: 
  - Image: `artifacts/labelguard/{original_filename}.jpg`
  - JSON: `artifacts/labelguard/{original_filename}.json`
- **HTTP paths** (returned to client):
  - Image: `/artifacts/labelguard/{original_filename}.jpg` → serves file directly
  - JSON: `/artifacts/labelguard/{original_filename}.json` → serves file directly
  - **Static file serving**: Server handles requests to `/artifacts/` paths and serves files from disk
- **Image masking**: When blocks provided, mask provided bbox areas using `cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)` (white rectangles)
- **Overwrite behavior**: If same image processed again, files are overwritten (acceptable per requirements)
- **Error handling**: Return error if image not found
- **Cleanup**: Not needed for MVP, can be added later (TTL, scheduled job)

### Response Format
- **First call** (empty blocks): JSON with file paths:
  ```json
  {
    "image_path": "/artifacts/labelguard/{original_filename}.jpg",
    "data_endpoint": "/artifacts/labelguard/{original_filename}.json",
    "original_filename": "{original_filename}"
  }
  ```
  - ERP opens static `viewer.html?data_id={original_filename}`
  - Browser loads image and fetches JSON from static file paths
- **Subsequent calls** (with blocks): JSON `LabelProcessingResult` only (fresh data after re-enrichment)
- JavaScript renders everything from `LabelProcessingResult` JSON
- Same rendering logic for all calls
- **Key point**: Full JSON is for rendering only. After edits, client sends lightweight blocks (full structure is stale). Server re-enriches and sends back fresh full JSON.

### Endpoint Implementation
- **Main endpoint**: `POST /labelguard/analyze`
  - Handles both initial detection (empty blocks) and corrections (non-empty blocks)
  - Processes image, saves artifacts, returns file paths
- **Static file serving**: 
  - Server handles requests to `/artifacts/` paths
  - Serves files directly from `artifacts/` directory on disk
  - Implemented in Ray Serve deployment (handles `/artifacts/labelguard/{filename}.jpg` and `.json`)
  - Returns files with appropriate content-type headers
- **Path-based routing in Ray Serve**
- **Static HTML**: `viewer.html` is static file at `/assets/labelguard/html/viewer.html` (served as static file)

### Block ID Management
- Use existing `TextBlock.index` field as block identifier
- Server-assigned blocks: Keep original `index` from detection (e.g., `"0"`, `"1"`, `"10"`, `"10_1"`)
- User-created blocks: Client assigns sequential ID = last ID + 1 (e.g., if last is `"10_2"`, new is `"11"`)
- IDs persist across `/Analyze` calls (stored in `TextBlock.index`)
- Deleted blocks: Simply omit from the list sent to server
- No hierarchical structure - just use index as-is

### Data Saving
- **Directory**: `artifacts/labelguard/corrections/`
- **File naming**: `{original-image-file-name}.json` (uses `original_filename` from LabelProcessingResult)
- **Structure**: Flat directory (all files in one directory)
- **History**: Overwrite on each save (latest only, no versioning)
- **Content**: Save only Part 1 (Label data): blocks with bbox, text, category, modified flag
- **When**: Save on every `/Analyze` click (overwrites previous version)
- Modified flag tracked during transport, saved in JSON

### Modified Flag Logic
- **Any user action sets `modified: true`**:
  - User moves/resizes bbox → `modified: true`
  - User changes category → `modified: true`
  - User creates new block → `modified: true`
- **Once `modified: true`, it stays `true`** (even if user reverts to original position/category)
- **Initial blocks from detection** → `modified: false`
- **Deleted blocks**: Simply omitted from list (no flag needed)

### Error Handling
- **Critical errors** (fail fast, return HTTP 400/404):
  - Image file missing: Return HTTP 404 with error message
  - OCR failure: Return HTTP 400 with error message (fail entire request)
  - Invalid bbox coordinates: Return HTTP 400 with specific error details
  - Invalid request format: Return HTTP 400 with validation error
- **Non-critical errors** (log, continue, return results):
  - Category detection failure: Set category to "UNKNOWN", log error, continue processing
  - Saving failure: Log error, don't fail request (processing succeeded)

---

## Unsolved Questions

1. **Conditional Processing** ✅
   - If category is empty/None → always run detection (simple rule, no flags needed)
   - If category has value → use provided value
   - Applies to both new blocks and existing blocks with cleared category

2. **Response Format Details** ✅
   - Embed full JSON `LabelProcessingResult` in `<script>` tag in HTML
   - JavaScript uses full JSON for rendering (needs complete structure)
   - After edits, client sends lightweight blocks (full structure is stale)
   - Server re-enriches and sends back fresh full JSON

3. **Image Handling Details** ✅
   - Storage: `artifacts/labelguard/temp/{uuid}.jpg` (relative path from project root)
   - Path format: Relative path, server resolves using `ROOT_DIR / image_path`
   - Return `image_path` in embedded JSON wrapper (Option A)
   - Save image during first `/Analyze` call (empty blocks) with generated UUID filename
   - Load image during subsequent `/Analyze` calls using `image_path` from request
   - Image masking: When blocks provided, mask using `cv2.rectangle()` with white color (same approach as `layout_detector.py`)
   - Error handling: Return error if image not found
   - Cleanup: Not needed for MVP, can be added later (TTL, scheduled job)

4. **Unified Endpoint Design** ✅
   - Single `/Analyze` endpoint handles both initial detection and corrections
   - Empty blocks list → normal layout detection
   - Non-empty blocks list → mask provided areas, detect rest, filter overlaps >10%, merge
   - Always runs full pipeline after bbox collection
   - Overlap threshold: 10% (provided bboxes have priority)
   - Masking approach: White rectangles using `cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)`

## Open Implementation Questions

1. **HTML/JavaScript Implementation** (Technical details, to be decided during implementation)
   - How is the image loaded in HTML? (base64? URL? file path?)
   - What JavaScript libraries/frameworks for bbox manipulation? (Fabric.js? Konva.js? Custom canvas?)
   - How to handle image scaling/coordinate transformation?
   - UI framework: Vanilla JavaScript? Lightweight framework?

## Resolved Questions

1. **Client Request Format for `/Analyze`** ✅
   - JSON dict with `image_path`, `blocks` (list of dicts), optional `kmat`, `version`
   - Lightweight blocks: `{id, bbox, category, text?, modified}`
   - Client sends only what user can edit (bbox, category)
   - Text is optional (if provided, use it; if empty, server runs OCR)
   - No separate request classes - uses plain JSON dict

2. **Conversion Logic** ✅
   - Server enriches lightweight blocks: OCR → text → sentences → words
   - Server builds complete `TextBlock` structure from bbox
   - No reconstruction needed - server creates full structure

3. **Text Editing** ✅
   - Text is read-only (user cannot edit)
   - Preserves word structure (bboxes, bold flags)
   - If text is wrong, user works around it (or system handles via OCR re-run)

4. **Data Class Structure** ✅
   - **LabelInput**: Raw input (image + metadata: kmat, version, label_image, label_image_path)
   - **LabelProcessingResult**: Contains text_blocks, metadata (kmat, version, original_filename), and rule_check_results
   - **TextBlock**: Includes `modified: bool = False` field to track user modifications
   - Progression: LabelInput → LabelProcessingResult (via process_labels)
   - No separate Label class - metadata stored directly in LabelProcessingResult
   - No request classes - `/Analyze` accepts plain JSON dict

5. **Conditional Processing** ✅
   - Simple rule: If category is empty/None → always run detection
   - If category has value → use provided value
   - No flags needed - applies to both new blocks and existing blocks with cleared category

6. **Block ID Management** ✅
   - Use existing `TextBlock.index` field as block identifier
   - Server-assigned blocks: Keep original `index` from detection
   - User-created blocks: Client assigns sequential ID = last ID + 1
   - IDs persist across `/Analyze` calls (stored in `TextBlock.index`)
   - Deleted blocks: Simply omit from the list sent to server

7. **Serialization for Saving** ✅
   - `TextBlock` includes `modified: bool = False` field (implemented)
   - `modified` flag preserved through enrichment process in `analyze_corrected_blocks()` (implemented)
   - Function `label_processing_result_to_json()` to be implemented - extracts simplified data from `LabelProcessingResult.text_blocks`
   - Flattens nested structure: TextBlock → simplified block (id, bbox, text, category, modified)
   - Saved JSON uses `image_path` (not `image_id`)

8. **Data Saving Implementation** ✅
   - **Directory**: `artifacts/labelguard/corrections/`
   - **File naming**: `{original-image-file-name}.json` (uses `original_filename` from LabelProcessingResult)
   - **Structure**: Flat directory (all files in one directory)
   - **History**: Overwrite on each save (latest only, no versioning)
   - **When**: Save on every `/Analyze` click (overwrites previous version)

9. **Error Handling** ✅
   - **Critical errors** (fail fast, return HTTP 400/404):
     - Image file missing: Return HTTP 404 with error message
     - OCR failure: Return HTTP 400 with error message (fail entire request)
     - Invalid bbox coordinates: Return HTTP 400 with specific error details
     - Invalid request format: Return HTTP 400 with validation error
   - **Non-critical errors** (log, continue, return results):
     - Category detection failure: Set category to "UNKNOWN", log error, continue processing
     - Saving failure: Log error, don't fail request (processing succeeded)

10. **Modified Flag Logic** ✅
   - Any user action sets `modified: true`:
     - User moves/resizes bbox → `modified: true`
     - User changes category → `modified: true`
     - User creates new block → `modified: true`
   - Once `modified: true`, it stays `true` (even if user reverts to original position/category)
   - Initial blocks from detection → `modified: false`
   - Deleted blocks: Simply omitted from list (no flag needed)

---

## Implementation Status

### ✅ Completed

1. **Data Structure Modifications**
   - Added `modified: bool = False` field to `TextBlock` dataclass
   - Removed redundant `BlockInput` and `AnalyzeRequest` classes
   - Using plain JSON dict for `/Analyze` endpoint input

2. **Function Stub Implementation**
   - `analyze_corrected_blocks(request_json: dict)` function created
   - Internal helper: `create_label_processing_result_from_json()` - creates initial LabelProcessingResult from JSON
   - Function structure: JSON parsing → image loading → OCR enrichment (stubbed) → validation

3. **Spec Documentation**
   - Updated spec to reflect actual implementation
   - Documented JSON request format
   - Documented function signatures and structure

### 🚧 In Progress / TODO

1. **OCR Enrichment** (Step 3-6 in `analyze_corrected_blocks`)
   - Crop image by bbox
   - Run OCR (or use provided text)
   - Parse sentences and words
   - Detect category if empty
   - Update TextBlock with enriched data

2. **Image Handling**
   - Extract `original_filename` from image (without extension)
   - Save image to: `artifacts/labelguard/{original_filename}.jpg`
   - Save JSON to: `artifacts/labelguard/{original_filename}.json`
   - Return HTTP paths: `image_path` and `data_endpoint` (direct file paths)
   - Implement static file serving: Handle requests to `/artifacts/labelguard/{filename}.jpg` and `.json` in Ray Serve deployment

3. **Endpoints**
   - Implement main endpoint: `POST /labelguard/analyze` (Ray Serve)
   - Implement static file serving: Handle `/artifacts/labelguard/{filename}.jpg` and `.json` requests in deployment
   - Handle empty blocks (initial detection) and non-empty blocks (corrections)
   - Implement image masking logic (white rectangles for provided bboxes)
   - Implement overlap filtering (10% threshold, provided bboxes priority)
   - Update `viewer.html` JavaScript to load image and fetch JSON from static file paths (read `data_id` from URL)

4. **Data Saving**
   - Implement `label_processing_result_to_json()` serialization function
   - Save corrections to `artifacts/labelguard/corrections/`

5. **JavaScript/HTML**
   - HTML skeleton
   - JavaScript rendering logic
   - Bbox editing UI
   - See detailed specification: [labelguard-html-js-spec.md](labelguard-html-js-spec.md)