# LabelGuard HTML/JavaScript Features Specification

## File Structure
```
assets/labelguard/
  ├── html/
  │   └── viewer.html          # HTML skeleton (static structure)
  └── js/
      └── labelguard-ui.js     # JavaScript rendering logic
```

## Layout Description

**Two-panel layout:**
- **Left Panel** (`#imagePanel`): Image canvas with bboxes and overlays
- **Right Panel** (`#messagePanel`): Controls, filters, and details

```
┌─────────────────────┬─────────────────────┐
│                     │ [Analyze] [Overlay] │
│   Image Canvas      ├─────────────────────┤
│   with bboxes       │ [Filter Checkboxes] │
│   and highlights    ├─────────────────────┤
│                     │                     │
│                     │  Error List /       │
│                     │  Block Details      │
│                     │                     │
└─────────────────────┴─────────────────────┘
```

## Features by HTML Element

### **Image Panel** (`#imagePanel`)
- **Overlayed Image** - Display image with bboxes and error highlights
- **Modifying Bboxes** - User can drag, resize, create, delete bboxes
- **Zoom & Pan** - Ctrl+wheel zoom, mouse drag pan
- **Click Detection** - Click bbox → show details, click whitespace → show error list

### **Details Panel** (`#msgContent`)
- **Error List** - Summary of all validation errors (shown on whitespace click)
- **Block Details** - Detailed block information (shown on bbox click)

### **Filter Panel** (`#filterPanel`)
- **Rule Type Filtering** - Checkboxes to show/hide errors by rule type

### **Control Buttons** (`#controlPanel`)
- **Analyze Button** - Send corrections to server, receive fresh results
- **Generate Overlayed Image** - Download image with all overlays

## HTML Skeleton Structure

**File:** `assets/labelguard/html/viewer.html`

**Structure:**
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LabelGuard - Label Validation</title>
  <style>
    /* CSS styles */
  </style>
</head>
<body>
  <!-- Left Panel: Image with bboxes -->
  <div id="imagePanel">
    <canvas id="labelCanvas"></canvas>
    <div id="debugInfo"></div>
  </div>
  
  <!-- Right Panel: Controls and Details -->
  <div id="messagePanel">
    <!-- Control buttons -->
    <div id="controlPanel">
      <button id="analyzeBtn">Analyze</button>
      <button id="generateOverlayBtn">Generate Overlayed Image</button>
    </div>
    
    <!-- Filter panel for rule types -->
    <div id="filterPanel"></div>
    
    <!-- Error list / Block details -->
    <div id="msgContent"></div>
  </div>
  
  <!-- Embedded JSON data -->
  <script>
    const LABELGUARD_DATA = {
      "image_path": "...",
      "labelProcessingResult": {...}
    };
  </script>
  
  <!-- JavaScript logic -->
  <script src="assets/labelguard/js/labelguard-ui.js"></script>
</body>
</html>
```

## JavaScript Functions Required

**File:** `assets/labelguard/js/labelguard-ui.js`

### 1. **Initialization**
- `init()` - Called on page load
  - Parse `LABELGUARD_DATA` from embedded script
  - Load image from `image_path`
  - Render initial UI

### 2. **Image Rendering**
- `loadImage(imagePath)` - Load image into canvas
  - Create Image object
  - Set canvas dimensions
  - Calculate initial scale to fit viewport
  - Draw image on canvas

### 3. **Bbox Rendering**
- `renderBboxes(textBlocks)` - Draw all bboxes on canvas
  - For each TextBlock:
    - Determine color: red (failed validation) or green (passed)
    - Draw rectangle outline
    - Draw block index label
  - Handle zoom/pan transformations

### 4. **Error Highlighting**
- `renderErrorHighlights(ruleCheckResults)` - Draw error highlights
  - For each RuleCheckResult with visual_markers:
    - Draw highlight overlays (semi-transparent colored rectangles)
    - Use colors from VisualMarker.color

### 5. **Error List Rendering**
- `renderErrorList(ruleCheckResults)` - Generate error summary HTML
  - Count errors by rule type
  - Group errors by text block
  - Generate HTML for error summary
  - Show: error count, block index, rule type, score

### 6. **Block Details Rendering**
- `renderBlockDetails(textBlock, ruleCheckResults)` - Show block details on click
  - Generate HTML from TextBlock and related RuleCheckResults
  - Show: block index, type, text preview, validation details
  - Include rule-specific details (etalon matching, allergens, numbers)

### 7. **Click Handling**
- `handleCanvasClick(x, y)` - Handle clicks on canvas
  - Convert canvas coordinates to image coordinates
  - Find clicked bbox (if any)
  - If bbox clicked → show block details
  - If whitespace clicked → show error list

### 8. **Filtering**
- `initializeFiltering()` - Set up rule type filters
  - Create checkboxes for each rule type
  - Filter error display by selected rule types
  - Update counts dynamically

### 9. **Zoom/Pan**
- `handleZoom(event)` - Zoom with Ctrl+wheel
- `handlePan()` - Pan with mouse drag
- `drawCanvas()` - Redraw canvas with current transform

### 10. **Bbox Editing** (New Feature - Main)
- `enableBboxEditing()` - Enable edit mode
- `handleBboxResize(blockId, newBbox)` - Resize bbox (drag corners/edges)
- `handleBboxMove(blockId, newBbox)` - Move bbox (drag center)
- `createNewBlock(bbox)` - Create new block (draw rectangle)
- `deleteBlock(blockId)` - Delete block (keyboard or button)
- `changeCategory(blockId, newCategory)` - Change block category (dropdown)
- `trackModifications()` - Track modified flag (set to true on any edit)

### 11. **Data Extraction for Server**
- `extractLightweightBlocks()` - Extract blocks for `/Analyze` request
  - For each TextBlock:
    - Extract: id, bbox, category, text (if provided), modified
  - Return array of block dicts

### 12. **Server Communication**
- `sendCorrections()` - POST to `/Analyze` endpoint
  - Extract lightweight blocks
  - Send JSON: `{image_path, blocks: [...]}`
  - Receive fresh LabelProcessingResult
  - Re-render UI with new data

### 13. **Generate Overlayed Image** (New Feature)
- `generateOverlayedImage()` - Generate image with all overlays
  - Draw image on temporary canvas
  - Draw all bboxes (red/green outlines)
  - Draw all error highlights (semi-transparent overlays)
  - Draw block index labels
  - Convert canvas to blob/image
  - Trigger download or display in new window
  - Option: Send to server endpoint to save as artifact

## Data Flow

### Initial Load (from `/StartProcessing`):
1. HTML loads with embedded `LABELGUARD_DATA`
2. `init()` called
3. `loadImage()` loads image from `image_path`
4. `renderBboxes()` draws all bboxes
5. `renderErrorHighlights()` draws error highlights
6. `renderErrorList()` shows error summary

### User Interaction:
1. User clicks bbox → `handleCanvasClick()` → `renderBlockDetails()`
2. User clicks whitespace → `handleCanvasClick()` → `renderErrorList()`
3. User filters rules → `updateFilteredDisplay()`
4. User zooms/pans → `handleZoom()` / `handlePan()` → `drawCanvas()`

### After `/Analyze` (re-analysis):
1. User clicks "Analyze" → `sendCorrections()`
2. Server returns fresh `LabelProcessingResult`
3. Update `LABELGUARD_DATA.labelProcessingResult`
4. Re-render: `renderBboxes()`, `renderErrorHighlights()`, `renderErrorList()`

## UI Features

### Visual Elements:
- **Canvas**: Image with bboxes and highlights
- **Bbox Colors**: 
  - Red outline = failed validation
  - Green outline = passed validation
- **Error Highlights**: Semi-transparent colored overlays on error words
- **Block Labels**: Block index shown at bottom-right of each bbox

### Interactions:
- **Click bbox**: Show detailed block information
- **Click whitespace**: Show error list summary
- **Ctrl + Mouse Wheel**: Zoom in/out
- **Mouse Drag**: Pan image
- **Filter Checkboxes**: Show/hide errors by rule type
- **Analyze Button**: Send corrections to server

### Responsive Behavior:
- Canvas scales to fit viewport
- Maintains aspect ratio
- Scrollable message panel if content overflows

## Data Structures Used

### From `LABELGUARD_DATA`:
```javascript
{
  "image_path": "artifacts/labelguard/temp/uuid.jpg",
  "labelProcessingResult": {
    "text_blocks": [
      {
        "index": "0",
        "bbox": [x1, y1, x2, y2],
        "text": "...",
        "type": "A",
        "modified": false,
        "sentences": [...],
        "words": [...]
      }
    ],
    "rule_check_results": [
      {
        "rule_name": "etalon",
        "passed": false,
        "score": 85.0,
        "text_block": {...},
        "visual_markers": [
          {
            "type": "highlight",
            "bbox": [x1, y1, x2, y2],
            "color": [255, 100, 100],
            "opacity": 0.4
          }
        ],
        "affected_words": [...]
      }
    ],
    "kmat": "...",
    "version": "...",
    "original_filename": "..."
  }
}
```

## Error Handling

### Client-Side:
- Image load failure: Show error message
- Invalid JSON: Show error message
- Network error on `/Analyze`: Show error, keep current state

### Server Communication:
- Handle HTTP errors (400, 404, 500)
- Display error messages to user
- Retry logic (optional, future)

## Performance Considerations

- **Image Loading**: Load once, cache in memory
- **Canvas Rendering**: Only redraw on changes (zoom/pan/updates)
- **Error List**: Generate HTML once, update on filter changes
- **Large Images**: Scale down for display if needed

## Future Enhancements (Phase 2)

1. **Undo/Redo**: Track edit history for bbox modifications

2. **Keyboard Shortcuts**: 
   - Arrow keys to navigate blocks
   - Delete key to remove block
   - Ctrl+Z for undo

3. **Export Options**: 
   - Save corrections to JSON file
   - Export overlayed image in different formats

4. **Comparison View**: Side-by-side before/after validation results

5. **Batch Operations**: 
   - Select multiple blocks
   - Bulk category changes
   - Bulk delete

