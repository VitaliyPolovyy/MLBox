## LabelGuard – Business Requirements (Human‑in‑the‑Loop Label Validation)

### 1. Purpose

- **Goal**: Enable an external system (ERP) and a human reviewer to:
  - Upload a label image together with “etalon” reference text.
  - Automatically detect text blocks on the image and validate them.
  - Manually correct bounding boxes and categories on top of the automatic result.
  - Re‑run validation iteratively until the label is accepted.

---

### 2. Actors

- **ERP System**
  - Uploads label image and etalon files to LabelGuard (via `POST /labelguard/upload`).
  - Sends file paths to the LabelGuard analysis API (`POST /labelguard/analyze`).
  - Receives URLs to the analyzed image and its validation data.
  - Opens the human review UI in the browser for a specific label (e.g., `/labelguard/viewer.html?data_id=...`).

- **Human User (Label Reviewer)**
  - Works in the browser UI.
  - Reviews automated results.
  - Edits blocks (bboxes and categories).
  - Triggers re‑analysis until satisfied.

- **LabelGuard Service**
  - Exposes a file upload API (`POST /labelguard/upload`) for receiving image and etalon files.
  - Exposes an analysis API (`POST /labelguard/analyze`) for processing labels.
  - Serves the viewer UI (`/labelguwhereard/viewer.html`).
  - Stores intermediate artifacts (images, JSON results, corrections).
  - Performs detection, OCR, categorization, and validation.
  - Returns results in a structured JSON format suitable for the UI.

---

### 3. High‑Level Workflow

#### 3.1 Initial Analysis

- **Step 1: File Upload (from ERP)**
  - Upload label image file via `POST /labelguard/upload`.
  - Upload etalon JSON file via `POST /labelguard/upload` (optional).
  - Server saves files and returns file paths.

- **Step 2: Analysis Request (from ERP)**
  - Send file paths to `POST /labelguard/analyze`:
    - **Image path** (from upload response).
    - **Etalon path** (from upload response, optional).
    - An **empty list of blocks** (meaning: “detect layout normally”).

- **Expected behavior of LabelGuard**
  - Run full pipeline:
    - Layout detection.
    - OCR on detected regions.
    - Text block categorization.
    - Validation against etalon and other rules.
  - Persist artifacts:
    - Analyzed image.
    - Validation JSON.
  - Return to ERP:
    - Public path/URL to analyzed image.
    - Public path/URL (or endpoint) for validation JSON.
    - An **identifier** for this label (e.g. `data_id` / `original_filename`).

#### 3.2 Open Review UI

- **ERP behavior**
  - Opens a Viewer UI URL in a browser: `/labelguard/viewer.html?data_id={original_filename}`.
  - The `data_id` is the `original_filename` returned from the analyze response.

- **Expected UI behavior**
  - Load the image for that label from LabelGuard.
  - Load the corresponding validation JSON.
  - Render:
    - The image with bounding boxes and validation highlights.
    - An error list and per‑block details.

#### 3.3 User Review & Correction

- **User capabilities**
  - **Navigation**
    - Pan and zoom the image for comfortable viewing.
  - **Selection**
    - Click a block to see its details.
    - Click background/whitespace to see an aggregated error list or overview.
  - **Editing blocks**
    - Move and resize existing bounding boxes.
    - Create new blocks.
    - Delete blocks.
    - Change a block’s category (from a predefined set).
  - **Text**
    - Text content is **read‑only** in the UI (cannot be edited directly).

#### 3.4 Re‑Analysis with Corrections

- **User action**
  - Clicks **Analyze** in the UI when they want updated validation for the current set of blocks.

- **Data sent from UI to LabelGuard**
  - A **lightweight list of blocks**, each containing at least:
    - Block **ID** (string identifier).
    - **Bounding box** coordinates.
    - **Category** (may be empty/omitted to request auto‑detection).
    - Optional **text** (if the client chooses to send it; otherwise OCR is rerun).
    - **`modified` flag**.
  - The **image identifier/path** (so backend knows which stored image to use).
  - Optional metadata such as `kmat` and `version` if needed by the backend.

- **Expected behavior of LabelGuard**
  - Treat provided blocks as **authoritative** for their regions.
  - For the rest of the image:
    - Perform layout detection to find additional blocks.
    - Ensure that provided blocks have priority over automatically detected ones.
  - For **all** blocks (provided + newly detected):
    - Run OCR (or reuse provided text).
    - Build full text structure (sentences, words, formatting).
    - Run category detection if category is empty.
    - Run validation rules.
  - Return a **fresh validation JSON** that reflects the current layout and corrections.
  - Persist a simplified “corrections” view (latest blocks and metadata) for that label.

#### 3.5 Iteration

- The user may repeat:
  - Adjust blocks and categories → Analyze → Review results.
- The flow ends when:
  - User and ERP consider the label accepted / good enough (decision made outside LabelGuard).

---

### 4. Data & Semantics

#### 4.1 Etalon Input

- ERP provides an **etalon** as a list of reference text blocks.
- Each etalon block includes at least:
  - A **type/category** (e.g. product name, ingredients).
  - **Language code**.
  - **Text content**.
- LabelGuard uses etalon to drive validation rules (e.g. matching, coverage, consistency).

#### 4.2 Blocks on the Image

- Each block represents a region of text on the label image.
- Each block has:
  - A unique **ID** (string).
  - A **bounding box** on the image.
  - A **category**:
    - Can be provided by ERP, auto‑detected, or edited by the user.
  - Optional **text** value:
    - If omitted, backend is responsible for OCR on that region.
  - A **`modified` flag**:
    - Indicates whether the block has ever been changed by the user.

#### 4.3 `modified` Flag – Business Rules

- For blocks **created by automatic detection**:
  - `modified = false` initially.
- For blocks **touched by the user**:
  - Any user action sets `modified = true`:
    - Move or resize a bbox.
    - Change category.
    - Create a new block.
  - Once `modified` is `true`, it **never becomes false** again.
- **Deleted blocks**:
  - Are completely omitted from update payloads and from saved corrections data.
  - No special flag is required for “deleted”.

#### 4.4 Validation Result (Conceptual)

- LabelGuard produces a result that includes:
  - A list of **blocks**:
    - Bounding box, text, category, `modified` flag, and other derived structure.
  - A list of **validation findings**:
    - For each rule, whether it passed or failed.
    - Which blocks or parts of the text are affected.
  - Relevant **metadata** (e.g. kmat, version, label identifier).

- The UI must be able to:
  - Show which blocks failed which rules.
  - Highlight problematic regions directly on the image.
  - Present both:
    - An overview of all issues.
    - A detailed view for a specific block.

---

### 5. UI Requirements (Behavioral)

#### 5.1 Layout

- The Viewer UI presents:
  - **Left side**: Image canvas with bboxes and visual highlights.
  - **Right side**: Controls and information:
    - Analyze button (re‑run analysis with current blocks).
    - Optional button to download a combined “overlayed” image.
    - Area for:
      - Error list.
      - Block details.
    - Filters (e.g. by rule type) to control which issues are visible.

#### 5.2 Interactions

- The user can:
  - Pan and zoom the image.
  - Select blocks by clicking them.
  - Switch between:
    - A list of all validation errors.
    - A detail view for the selected block.
  - Edit blocks (move, resize, create, delete).
  - Change block categories from a predefined set.
  - Trigger **Analyze** to send current blocks and obtain updated validation results.

#### 5.3 Resilience

- If loading data or running analysis fails, the UI should:
  - Display a clear error message.
  - Avoid discarding already visible data unless a full reload is required.

---

### 6. Non‑Functional Requirements

#### 6.1 Performance

- UI should remain responsive for typical label sizes and realistic numbers of blocks.
- Re‑analysis latency should be acceptable for interactive human‑in‑the‑loop use.

#### 6.2 Persistence

- LabelGuard must persist per label:
  - The analyzed image.
  - The latest set of blocks (including `modified` flags) and relevant metadata.
- It is acceptable in the current scope to:
  - Overwrite previous corrections for the same label.
  - Not maintain a historical version timeline.

#### 6.3 Error Handling (Business Level)

- For **invalid input** (e.g. malformed request, impossible bbox coordinates):
  - LabelGuard returns a clear error instead of partial results.
- For **non‑critical internal issues** (e.g. category detection fails for one block):
  - Mark the affected part as unknown or degraded.
  - Still return a usable result and log the problem internally.


