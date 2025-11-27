// LabelGuard UI JavaScript
// This file contains all the rendering and interaction logic

console.log('LabelGuard UI script loaded');

// Global variables
let canvas, ctx, img;
let scale = 1;
let offsetX = 0, offsetY = 0;
let isDragging = false;
let startX = 0, startY = 0;

// Bbox editing state
// editMode === true means the currently selected block is in edit mode
// and can be moved/resized. When false, blocks are only selectable and
// the image pans on drag.
let editMode = false;
let selectedBboxIndex = null;
let isResizing = false;
let resizeHandle = null; // 'nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w', 'move'
let isCreatingBbox = false;
let createStartX = 0, createStartY = 0;
let hoveredBboxIndex = null;

// Resize handle size (in canvas coordinates, scaled)
const HANDLE_SIZE = 8;

// Global variable for data
let LABELGUARD_DATA = null;

// Global variable for text block categories (code -> name mapping)
let CATEGORIES = {};

// Global variable for sentence categories (code -> name mapping)
let SENTENCE_CATEGORIES = {};

// Server configuration
// For development: if using Live Preview (port 3000), use relative paths
// For production: API server is on the same origin
const USE_RELATIVE_PATHS = (() => {
    const hostname = window.location.hostname;
    const port = window.location.port;
    // Use relative paths when running on Live Preview (localhost:3000)
    return hostname === 'localhost' && port === '3000';
})();

const API_SERVER_URL = (() => {
    if (USE_RELATIVE_PATHS) {
        return ''; // Empty string = use relative paths
    }
    // For Ray Serve or production
    const hostname = window.location.hostname;
    const port = window.location.port;
    if (hostname === 'localhost' && port === '3000') {
        return 'http://localhost:8001'; // Ray Serve
    }
    return window.location.origin;
})();

// Load categories from JSON file
async function loadCategories() {
    try {
        // Use relative path for Live Preview, or API_SERVER_URL for Ray Serve
        let categoriesPath;
        if (USE_RELATIVE_PATHS) {
            categoriesPath = '../../../assets/labelguard/json/data-categories.json';
        } else {
            // Ray Serve: add /labelguard route prefix
            categoriesPath = `${API_SERVER_URL}/labelguard/assets/labelguard/json/data-categories.json`;
        }
        
        const response = await fetch(categoriesPath);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        CATEGORIES = await response.json();
        console.log('Categories loaded:', CATEGORIES);
        
        // Populate category dropdown
        populateCategoryDropdown();
    } catch (error) {
        console.error('Failed to load categories:', error);
        // Fallback: use empty categories object
        CATEGORIES = {};
    }
}

// Load sentence categories for \"Структура текста\" section
async function loadSentenceCategories() {
    try {
        let path;
        if (USE_RELATIVE_PATHS) {
            path = '../../../assets/labelguard/json/ingr-sentence-categories.json';
        } else {
            // Ray Serve: add /labelguard route prefix
            path = `${API_SERVER_URL}/labelguard/assets/labelguard/json/ingr-sentence-categories.json`;
        }
        const response = await fetch(path);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        SENTENCE_CATEGORIES = await response.json();
        console.log('Sentence categories loaded:', SENTENCE_CATEGORIES);
    } catch (error) {
        console.error('Failed to load sentence categories:', error);
        SENTENCE_CATEGORIES = {};
    }
}

// Populate category dropdown with code -> name mapping
function populateCategoryDropdown() {
    const categorySelect = document.getElementById('categorySelect');
    if (!categorySelect) return;
    
    // Clear existing options (except first empty one)
    categorySelect.innerHTML = '<option value="">-- Select --</option>';
    
    // Add options from categories JSON
    for (const [code, info] of Object.entries(CATEGORIES)) {
        const option = document.createElement('option');
        option.value = code;  // Store code as value
        // Display only human-readable name from JSON, without code letter
        option.textContent = info.name || code;
        categorySelect.appendChild(option);
    }
}

// Initialize on page load
// Handle both cases: script loaded before or after DOMContentLoaded
async function initializeUI() {
    console.log('DOM loaded, initializing UI');
    
    canvas = document.getElementById('labelCanvas');
    ctx = canvas.getContext('2d');
    const msgDiv = document.getElementById('msgContent');
    
    // Load categories first
    await loadCategories();
    await loadSentenceCategories();
    
    // Get data_id from URL parameters
    const urlParams = new URLSearchParams(window.location.search);
    const dataId = urlParams.get('data_id');
    
    if (!dataId) {
        // Fallback: check if embedded data exists (for development)
        if (typeof LABELGUARD_DATA !== 'undefined' && LABELGUARD_DATA) {
            console.log('Using embedded LABELGUARD_DATA (development mode)');
            loadData(LABELGUARD_DATA);
        } else {
            console.error('No data_id in URL and no embedded data found');
            msgDiv.innerHTML = '<div style="color: red;">Error: No data_id parameter in URL. Usage: viewer.html?data_id=filename</div>';
        }
        return;
    }
    
    console.log('Fetching data for data_id:', dataId);
    
    // Fetch JSON data from server
    // Use relative paths for Live Preview, or API_SERVER_URL for Ray Serve/production
    let jsonPath;
    if (USE_RELATIVE_PATHS) {
        // Live Preview: relative path from assets/labelguard/html/ to artifacts/labelguard/
        // HTML is at: assets/labelguard/html/viewer.html
        // Artifacts are at: artifacts/labelguard/filename.json
        // Go up 3 levels: ../../../ (html -> labelguard -> assets -> root)
        // Then into: artifacts/labelguard/
        jsonPath = `../../../artifacts/labelguard/` + encodeURIComponent(dataId) + '.json';
    } else {
        // Ray Serve: /labelguard/artifacts/labelguard/{filename}.json
        jsonPath = `${API_SERVER_URL}/labelguard/artifacts/labelguard/` + encodeURIComponent(dataId) + '.json';
    }
    console.log('Fetching JSON from:', jsonPath);
    fetch(jsonPath)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Data loaded successfully:', data);
            LABELGUARD_DATA = data;
            loadData(data);
        })
        .catch(error => {
            console.error('Failed to load data:', error);
            msgDiv.innerHTML = '<div style="color: red;">Error loading data: ' + error.message + '</div>';
            msgDiv.innerHTML += '<div>Expected file: ' + jsonPath + '</div>';
        });
    
    // Setup event listeners
    setupEventListeners();

    // Setup resizable split between image and message panels
    setupSplitPane();
}

// Initialize when DOM is ready - handle both early and late script loading
if (document.readyState === 'loading') {
    // DOM hasn't loaded yet, wait for it
    document.addEventListener('DOMContentLoaded', initializeUI);
} else {
    // DOM already loaded (script loaded dynamically after page load)
    initializeUI();
}

// Make the vertical separator between image and details panel draggable
function setupSplitPane() {
    const imagePanel = document.getElementById('imagePanel');
    const messagePanel = document.getElementById('messagePanel');
    const divider = document.getElementById('divider');

    if (!imagePanel || !messagePanel || !divider) {
        console.warn('SplitPane: required elements not found');
        return;
    }

    let isResizing = false;

    divider.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isResizing = true;
        document.body.classList.add('resizing');
        console.log('SplitPane: start resizing');
    });

    window.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const totalWidth = window.innerWidth;
        const minPercent = 20;
        const maxPercent = 80;

        let leftPercent = (e.clientX / totalWidth) * 100;
        if (leftPercent < minPercent) leftPercent = minPercent;
        if (leftPercent > maxPercent) leftPercent = maxPercent;

        imagePanel.style.flex = `0 0 ${leftPercent}%`;
        messagePanel.style.flex = `0 0 ${100 - leftPercent}%`;

        // Recompute canvas scale and offset to keep image centered in resized panel
        if (img && canvas) {
            const panelWidth = imagePanel.clientWidth;
            const panelHeight = imagePanel.clientHeight;
            const scaleX = (panelWidth - 50) / img.width;
            const scaleY = (panelHeight - 50) / img.height;
            scale = Math.min(scaleX, scaleY, 1);
            offsetX = (panelWidth - img.width * scale) / 2;
            offsetY = (panelHeight - img.height * scale) / 2;
            drawCanvas();
        }
    });

    window.addEventListener('mouseup', () => {
        if (!isResizing) return;
        isResizing = false;
        document.body.classList.remove('resizing');
        console.log('SplitPane: stop resizing');
    });
}

// Load data and render UI
function loadData(data) {
    LABELGUARD_DATA = data;
    
    // Load image
    img = new Image();
    // Note: crossOrigin removed - causes 401 with Live Preview security
    
    // URL encode the image path to handle special characters (Cyrillic, etc.)
    // Construct full URL - handle paths starting with / by using API_SERVER_URL or relative paths
    let imagePath = data.image_path;
    if (imagePath.startsWith('/')) {
        if (USE_RELATIVE_PATHS) {
            // Live Preview: convert /artifacts/labelguard/... to relative path
            // HTML is at: assets/labelguard/html/viewer.html
            // Artifacts are at: artifacts/labelguard/filename.jpg
            // Go up 3 levels: ../../../ (html -> labelguard -> assets -> root)
            // /artifacts/labelguard/filename.jpg -> ../../../artifacts/labelguard/filename.jpg
            if (imagePath.startsWith('/artifacts/labelguard/')) {
                const filename = imagePath.replace('/artifacts/labelguard/', '');
                imagePath = `../../../artifacts/labelguard/${filename}`;
            } else if (imagePath.startsWith('/artifacts/')) {
                const relativePath = imagePath.replace('/artifacts/', '');
                imagePath = `../../../artifacts/${relativePath}`;
            }
        } else {
            // Ray Serve: convert /artifacts/... to /labelguard/artifacts/...
            if (imagePath.startsWith('/artifacts/')) {
                imagePath = '/labelguard' + imagePath;
            }
            imagePath = API_SERVER_URL + imagePath;
        }
    }
    imagePath = encodeURI(imagePath);
    console.log('Loading image from:', data.image_path);
    console.log('Encoded full URL:', imagePath);
    console.log('Current page URL:', window.location.href);
    
    img.onload = function() {
        console.log('Image loaded successfully! Dimensions:', img.width, 'x', img.height);
        // Set canvas size to match image
        canvas.width = img.width;
        canvas.height = img.height;
        
        // Calculate initial scale to fit viewport
        const panel = document.getElementById('imagePanel');
        const scaleX = (panel.clientWidth - 50) / img.width;
        const scaleY = (panel.clientHeight - 50) / img.height;
        scale = Math.min(scaleX, scaleY, 1);
        
        // Center the image
        offsetX = (panel.clientWidth - img.width * scale) / 2;
        offsetY = (panel.clientHeight - img.height * scale) / 2;
        
        drawCanvas();
        renderErrorList();
        initializeFiltering();
    };
    
    img.onerror = function() {
        console.error('Failed to load image:', data.image_path);
        console.error('Tried to load from:', imagePath);
        // Draw placeholder
        canvas.width = 800;
        canvas.height = 600;
        ctx.fillStyle = '#f0f0f0';
        ctx.fillRect(0, 0, 800, 600);
        ctx.fillStyle = '#d00';
        ctx.font = '20px Arial';
        ctx.fillText('Image failed to load: ' + data.image_path, 50, 50);
        ctx.fillText('Check browser console (F12) for details', 50, 80);
    };
    
    img.src = imagePath; // Use encoded path
}

// Draw everything on canvas
function drawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);
    
    // Draw image only if it's loaded successfully
    if (img && img.complete && img.naturalWidth > 0) {
        ctx.drawImage(img, 0, 0);
    }
    
    // Draw error highlights first (behind bboxes)
    renderErrorHighlights();
    
    // Draw bboxes
    renderBboxes();
    
    ctx.restore();
}

// Render bboxes
function renderBboxes() {
    if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) return;
    const textBlocks = LABELGUARD_DATA.labelProcessingResult.text_blocks;
    const ruleResults = LABELGUARD_DATA.labelProcessingResult.rule_check_results;
    
    // Group rule results by text block
    const blockResults = {};
    ruleResults.forEach(result => {
        if (result.text_block && result.text_block.index) {
            const blockId = result.text_block.index;
            if (!blockResults[blockId]) {
                blockResults[blockId] = [];
            }
            blockResults[blockId].push(result);
        }
    });
    
    textBlocks.forEach(block => {
        const [x1, y1, x2, y2] = block.bbox;
        const width = x2 - x1;
        const height = y2 - y1;
        
        // Determine color: red if any failed, green if all passed
        const results = blockResults[block.index] || [];
        const hasError = results.some(r => !r.passed);
        const color = hasError ? '#ff0000' : '#00ff00';
        const isSelected = selectedBboxIndex === block.index;
        const isHovered = hoveredBboxIndex === block.index;
        
        // Draw bbox outline (thicker if selected/hovered)
        ctx.strokeStyle = color;
        ctx.lineWidth = isSelected ? 5 : (isHovered ? 4 : 3);
        ctx.strokeRect(x1, y1, width, height);
        
        // Draw selection highlight only when in edit mode
        if (editMode && isSelected) {
            ctx.fillStyle = 'rgba(100, 150, 255, 0.2)';
            ctx.fillRect(x1, y1, width, height);
        }
        
        // Draw resize handles only when in edit mode
        if (editMode && isSelected) {
            drawResizeHandles(x1, y1, x2, y2);
        }
    });
}

// Draw resize handles for a bbox
function drawResizeHandles(x1, y1, x2, y2) {
    const handleSize = HANDLE_SIZE / scale; // Adjust for current zoom
    const halfHandle = handleSize / 2;
    
    ctx.fillStyle = '#0066ff';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    
    // Corner handles
    const handles = [
        { x: x1, y: y1, type: 'nw' }, // Top-left
        { x: x2, y: y1, type: 'ne' }, // Top-right
        { x: x1, y: y2, type: 'sw' }, // Bottom-left
        { x: x2, y: y2, type: 'se' }, // Bottom-right
        { x: (x1 + x2) / 2, y: y1, type: 'n' }, // Top
        { x: (x1 + x2) / 2, y: y2, type: 's' }, // Bottom
        { x: x1, y: (y1 + y2) / 2, type: 'w' }, // Left
        { x: x2, y: (y1 + y2) / 2, type: 'e' }  // Right
    ];
    
    handles.forEach(handle => {
        ctx.beginPath();
        ctx.rect(handle.x - halfHandle, handle.y - halfHandle, handleSize, handleSize);
        ctx.fill();
        ctx.stroke();
    });
}

// Render error highlights
function renderErrorHighlights() {
    if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) return;
    const ruleResults = LABELGUARD_DATA.labelProcessingResult.rule_check_results;
    
    ruleResults.forEach(result => {
        if (result.visual_markers) {
            result.visual_markers.forEach(marker => {
                if (marker.type === 'highlight') {
                    const [x1, y1, x2, y2] = marker.bbox;
                    const width = x2 - x1;
                    const height = y2 - y1;
                    
                    // Draw semi-transparent highlight
                    ctx.fillStyle = `rgba(${marker.color[0]}, ${marker.color[1]}, ${marker.color[2]}, ${marker.opacity || 0.4})`;
                    ctx.fillRect(x1, y1, width, height);
                }
            });
        }
    });
}

// Render error list
function renderErrorList() {
    const ruleResults = LABELGUARD_DATA.labelProcessingResult.rule_check_results;
    const msgDiv = document.getElementById('msgContent');
    const filterPanel = document.getElementById('filterPanel');
    
    // In error list mode, show the filter panel
    if (filterPanel) {
        filterPanel.style.display = 'block';
    }
    
    // Count errors
    const criticalErrors = ruleResults.filter(r => !r.passed);
    const passedCount = ruleResults.filter(r => r.passed).length;
    
    if (criticalErrors.length === 0) {
        msgDiv.innerHTML = '<div class="lg-heading lg-heading--ok">✅ Немає критичних помилок!</div>';
        return;
    }
    
    // Group by block
    const errorsByBlock = {};
    criticalErrors.forEach(error => {
        if (error.text_block && error.text_block.index) {
            const blockId = error.text_block.index;
            if (!errorsByBlock[blockId]) {
                errorsByBlock[blockId] = [];
            }
            errorsByBlock[blockId].push(error);
        }
    });
    
    let html = `<div class="lg-heading lg-heading--error">❌ Критичних помилок - ${criticalErrors.length}</div>`;
    html += `<div class="lg-heading">✅ Всього перевірок - ${passedCount}</div>`;
    html += '<hr class="lg-separator">';
    
    Object.keys(errorsByBlock).forEach(blockId => {
        const block = LABELGUARD_DATA.labelProcessingResult.text_blocks.find(b => b.index === blockId);
        const errors = errorsByBlock[blockId];
        
        html += `<div class="lg-section lg-error-block">`;
        
        // Header: category name + languages + block index, plus preview text
        let categoryName = 'N/A';
        if (block) {
            if (block.type && CATEGORIES[block.type] && CATEGORIES[block.type].name) {
                categoryName = CATEGORIES[block.type].name;
            } else if (block.type) {
                categoryName = block.type;
            }
        }
        const languages = block
            ? (Array.isArray(block.languages)
                ? block.languages
                : (block.languages ? [block.languages] : []))
            : [];
        const langPart = languages.length ? ` (${languages.join(', ')})` : '';

        // Build header text with inline preview
        let headerHtml = `${categoryName}${langPart} #${blockId}`;

        // Text preview: first part of block text, if available, appended inline
        if (block && block.text) {
            const previewLimit = 140;
            let preview = block.text.trim();
            if (preview.length > previewLimit) {
                preview = preview.substring(0, previewLimit) + '...';
            }
            headerHtml += ` <span class="lg-muted">${preview}</span>`;
        }
        
        html += `<div class="lg-block-header">${headerHtml}</div>`;
        
        // Per-rule error details
        errors.forEach(error => {
            if (error.html_details) {
                html += `<div class="rule-error lg-rule-error" data-rule-${error.rule_name}>${error.html_details}</div>`;
            }
        });
        
        html += `</div>`;
    });
    
    msgDiv.innerHTML = html;

    // Re-apply current filter settings after re-rendering the list,
    // but only if filters have already been initialized.
    if (typeof updateFilteredDisplay === 'function' &&
        document.querySelectorAll('.rule-filter').length > 0) {
        updateFilteredDisplay();
    }
}

// Initialize filtering
function initializeFiltering() {
    if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) return;
    const filterPanel = document.getElementById('filterPanel');
    if (!filterPanel) return;
    
    const ruleNames = {
        'etalon': 'Перевірка відповідності тексту',
        'allergens': 'Алергени',
        'numbers': 'Числа в інгредієнтах'
    };
    
    const ruleResults = LABELGUARD_DATA.labelProcessingResult.rule_check_results;
    const ruleCounts = {};
    ruleResults.forEach(r => {
        ruleCounts[r.rule_name] = (ruleCounts[r.rule_name] || 0) + (r.passed ? 0 : 1);
    });
    
    let html = '';
    Object.keys(ruleNames).forEach(ruleKey => {
        const count = ruleCounts[ruleKey] || 0;
        const hasErrors = count > 0;
        html += `<label>
            <input type="checkbox" class="rule-filter" value="${ruleKey}" checked 
                   style="transform: scale(1.2); vertical-align: middle; margin-right: 6px;">
            ${ruleNames[ruleKey]} 
            <span class="rule-count ${hasErrors ? 'has-errors' : 'no-errors'}">(${count})</span>
        </label>`;
    });
    
    filterPanel.innerHTML = html;
    
    // Add event listeners
    filterPanel.addEventListener('change', function(e) {
        if (e.target.classList.contains('rule-filter')) {
            updateFilteredDisplay();
        }
    });

    // Apply initial filter state (all checked) to current content
    updateFilteredDisplay();
}

// Update filtered display
function updateFilteredDisplay() {
    const enabledRules = new Set();
    document.querySelectorAll('.rule-filter:checked').forEach(cb => {
        enabledRules.add(cb.value);
    });
    
    const msgDiv = document.getElementById('msgContent');
    const emptyStateId = 'no-errors-filtered';
    let emptyState = document.getElementById(emptyStateId);
    
    // If no rules are enabled, hide all error sections and show an info message
    if (enabledRules.size === 0) {
        document.querySelectorAll('.lg-section').forEach(sec => {
            const hasRuleErrors = sec.querySelector('.rule-error') !== null;
            if (hasRuleErrors) {
                sec.style.display = 'none';
            } else {
                sec.style.display = '';
            }
        });
        if (!emptyState && msgDiv) {
            emptyState = document.createElement('div');
            emptyState.id = emptyStateId;
            emptyState.className = 'lg-muted';
            emptyState.style.marginTop = '10px';
            emptyState.textContent = 'Фільтри вимкнули всі типи помилок. Увімкніть хоча б один тип, щоб побачити список.';
            msgDiv.appendChild(emptyState);
        }
        return;
    } else {
        // Some rules enabled: remove empty state message
        if (emptyState && emptyState.parentNode) {
            emptyState.parentNode.removeChild(emptyState);
        }
    }
    
    document.querySelectorAll('.rule-error').forEach(errorDiv => {
        let shouldShow = false;
        enabledRules.forEach(rule => {
            if (errorDiv.hasAttribute(`data-rule-${rule}`)) {
                shouldShow = true;
            }
        });
        errorDiv.style.display = shouldShow ? '' : 'none';
    });

    // Hide sections that have no visible rule-error for enabled rules
    document.querySelectorAll('.lg-section').forEach(sec => {
        const ruleErrors = sec.querySelectorAll('.rule-error');
        if (ruleErrors.length === 0) {
            // Not an error container section (e.g. text structure), keep visible
            return;
        }
        const hasVisible = Array.from(ruleErrors).some(div => div.style.display !== 'none');
        sec.style.display = hasVisible ? '' : 'none';
    });
}

// Helper function to delete selected bbox
function deleteSelectedBbox() {
    if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) return;
    if (selectedBboxIndex !== null) {
        const textBlocks = LABELGUARD_DATA.labelProcessingResult.text_blocks;
        const index = textBlocks.findIndex(b => b.index === selectedBboxIndex);
        if (index !== -1) {
            textBlocks.splice(index, 1);
            selectedBboxIndex = null;
            const categorySelector = document.getElementById('categorySelector');
            const languageLabel = document.getElementById('languageLabel');
            categorySelector.style.display = 'none';
            if (languageLabel) {
                languageLabel.textContent = '';
            }
            drawCanvas();
            renderErrorList();
        }
    }
}

// Setup event listeners
function setupEventListeners() {
    const categorySelector = document.getElementById('categorySelector');
    const categorySelect = document.getElementById('categorySelect');
    const languageLabel = document.getElementById('languageLabel');
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    // Category change
    categorySelect.addEventListener('change', function() {
        if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) return;
        if (selectedBboxIndex !== null) {
            const textBlocks = LABELGUARD_DATA.labelProcessingResult.text_blocks;
            const block = textBlocks.find(b => b.index === selectedBboxIndex);
            if (block) {
                block.type = categorySelect.value;
                block.modified = true;
                drawCanvas();
            }
        }
    });
    
    // Analyze button
    analyzeBtn.addEventListener('click', async function() {
        if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) {
            console.error('No data available for analysis');
            return;
        }
        
        // Check if Ray Serve is needed (when using relative paths/Live Preview)
        if (USE_RELATIVE_PATHS) {
            // Test if Ray Serve is available on default HTTP port 8000
            try {
                const testResponse = await fetch('http://localhost:8000/labelguard/analyze', {
                    method: 'OPTIONS'  // Just check if server responds
                });
            } catch (e) {
                const msgDiv = document.getElementById('msgContent');
                msgDiv.innerHTML = '<div style="color: orange; padding: 20px; font-size: 16px;">' +
                    '⚠️ <strong>Ray Serve is not running</strong><br>' +
                    'The Analyze button requires Ray Serve to be running on port 8000.<br>' +
                    'For now, you can view and edit bboxes. Start Ray Serve to use Analyze.</div>';
                return;
            }
        }
        
        // Disable button during processing
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        
        try {
            // Collect current bbox data
            const textBlocks = LABELGUARD_DATA.labelProcessingResult.text_blocks;
            const blocks = textBlocks.map(block => {
                const rawBbox = Array.isArray(block.bbox) ? block.bbox : [block.bbox];
                // Normalize bbox coordinates to integers (server expects ints for PIL)
                const intBbox = rawBbox.map(v => Math.round(v));
                return {
                    id: block.index,
                    bbox: intBbox,
                    category: block.type || '',
                    // text is not needed for re-analysis payload; server will OCR if needed
                    modified: !!block.modified
                };
            });
            
            // Get image path and metadata
            // The image_path from LABELGUARD_DATA is already the HTTP path (e.g., /artifacts/labelguard/filename.jpg)
            // The server expects this same format and will resolve it relative to ROOT_DIR
            let imagePath = LABELGUARD_DATA.image_path;
            
            // For Ray Serve: add /labelguard prefix (route_prefix)
            // For Live Preview testing: keep as is (won't work, but that's OK - Analyze needs Ray Serve)
            if (!USE_RELATIVE_PATHS) {
                // Ray Serve: convert /artifacts/... to /labelguard/artifacts/...
                if (imagePath.startsWith('/artifacts/')) {
                    imagePath = '/labelguard' + imagePath;
                } else if (!imagePath.startsWith('/labelguard/artifacts/') && !imagePath.startsWith('artifacts/')) {
                    // Try to construct the artifacts path with route prefix
                    const originalFilename = LABELGUARD_DATA.labelProcessingResult.original_filename;
                    if (originalFilename) {
                        imagePath = `/labelguard/artifacts/labelguard/${originalFilename}.jpg`;
                    }
                }
            }
            // Note: When using relative paths (Live Preview), Analyze button still requires Ray Serve.
            // Here we construct the JSON payload expected by the Python analyze() function.
            
            const lp = LABELGUARD_DATA.labelProcessingResult;
            const originalFilename = lp.original_filename;
            const kmat = lp.kmat || 'UNKNOWN';
            const version = lp.version || 'v1.0';
            
            // Only send modified blocks back to the server.
            // Empty list = initial detection; non-empty = corrections for these bboxes.
            const requestBlocks = blocks.filter(b => b.modified);
            
            const requestData = {
                image_path: imagePath,
                blocks: requestBlocks,
                kmat: kmat,
                version: version
            };
            
            // Derive explicit etalon_path so backend can load the *_etalon.json file
            if (originalFilename) {
                // Use raw original_filename for filesystem path; no URL encoding here
                requestData.etalon_path = `/artifacts/labelguard/${originalFilename}_etalon.json`;
            }
            
            console.log('Sending analyze request:', requestData);
            console.log('API Server URL:', API_SERVER_URL);
            
            // Send POST request to /labelguard/analyze
            // Note: This requires Ray Serve to be running (won't work with relative paths)
            const analyzeUrl = USE_RELATIVE_PATHS 
                ? 'http://localhost:8000/labelguard/analyze'  // Ray Serve default HTTP port
                : `${API_SERVER_URL}/labelguard/analyze`;
            const response = await fetch(analyzeUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('Analyze response:', result);
            
            if (result.status === 'success' && result.labelProcessingResult) {
                // Update LABELGUARD_DATA with new results
                LABELGUARD_DATA.labelProcessingResult = result.labelProcessingResult;
                
                // Re-render UI
                drawCanvas();
                renderErrorList();
                initializeFiltering();
                
                // Clear selection
                selectedBboxIndex = null;
                categorySelector.style.display = 'none';
                
                console.log('Analysis complete! Updated UI with new results.');
            } else {
                throw new Error(result.message || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('Analyze error:', error);
            const msgDiv = document.getElementById('msgContent');
            msgDiv.innerHTML = `<div style="color: red;">Error during analysis: ${error.message}</div>`;
        } finally {
            // Re-enable button
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze';
        }
    });
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Delete' || e.key === 'Backspace') {
            if (editMode && selectedBboxIndex !== null && e.target.tagName !== 'INPUT') {
                deleteSelectedBbox();
                e.preventDefault();
            }
        }
    });
    
    // Zoom with Ctrl+wheel (disable if resizing)
    canvas.addEventListener('wheel', function(e) {
        if (e.ctrlKey && !isResizing) {
            e.preventDefault();
            const zoom = e.deltaY < 0 ? 1.1 : 0.9;
            const mx = (e.offsetX - offsetX) / scale;
            const my = (e.offsetY - offsetY) / scale;
            
            scale *= zoom;
            offsetX -= mx * (zoom - 1) * scale;
            offsetY -= my * (zoom - 1) * scale;
            drawCanvas();
        }
    });
    
    // Mouse down - handle bbox editing or pan
    canvas.addEventListener('mousedown', function(e) {
        const x = (e.offsetX - offsetX) / scale;
        const y = (e.offsetY - offsetY) / scale;
        const textBlocks = LABELGUARD_DATA.labelProcessingResult.text_blocks;
        
        // Check if clicking on resize handle (only when in edit mode)
        if (editMode && selectedBboxIndex !== null) {
            const selectedBlock = textBlocks.find(b => b.index === selectedBboxIndex);
            if (selectedBlock) {
                const [x1, y1, x2, y2] = selectedBlock.bbox;
                const handle = getHandleAt(x, y, x1, y1, x2, y2);
                
                if (handle) {
                    // Start resizing
                    isResizing = true;
                    resizeHandle = handle;
                    startX = e.clientX;
                    startY = e.clientY;
                    canvas.style.cursor = getCursorForHandle(handle);
                    e.preventDefault();
                    return;
                }
            }
        }
        
        // Check if clicking on existing bbox (to move or select)
        let clickedBlock = null;
        for (const block of textBlocks) {
            const [x1, y1, x2, y2] = block.bbox;
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                clickedBlock = block;
                break;
            }
        }
        
        if (clickedBlock) {
            if (e.shiftKey) {
                // Create mode - don't select
            } else if (editMode && selectedBboxIndex === clickedBlock.index) {
                // In edit mode for this block: start moving it
                isResizing = true;
                resizeHandle = 'move';
                startX = e.clientX;
                startY = e.clientY;
                const [x1, y1] = clickedBlock.bbox;
                createStartX = x - x1; // Offset from bbox corner
                createStartY = y - y1;
                canvas.style.cursor = 'move';
                e.preventDefault();
            } else {
                // Not in edit mode: drag on a block pans the image
                isDragging = true;
                startX = e.clientX - offsetX;
                startY = e.clientY - offsetY;
                canvas.style.cursor = 'grabbing';
                // Selection will be handled by click/double-click
            }
        } else if (e.shiftKey) {
            // Start creating new bbox
            isCreatingBbox = true;
            createStartX = x;
            createStartY = y;
            canvas.style.cursor = 'crosshair';
            e.preventDefault();
        } else {
            // Clicking outside bboxes - prepare for panning
            isDragging = true;
            startX = e.clientX - offsetX;
            startY = e.clientY - offsetY;
            canvas.style.cursor = 'grabbing';
            
            // Deselect and exit edit mode
            selectedBboxIndex = null;
            editMode = false;
            const languageLabel = document.getElementById('languageLabel');
            categorySelector.style.display = 'none';
            if (languageLabel) {
                languageLabel.textContent = '';
            }
            drawCanvas();
        }
    });
    
    // Mouse move - handle resize/move/create or pan
    canvas.addEventListener('mousemove', function(e) {
        // Handle panning (when dragging and not resizing/creating)
        if (isDragging && !isResizing && !isCreatingBbox) {
            offsetX = e.clientX - startX;
            offsetY = e.clientY - startY;
            drawCanvas();
            return;
        }
        
        // Edit mode - handle hover, resize, move, create
        if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) return;
        const x = (e.offsetX - offsetX) / scale;
        const y = (e.offsetY - offsetY) / scale;
        const textBlocks = LABELGUARD_DATA.labelProcessingResult.text_blocks;
        
        if (isResizing && editMode && selectedBboxIndex !== null) {
            // Resize or move existing bbox
            const block = textBlocks.find(b => b.index === selectedBboxIndex);
            if (block) {
                let [x1, y1, x2, y2] = block.bbox;
                const deltaX = (e.clientX - startX) / scale;
                const deltaY = (e.clientY - startY) / scale;
                
                if (resizeHandle === 'move') {
                    // Move entire bbox
                    const newX1 = x - createStartX;
                    const newY1 = y - createStartY;
                    const width = x2 - x1;
                    const height = y2 - y1;
                    block.bbox = [newX1, newY1, newX1 + width, newY1 + height];
                    block.modified = true;
                } else {
                    // Resize based on handle
                    switch (resizeHandle) {
                        case 'nw': x1 += deltaX; y1 += deltaY; break;
                        case 'ne': x2 += deltaX; y1 += deltaY; break;
                        case 'sw': x1 += deltaX; y2 += deltaY; break;
                        case 'se': x2 += deltaX; y2 += deltaY; break;
                        case 'n': y1 += deltaY; break;
                        case 's': y2 += deltaY; break;
                        case 'e': x2 += deltaX; break;
                        case 'w': x1 += deltaX; break;
                    }
                    
                    // Ensure valid bbox (x1 < x2, y1 < y2)
                    if (x1 > x2) { const tmp = x1; x1 = x2; x2 = tmp; }
                    if (y1 > y2) { const tmp = y1; y1 = y2; y2 = tmp; }
                    
                    block.bbox = [x1, y1, x2, y2];
                    block.modified = true;
                }
                
                startX = e.clientX;
                startY = e.clientY;
                drawCanvas();
            }
        } else if (isCreatingBbox) {
            // Drawing new bbox
            drawCanvas();
            // Draw preview rectangle
            ctx.save();
            ctx.translate(offsetX, offsetY);
            ctx.scale(scale, scale);
            ctx.strokeStyle = '#0066ff';
            ctx.lineWidth = 3;
            ctx.setLineDash([5, 5]);
            const width = x - createStartX;
            const height = y - createStartY;
            ctx.strokeRect(createStartX, createStartY, width, height);
            ctx.restore();
        } else {
            // Hover detection
            let foundHover = false;
            hoveredBboxIndex = null;
            
            if (editMode && selectedBboxIndex !== null) {
                const selectedBlock = textBlocks.find(b => b.index === selectedBboxIndex);
                if (selectedBlock) {
                    const [x1, y1, x2, y2] = selectedBlock.bbox;
                    const handle = getHandleAt(x, y, x1, y1, x2, y2);
                    if (handle) {
                        canvas.style.cursor = getCursorForHandle(handle);
                        foundHover = true;
                    }
                }
            }
            
            if (!foundHover) {
                for (const block of textBlocks) {
                    const [x1, y1, x2, y2] = block.bbox;
                    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                        hoveredBboxIndex = block.index;
                        // When not in edit mode, dragging a block pans the image,
                        // so indicate that with a grab cursor. In edit mode, show move.
                        if (editMode && selectedBboxIndex === block.index) {
                            canvas.style.cursor = 'move';
                        } else {
                            canvas.style.cursor = 'grab';
                        }
                        foundHover = true;
                        break;
                    }
                }
            }
            
            if (!foundHover) {
                canvas.style.cursor = e.shiftKey ? 'crosshair' : 'grab';
            }
            
            drawCanvas();
        }
    });
    
    // Mouse up
    canvas.addEventListener('mouseup', function(e) {
        // Edit mode - handle all interactions
        if (isCreatingBbox) {
            const x = (e.offsetX - offsetX) / scale;
            const y = (e.offsetY - offsetY) / scale;
            const width = Math.abs(x - createStartX);
            const height = Math.abs(y - createStartY);
            
            if (width > 10 && height > 10) { // Minimum size
                // Create new bbox
                const textBlocks = LABELGUARD_DATA.labelProcessingResult.text_blocks;
                const newIndex = textBlocks.length > 0 
                    ? String(Math.max(...textBlocks.map(b => parseInt(b.index) || 0)) + 1)
                    : '0';
                
                const x1 = Math.min(createStartX, x);
                const y1 = Math.min(createStartY, y);
                const x2 = Math.max(createStartX, x);
                const y2 = Math.max(createStartY, y);
                
                const newBlock = {
                    index: newIndex,
                    bbox: [x1, y1, x2, y2],
                    text: '',
                    type: '',  // Empty initially, user will select category
                    modified: true,
                    languages: [],
                    sentences: [],
                    etalon_text: null
                };
                
                textBlocks.push(newBlock);
                selectedBboxIndex = newIndex;
                categorySelector.style.display = 'block';
                categorySelect.value = '';  // Empty for new block
            }
            
            isCreatingBbox = false;
            canvas.style.cursor = 'crosshair';
            drawCanvas();
        }
        
        if (isResizing) {
            isResizing = false;
            resizeHandle = null;
            canvas.style.cursor = 'default';
            // When finishing a resize/move, automatically exit edit mode
            if (editMode) {
                editMode = false;
                drawCanvas();
            }
        }
        
        // Stop panning
        if (isDragging) {
            isDragging = false;
            canvas.style.cursor = 'default';
        }
    });
    
    canvas.addEventListener('mouseleave', function() {
        isDragging = false;
        isCreatingBbox = false;
        isResizing = false;
        editMode = false;
        hoveredBboxIndex = null;
        canvas.style.cursor = 'default';
        drawCanvas();
    });
    
    // Click detection (for viewing details - only when not editing)
    canvas.addEventListener('click', function(e) {
        // If we just finished dragging/resizing, don't process click
        if (isDragging || isResizing || isCreatingBbox) {
            return;
        }
        
        const x = (e.offsetX - offsetX) / scale;
        const y = (e.offsetY - offsetY) / scale;
        
        // Find clicked bbox
        if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) return;
        const textBlocks = LABELGUARD_DATA.labelProcessingResult.text_blocks;
        let clickedBlock = null;
        
        for (const block of textBlocks) {
            const [x1, y1, x2, y2] = block.bbox;
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                clickedBlock = block;
                break;
            }
        }
        
        if (clickedBlock) {
            // Select block (without entering edit mode) and show category selector
            selectedBboxIndex = clickedBlock.index;
            categorySelector.style.display = 'block';
            const blockCategoryCode = clickedBlock.type || '';
            if (blockCategoryCode && CATEGORIES[blockCategoryCode]) {
                categorySelect.value = blockCategoryCode;
            } else {
                categorySelect.value = '';
            }
            drawCanvas();
            // Show block details
            renderBlockDetails(clickedBlock);
        } else {
            // Show error list
            renderErrorList();
        }
    });

    // Double-click toggles edit mode for the block under cursor
    canvas.addEventListener('dblclick', function(e) {
        if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) return;
        const x = (e.offsetX - offsetX) / scale;
        const y = (e.offsetY - offsetY) / scale;
        const textBlocks = LABELGUARD_DATA.labelProcessingResult.text_blocks;
        let clickedBlock = null;
        
        for (const block of textBlocks) {
            const [x1, y1, x2, y2] = block.bbox;
            if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
                clickedBlock = block;
                break;
            }
        }
        
        if (clickedBlock) {
            // Enter or switch edit mode to this block
            if (!editMode || selectedBboxIndex !== clickedBlock.index) {
                selectedBboxIndex = clickedBlock.index;
                editMode = true;
                // Show and sync category selector
                categorySelector.style.display = 'block';
                const blockCategoryCode = clickedBlock.type || '';
                if (blockCategoryCode && CATEGORIES[blockCategoryCode]) {
                    categorySelect.value = blockCategoryCode;
                } else {
                    categorySelect.value = '';
                }
            } else {
                // Already editing this block -> exit edit mode
                editMode = false;
                isResizing = false;
                resizeHandle = null;
            }
            drawCanvas();
        } else {
            // Double-click on empty area: exit edit mode and clear selection
            editMode = false;
            selectedBboxIndex = null;
            isResizing = false;
            resizeHandle = null;
            categorySelector.style.display = 'none';
            drawCanvas();
        }
    });

    // Toggle collapsible sections in details panel
    const msgDiv = document.getElementById('msgContent');
    msgDiv.addEventListener('click', function(e) {
        const header = e.target.closest('.lg-collapsible-header');
        if (!header) return;
        const container = header.parentElement;
        if (!container.classList.contains('lg-collapsible')) return;
        container.classList.toggle('lg-collapsible--open');
        const icon = header.querySelector('.lg-collapsible-icon');
        if (icon) {
            icon.textContent = container.classList.contains('lg-collapsible--open') ? '▾' : '▸';
        }
    });
}

// Get resize handle at position (x, y)
function getHandleAt(x, y, x1, y1, x2, y2) {
    const handleSize = HANDLE_SIZE / scale;
    const halfHandle = handleSize / 2;
    
    const handles = [
        { x: x1, y: y1, type: 'nw' },
        { x: x2, y: y1, type: 'ne' },
        { x: x1, y: y2, type: 'sw' },
        { x: x2, y: y2, type: 'se' },
        { x: (x1 + x2) / 2, y: y1, type: 'n' },
        { x: (x1 + x2) / 2, y: y2, type: 's' },
        { x: x1, y: (y1 + y2) / 2, type: 'w' },
        { x: x2, y: (y1 + y2) / 2, type: 'e' }
    ];
    
    for (const handle of handles) {
        if (x >= handle.x - halfHandle && x <= handle.x + halfHandle &&
            y >= handle.y - halfHandle && y <= handle.y + halfHandle) {
            return handle.type;
        }
    }
    return null;
}

// Get cursor style for resize handle
function getCursorForHandle(handle) {
    const cursors = {
        'nw': 'nw-resize',
        'ne': 'ne-resize',
        'sw': 'sw-resize',
        'se': 'se-resize',
        'n': 'n-resize',
        's': 's-resize',
        'e': 'e-resize',
        'w': 'w-resize',
        'move': 'move'
    };
    return cursors[handle] || 'default';
}

// Render block details
function renderBlockDetails(textBlock) {
    if (!LABELGUARD_DATA || !LABELGUARD_DATA.labelProcessingResult) return;
    const msgDiv = document.getElementById('msgContent');
    const filterPanel = document.getElementById('filterPanel');
    const languageLabel = document.getElementById('languageLabel');
    
    // In block details mode, hide the filter panel
    if (filterPanel) {
        filterPanel.style.display = 'none';
    }
    const ruleResults = LABELGUARD_DATA.labelProcessingResult.rule_check_results;
    const blockResults = ruleResults.filter(r => 
        r.text_block && r.text_block.index === textBlock.index
    );
    
    const languages = Array.isArray(textBlock.languages)
        ? textBlock.languages
        : (textBlock.languages ? [textBlock.languages] : []);
    
    // Update language marker next to the category dropdown: "(lang) #index"
    if (languageLabel) {
        if (languages.length) {
            languageLabel.textContent = `(${languages.join(', ')})  #${textBlock.index}`;
        } else {
            languageLabel.textContent = `#${textBlock.index}`;
        }
    }
    
    // Details content (no separate block header – category, lang and index are in the selector row)
    let html = '';
    
    // Full extracted text, no caption
    if (textBlock.text) {
        html += `<div class="lg-section">${textBlock.text}</div>`;
    }
    
    // Full etalon text if available, wrapped in a collapsible section (collapsed by default)
    if (textBlock.etalon_text) {
        html += `
        <div class="lg-section lg-collapsible" data-collapsible="etalon">
          <div class="lg-collapsible-header">
            <span class="lg-collapsible-icon">▸</span>
            <span class="lg-collapsible-title">Еталон</span>
          </div>
          <div class="lg-collapsible-body">${textBlock.etalon_text}</div>
        </div>`;
    }
    
    // Text structure: group sentences by category and join their texts, in a collapsible section
    if (textBlock.sentences && textBlock.sentences.length > 0) {
        const grouped = [];
        const indexByCategory = {};
        
        textBlock.sentences.forEach(sentence => {
            const cat = sentence.category || 'N/A';
            if (indexByCategory[cat] === undefined) {
                indexByCategory[cat] = grouped.length;
                grouped.push({ category: cat, text: sentence.text || '' });
            } else {
                const entry = grouped[indexByCategory[cat]];
                // Join with a space to keep text readable
                entry.text = entry.text ? `${entry.text} ${sentence.text || ''}` : (sentence.text || '');
            }
        });
        
        html += `
        <div class="lg-section lg-collapsible" data-collapsible="structure">
          <div class="lg-collapsible-header">
            <span class="lg-collapsible-icon">▸</span>
            <span class="lg-collapsible-title">Структура текста</span>
          </div>
          <div class="lg-collapsible-body">
            <ul class="lg-list">`;
        grouped.forEach(entry => {
            const catInfo = SENTENCE_CATEGORIES[entry.category];
            const catLabel = catInfo && catInfo.name ? catInfo.name : entry.category;
            html += `<li><strong>[${catLabel}]</strong> ${entry.text}</li>`;
        });
        html += `</ul>
          </div>
        </div>`;
    }
    
    // Rule-level details for this block
    blockResults.forEach(result => {
        if (!result.html_details) {
            return;
        }
        let detailsHtml = result.html_details;
        // In block details view, keep etalon rule but strip the long etalon text chunk
        if (result.rule_name === 'etalon') {
            const marker = 'Еталон:';
            const idx = detailsHtml.indexOf(marker);
            if (idx !== -1) {
                detailsHtml = detailsHtml.slice(0, idx).trim();
            }
        }
        html += `<div class="rule-error lg-rule-error" data-rule-${result.rule_name}>${detailsHtml}</div>`;
    });
    
    msgDiv.innerHTML = html;
}
