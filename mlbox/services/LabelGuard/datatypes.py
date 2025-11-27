"""
Data type definitions for the LabelGuard service.

This module contains all dataclasses and enums used throughout the label validation system.
No business logic - pure data structures only.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, TYPE_CHECKING
from enum import Enum

# Import dependencies from other modules
from mlbox.services.LabelGuard.ocr_processor import OCRWord
from mlbox.utils.lcs import Match

# Only import PIL for type checking, not at runtime
if TYPE_CHECKING:
    from PIL import Image


# ============================================================================
# ENUMS
# ============================================================================

class RulesName(Enum):
    """Names of validation rules that can be checked"""
    ETALON_MATCHING = "etalon"  # text not found in etalon
    ALLERGENS = "allergens"  # count of allergen doesn't match
    NUMBERS_IN_INGRIDIENTS = "numbers"  # count of numbers doesn't match


# ============================================================================
# INPUT DATA STRUCTURES
# ============================================================================

@dataclass
class LabelInput:
    """Input data for label processing pipeline"""
    kmat: str
    version: str
    label_image: "Image.Image"
    label_image_path: str


# ============================================================================
# TEXT PROCESSING DATA STRUCTURES
# ============================================================================

@dataclass
class Sentence:
    """A sentence extracted from OCR results"""
    text: str
    category: str  # enumerates: ingredients, product_name, allergen_phrase, contact_info, other
    words: List[OCRWord]
    index: int = 0


@dataclass
class TextBlock:
    """A text block detected on the label image"""
    bbox: tuple
    sentences: List[Sentence]
    text: str
    type: str  # enumerates: ingredients, nutrition, manufacturing_date, other
    allergens: List[OCRWord]  # list of allergens if type = ingredients
    languages: str
    index:Optional[str] = None
    lcs_results: Optional[List[Match]] = None
    etalon_text: Optional[str] = None
    modified: bool = False  # Track if user has edited this block

# ============================================================================
# VALIDATION RESULT DATA STRUCTURES
# ============================================================================

@dataclass
class VisualMarker:
    """Visual marker for drawing on images"""
    type: Literal["outline", "highlight"]  # outline = border, highlight = filled
    bbox: tuple  # (x1, y1, x2, y2)
    color: tuple  # (R, G, B)
    opacity: Optional[float] = None  # For highlight type (0.0 - 1.0)
    width: Optional[int] = None  # For outline type (pixels)


@dataclass
class CategoryNumberResult:
    """Numbers for a specific sentence category"""
    category: str  # sentence category code (e.g., 'D' for ingredients, 'C' for storage)
    actual_numbers: List[str] = field(default_factory=list)      # Numbers found in THIS block's sentences
    reference_numbers: List[str] = field(default_factory=list)   # Reference set from ALL blocks


@dataclass
class RuleCheckResult:
    """Result of checking a single validation rule"""
    rule_name: RulesName  # Name of the validation rule being checked
    scope: Literal["block", "label"]  # Applied to block or entire label
    
    # Validation result
    passed: bool = True
    score: float = 100.0  # 0-100
    threshold: Optional[float] = None  # Expected threshold (e.g., 90.0)
    score_expression: Optional[str] = None  # Human-readable score calculation (e.g., "100 - 5 / 50 * 100")
    
    # Link to text block
    text_block: Optional[TextBlock] = None  # Link to the TextBlock where check occurred

    # Evidence data - visualization-agnostic
    affected_words: List[OCRWord] = field(default_factory=list)  # Words that caused the rule to fail/pass
    visual_markers: List[VisualMarker] = field(default_factory=list)  # Visual indicators for rendering
    
    # Additional data for rendering (optional, rule-specific)
    metadata: dict = field(default_factory=dict)

    html_details : str = ""


@dataclass
class NumbersCheckResult(RuleCheckResult):
    """Result for numbers validation across ingredient-related sentence categories"""
    category_results: List[CategoryNumberResult] = field(default_factory=list)


@dataclass
class LabelProcessingResult:
    """Complete result of label processing and validation"""
    text_blocks: List[TextBlock] = field(default_factory=list)
    html_report: str = ""
    rule_check_results: List[RuleCheckResult] = field(default_factory=list)  # Validation results
    # None if success
    kmat: Optional[str] = None
    version: Optional[str] = None
    original_filename: Optional[str] = None
    success: bool = True
    
    def to_json(self) -> dict:
        """
        Convert LabelProcessingResult to JSON-serializable dict for JS frontend.
        Only includes fields needed by the frontend UI.
        
        Returns:
            Dict that can be serialized to JSON
        """
        def serialize_rule_check_result(rule_result: RuleCheckResult) -> dict:
            """Serialize RuleCheckResult - only fields needed by JS"""
            # Serialize metadata if present (convert any non-serializable objects to strings)
            serialized_metadata = {}
            if rule_result.metadata:
                for key, value in rule_result.metadata.items():
                    # Skip non-serializable objects, convert to string if needed
                    try:
                        import json
                        json.dumps(value)  # Test if serializable
                        serialized_metadata[key] = value
                    except (TypeError, ValueError):
                        # If not serializable, convert to string representation
                        serialized_metadata[key] = str(value)
            
            return {
                "rule_name": rule_result.rule_name.value if isinstance(rule_result.rule_name, RulesName) else rule_result.rule_name,
                "passed": bool(rule_result.passed),  # Convert numpy bool_ to Python bool
                "text_block": {"index": rule_result.text_block.index} if rule_result.text_block else None,
                "visual_markers": [
                    {
                        "type": marker.type,
                        "bbox": list(marker.bbox),
                        "color": list(marker.color),
                        "opacity": float(marker.opacity) if marker.opacity is not None else None
                    }
                    for marker in rule_result.visual_markers
                ],
                "html_details": rule_result.html_details,
                "metadata": serialized_metadata  # Include metadata if present
            }
        
        def serialize_text_block(block: TextBlock) -> dict:
            """Serialize TextBlock - only fields needed by JS"""
            return {
                "index": block.index,
                "bbox": list(block.bbox),
                "text": block.text,
                "etalon_text": block.etalon_text,
                "type": block.type,
                "modified": bool(block.modified),  # Convert numpy bool_ to Python bool
                "languages": block.languages if isinstance(block.languages, list) else [block.languages] if block.languages else [],
                "sentences": [
                    {
                        "index": int(sentence.index) if sentence.index is not None else 0,  # Convert numpy int types
                        "text": sentence.text,
                        "category": sentence.category
                    }
                    for sentence in block.sentences
                ]
            }
        
        return {
            "kmat": self.kmat,
            "version": self.version,
            "original_filename": self.original_filename,
            "text_blocks": [serialize_text_block(block) for block in self.text_blocks],
            "rule_check_results": [serialize_rule_check_result(rule) for rule in self.rule_check_results]
        }


@dataclass
class ValidationArtifacts:
    """Artifacts generated from validation report"""
    html_report: str
    html_filename: str
    images: List[tuple[str, "Image.Image"]] = field(default_factory=list)  # [(filename, image), ...]

