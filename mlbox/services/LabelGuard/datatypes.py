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
    ETALON_MATCHING = "etalon_matching"  # text not found in etalon
    ALLERGENS = "allergen_error"  # count of allergen doesn't match
    NUMBERS_IN_INGRIDIENTS = "numbers_in_ingridients"  # count of numbers doesn't match


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
    index: int
    text: str
    type: str  # enumerates: ingredients, nutrition, manufacturing_date, other
    allergens: List[OCRWord]  # list of allergens if type = ingredients
    languages: str
    lcs_results: Optional[List[Match]] = None
    etalon_text: Optional[str] = None


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


@dataclass
class ValidationArtifacts:
    """Artifacts generated from validation report"""
    html_report: str
    html_filename: str
    images: List[tuple[str, "Image.Image"]] = field(default_factory=list)  # [(filename, image), ...]

