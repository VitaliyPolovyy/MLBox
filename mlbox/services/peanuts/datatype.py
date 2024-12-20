from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import supervision as sv
from cv2.typing import RotatedRect
from PIL import Image as PILImage


# Request Dataclass
@dataclass
class PeanutProcessingRequest:
    image: PILImage
    alias: str
    key: str
    service_code: str
    client_host: Optional[str] = None
    username: Optional[str] = None
    request_time: Optional[str] = None


@dataclass
class Ellipse:
    center: Tuple[float, float]  # (x, y)
    axes: Tuple[float, float]  # (minor_axis, major_axis)
    angle: float  # Rotation angle


# Peanut-Level Result Dataclass
@dataclass
class OnePeanutProcessingResult:
    index: int
    xyxy: Tuple[int, int, int, int]  # Bbox coordinates (x1, y1, x2, y2)
    mask: Optional[np.ndarray]
    det_confidence: Optional[float]  # Confidence score for the detection
    class_id: Optional[int] = None  # Class ID for the classification
    class_confidence: Optional[float] = None  # Confidence score for the classification
    image: Optional[PILImage.Image] = None  # Image of the peanut
    contour: Optional[np.ndarray] = None
    ellipse: Optional[Ellipse] = None
    rotated_bbox: Optional[RotatedRect] = None


# Image-Level Result Dataclass
@dataclass
class PeanutProcessingResult:
    peanuts: List[OnePeanutProcessingResult]  # List of peanuts detected in the image
    accuracy: Optional[float] = None  # General accuracy of the processing result
    weight: Optional[float] = None
    pixels_per_mm: Optional[float] = None
