from dataclasses import dataclass  # Pylint warning: Missing module docstring
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from dataclasses_json import dataclass_json
from PIL import Image as PILImage


@dataclass
class Status(Enum):
    SUCCESS = "success"
    ERROR = "error"


# Request Dataclass
@dataclass
class PeanutProcessingRequest:
    image: PILImage
    alias: str
    key: str
    response_method: str
    response_endpoint: str
    client_host: Optional[str] = None
    username: Optional[str] = None
    request_time: Optional[str] = None
    image_filename: Optional[str] = None


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
    class_id: Optional[np.ndarray] = None
    class_confidence: Optional[np.ndarray] = None
    image: Optional[PILImage.Image] = None  # Image of the peanut
    contour: Optional[np.ndarray] = None
    ellipse: Optional[Ellipse] = None

    @property
    def real_class(self):
        if self.class_confidence is not None and self.class_id is not None:
            max_conf_index = np.argmax(self.class_confidence)
            return self.class_id[max_conf_index], self.class_confidence[max_conf_index]
        return None, None


# Image-Level Result Dataclass
@dataclass
class PeanutProcessingResult:
    def __init__(
        self,
        peanuts: List[OnePeanutProcessingResult],
        accuracy: Optional[float] = None,
        weight_g: Optional[float] = None,
        pixels_per_mm: Optional[float] = None,
        original_image: Optional[PILImage.Image] = None,
        original_image_filename: Optional[str] = None,
        status: Optional[Status] = None,
        message: Optional[str] = None,
    ):
        self.peanuts = peanuts
        self.accuracy = accuracy
        self.weight_g = weight_g
        self.pixels_per_mm = pixels_per_mm
        self.original_image = original_image
        self.original_image_filename = original_image_filename
        self.status = status
        self.message = message
        self.create_result_image()

    peanuts: Optional[List[OnePeanutProcessingResult]] = (
        None  # List of peanuts detected in the image
    )
    accuracy: Optional[float] = None  # General accuracy of the processing result
    weight_g: Optional[float] = None
    pixels_per_mm: Optional[float] = None
    original_image: Optional[PILImage.Image] = None
    original_image_filename: Optional[str] = None
    result_image: Optional[PILImage.Image] = None
    status: Optional[Status] = None
    message: Optional[str] = None
    excel_filename: Optional[Path] = None

    @property
    def standard_deviation_minor_axe(self) -> float:
        minor_axes = [
            peanut.ellipse.axes[0] / self.pixels_per_mm
            for peanut in self.peanuts
            if peanut.ellipse
        ]
        return round(np.std(minor_axes), 2) if minor_axes else None

    @property
    def coefficient_variation_minor_axe(self) -> float:
        minor_axes = [
            peanut.ellipse.axes[0] / self.pixels_per_mm
            for peanut in self.peanuts
            if peanut.ellipse
        ]
        mean = np.mean(minor_axes)
        return (
            round(self.standard_deviation_minor_axe / mean * 100, 2)
            if minor_axes
            else None
        )

    @property
    def standard_deviation_ratio_axes(self) -> float:
        ratios = [
            peanut.ellipse.axes[1] / peanut.ellipse.axes[0]
            for peanut in self.peanuts
            if peanut.ellipse
        ]
        return round(np.std(ratios), 2) if ratios else None

    @property
    def coefficient_variation_ratio_axes(self) -> float:
        ratios = [
            peanut.ellipse.axes[1] / peanut.ellipse.axes[0]
            for peanut in self.peanuts
            if peanut.ellipse
        ]
        mean = np.mean(ratios)
        return (
            round(self.standard_deviation_ratio_axes / mean * 100, 2)
            if ratios
            else None
        )

    def create_result_image(self) -> PILImage.Image:

        result_image = self.original_image.copy()
        cv_image = np.array(result_image)

        for peanut in self.peanuts:

            center = (int(peanut.ellipse.center[0]), int(peanut.ellipse.center[1]))
            axes = (int(peanut.ellipse.axes[0] / 2), int(peanut.ellipse.axes[1] / 2))
            class_id = peanut.real_class[0]

            # Determine the color based on the class_id
            if class_id == 0:
                color = (255, 0, 0)
            elif class_id == 1:
                color = (0, 255, 0)
            elif class_id == 2:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)  # Default color (Blue)

            cv2.ellipse(cv_image, center, axes, peanut.ellipse.angle, 0, 360, color, 2)

            # Put peanut index in the middle of the ellipse
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 2
            text_size = cv2.getTextSize(
                str(peanut.index), font, font_scale, font_thickness
            )[0]
            text_x = int(peanut.ellipse.center[0] - text_size[0] // 2)
            text_y = int(peanut.ellipse.center[1] + text_size[1] // 2)
            cv2.putText(
                cv_image,
                str(peanut.index),
                (text_x, text_y),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness,
            )

            if peanut.mask is not None and False:
                contours, _ = cv2.findContours(peanut.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(cv_image, contours, -1, (0,0,150), 2)

            self.result_image = PILImage.fromarray(cv_image)


@dataclass_json
@dataclass
class BaseResponseJson:
    status: str
    message: str
    service_name: str
    timestamp: datetime
    data: str

@dataclass_json
@dataclass
class PeanutDataResponseJson:
    alias: str
    key: str
    image_filename: str
    excel_file: str


@dataclass_json
@dataclass
class PeanutInputJson:
    alias: str
    key: str
    response_method: str
    response_endpoint: Optional[str] = None
