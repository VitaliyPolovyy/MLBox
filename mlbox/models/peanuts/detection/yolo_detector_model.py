from typing import List

import cv2
import numpy as np
import supervision as sv
import ultralytics.utils.instance
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image

from mlbox.models.peanuts.detection.abstract_detection_model import \
    AbstractPeanutsDetector


class YOLOPeanutsDetector(AbstractPeanutsDetector):
    def __init__(self, weights_path: str, *inference_args, **inference_kwargs):
        super().__init__()
        self.yolo = YOLO(weights_path)
        self.inference_args = inference_args
        self.inference_kwargs = inference_kwargs
        

    def detect(self, images: List[np.ndarray], *args, **kwargs) -> List[sv.Detections]:
        inference_args = self.inference_args
        inference_kwargs = self.inference_kwargs

        if args:
            inference_args = args

        if kwargs:
            inference_kwargs = kwargs

        # Get original image sizes
        original_sizes = [image.shape[:2] for image in images]  # (height, width)

        # Perform detection
        yolo_detections = self.yolo.predict(images, *inference_args, **inference_kwargs)
        
        # Scale detections to original image sizes
        sv_detections = []
        for det in yolo_detections:
            sv_detection = sv.Detections.from_ultralytics(det)
            sv_detection.xyxy = sv_detection.xyxy.astype(int)
            sv_detections.append(sv_detection)

        return sv_detections

