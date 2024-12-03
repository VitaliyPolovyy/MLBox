from mlbox.models.peanuts.detection.abstract_detection_model import AbstractPeanutsDetector, PeanutsDetectionResult
from typing import List
from PIL import Image
from ultralytics import YOLO
import supervision as sv


class YOLOPeanutsDetector(AbstractPeanutsDetector):
    def __init__(self, weights_path: str, *inference_args, **inference_kwargs):
        super().__init__()
        self.yolo = YOLO(weights_path)
        self.inference_args = inference_args
        self.inference_kwargs = inference_kwargs

    def detect(self, images: List[Image.Image], *args, **kwargs) -> List[List[PeanutsDetectionResult]]:
        inference_args = self.inference_args
        inference_kwargs = self.inference_kwargs

        if args:
            inference_args = args

        if kwargs:
            inference_kwargs = kwargs

        detections = self.yolo.predict(images, *inference_args, **inference_kwargs)
        detections = [sv.Detections.from_ultralytics(det) for det in detections]
        return detections
