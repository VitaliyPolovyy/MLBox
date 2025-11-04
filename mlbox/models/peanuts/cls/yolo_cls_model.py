from typing import List
import numpy as np
import supervision as sv
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.ops import scale_image

from mlbox.models.peanuts.cls.abstract_cls_model import AbstractPeanutsClassifier


class YOLOPeanutsClassifier(AbstractPeanutsClassifier):
    def __init__(self, weights_path: str, *inference_args, **inference_kwargs):
        super().__init__()
        self.yolo = YOLO(weights_path)
        self.inference_args = inference_args
        self.inference_kwargs = inference_kwargs
        

    def classify(self, images: List[np.ndarray], *args, **kwargs) -> List[sv.Classifications]:
        inference_args = self.inference_args
        inference_kwargs = self.inference_kwargs

        if args:
            inference_args = args

        if kwargs:
            inference_kwargs = kwargs

        # Perform detection
        yolo_predictions = self.yolo.predict(images, *inference_args, **inference_kwargs)
        
        # Scale detections to original image sizes
        classifications  = []
        for pred in yolo_predictions:
            classification = sv.Classifications.from_ultralytics(pred)
            classifications.append(classification)

        return classifications

    class_names = {
        0: 'peanut',
        1: 'skinless_peanut',
        2: 'peanut_half',
        # Add more class names as needed
    }
