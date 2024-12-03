from dataclasses import dataclass
from typing import List
from PIL import Image
from abc import ABC, abstractmethod
import supervision as sv


class AbstractPeanutesDetector(ABC):
    @abstractmethod
    def detect(images: List[Image.Image]) -> List[sv.Detections]:
        pass
