from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import supervision as sv
from PIL import Image


class AbstractPeanutsDetector(ABC):
    @abstractmethod
    def detect(images: List[Image.Image]) -> List[sv.Detections]:
        pass
