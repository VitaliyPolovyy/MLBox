from abc import ABC, abstractmethod
from typing import List

import supervision as sv
from PIL import Image


class AbstractPeanutsClassifier(ABC):
    @abstractmethod
    def classify(self, images: List[Image.Image]) -> List[sv.Classifications]:
        pass
