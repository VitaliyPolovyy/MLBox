import json
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent

if __name__ == "__main__":

    # model = YOLO( ROOT_DIR / "assets" / "models" / "Yolo" / "yolov8s-cls.pt")
    model = YOLO("yolo11n-cls.pt")

    input_folder = ROOT_DIR / "assets" / "DataSet" / "peanuts_class"

    model.train(
        data=str(input_folder.resolve()),
        epochs=100,
        imgsz=128,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(CURRENT_DIR.resolve()),
        workers=8,
        # augmentation params
        hsv_h=0,
        hsv_s=0,
        hsv_v=0.1,
        translate=0,
        scale=0,
        flipud=0.5,
        erasing=0.0,
        auto_augment=None,
    )
