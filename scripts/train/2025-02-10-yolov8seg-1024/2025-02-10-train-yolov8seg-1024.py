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
    model = YOLO("yolov8n-seg.pt")

    input_folder = ROOT_DIR / "assets" / "datasets" / "peanut" / "for-training"

    model.train(
        data=input_folder / "dataset.yaml",
        epochs=20,
        imgsz=640,
        batch=4,
        device=0 if torch.cuda.is_available() else "cpu",
        cache=True,
        workers=4,
        show_labels=False,  # Disable label visualization
        plots=False,  # Keep if you still want other plots
        single_cls=True,

       # augmentation params
        hsv_h=0,
        hsv_s=0,
        hsv_v=0.1,
        scale=0,
        degrees=10.0,
        translate=0.1,
        flipud=0.5,
        erasing=0.0,
        auto_augment=None,
    )