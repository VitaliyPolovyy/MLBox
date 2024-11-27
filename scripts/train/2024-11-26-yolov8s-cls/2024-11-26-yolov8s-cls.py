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

    model = YOLO( ROOT_DIR / "assets" / "models" / "Yolo" / "yolov8s-cls.pt")


    input_folder = ROOT_DIR / "assets" / "DataSet" / "peanuts_class"
    yaml_file = input_folder / "dataset.yaml"
    #yaml_file = Path (r"/mnt/c/My storage/Python projects/MLBox/assets/DataSet/peanuts_class/dataset.yaml").resolve()
    yaml_file = Path (r"dataset.yaml").resolve()

    print(str (yaml_file.resolve()),)

    """
    model.train(
        data=str(yaml_file.resolve()),
        epochs=2,
        imgsz=128,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=8,
        task="classify",
    )
    """

    

    
    # Train the model
    #results = model.train(data="mnist160", epochs=100, imgsz=64)