import os
import random
import shutil

import ssl
from pathlib import Path
import dotenv
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO  # Import YOLO for model loading and prediction
from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent
os.environ["CURL_CA_BUNDLE"] = ""


def create_dataset_split(input_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    image_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    random.shuffle(image_files)

    train_count = int(len(image_files) * train_ratio)
    val_count = int(len(image_files) * val_ratio)

    splits = {
        'train': image_files[:train_count],
        'val': image_files[train_count:train_count + val_count],
        'test': image_files[train_count + val_count:]
    }

    for split_name in ['train', 'val', 'test']:
        images_path = os.path.join(output_dir, split_name, 'images')
        labels_path = os.path.join(output_dir, split_name, 'labels')
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)

    for split_name, split_files in splits.items():
        for file in split_files:
            class_id = file.split('_')[0]
            src_image_path = os.path.join(input_dir, file)
            dst_image_path = os.path.join(output_dir, split_name, 'images', file)
            shutil.copy(src_image_path, dst_image_path)

            label_file_path = os.path.join(output_dir, split_name, 'labels', file.replace('.jpg', '.txt'))
            with open(label_file_path, 'w') as label_file:
                label_file.write(f"{class_id} 0.5 0.5 1 1\n")

if __name__ == "__main__":

    image_folder = ROOT_DIR / "assets" / "DataSet" / "peanuts_class"
    result_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "output"


    create_dataset_split(image_folder, result_folder)
