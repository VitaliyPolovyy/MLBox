import os
import random
import shutil
import cv2
from pathlib import Path
from PIL import Image as PILImage, Image
import numpy as np
import torch
import yaml
from ultralytics import YOLO



from mlbox.settings import ROOT_DIR
import json


CURRENT_DIR = Path(__file__).parent

def split_dataset(input_folder: Path, output_folder: Path, coco_annotation, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1) -> None:
    def copy_images_and_annotations(images, subset):
        subset_dir = output_folder / subset / "images"
        subset_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir = output_folder / subset / "labels"
        annotations_dir.mkdir(parents=True, exist_ok=True)

        for img in images:
            img_id = img['id']
            img_filename = img['file_name']
            shutil.copy(input_folder / img_filename, subset_dir / img_filename)

            img_annotations = [ann for ann in annotations if ann['image_id'] == img_id]
            yolo_annotations = []
            for ann in img_annotations:
                bbox = ann['bbox']
                x_center = bbox[0] + bbox[2] / 2
                y_center = bbox[1] + bbox[3] / 2
                width = bbox[2]
                height = bbox[3]
                category_id = ann['category_id']
                yolo_annotations.append(f"{category_id} {x_center} {y_center} {width} {height}")

            annotation_filename = Path(img_filename).stem + '.txt'
            with open(annotations_dir / annotation_filename, 'w') as f:
                f.write("\n".join(yolo_annotations))

    with open(coco_annotation, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']

    random.shuffle(images)

    num_images = len(images)
    train_end = int(train_ratio * num_images)
    val_end = train_end + int(val_ratio * num_images)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    copy_images_and_annotations(train_images, 'train')
    copy_images_and_annotations(val_images, 'val')
    copy_images_and_annotations(test_images, 'test')
    

def predict_and_save(input_dir, output_dir, yolo_path, annotation_folder):
    model = YOLO(yolo_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    annotation_folder = Path(annotation_folder)
    output_dir.mkdir(parents=True, exist_ok=True)

    iou_scores = []

    for img_path in input_dir.glob("*.jpg"):  # Assuming images are in .jpg format
        img = np.array(Image.open(img_path))
        results = model.predict(img, verbose=False)
        results = results[0] if results else None

        mask = np.zeros_like(img, dtype=np.uint8)
        true_mask = np.zeros_like(img, dtype=np.uint8)

        # Load the true mask
        annotation_path = annotation_folder / (img_path.stem + '.txt')
        
        with open(annotation_path, 'r') as f:
            for line in f:
                coords = list(map(float, line.strip().split()[1:]))
                points = np.array(coords).reshape(-1, 2)
                points[:, 0] *= img.shape[1]  # Scale x coordinates
                points[:, 1] *= img.shape[0]  # Scale y coordinates
                points = points.astype(np.int32)
                #cv2.drawContours(img, [points], -1, (0, 255, 0), 2)  # Draw the contour on the true mask

        cv2.fillPoly(true_mask, [points], (255, 255, 255))  # Fill the contour on the mask

        if results and results.masks.xy:
            points = results.masks.xy[0].astype(np.int32)
            cv2.drawContours(img, [points], -1, (255, 0, 0), 2)  # Draw the contour with white color and fill it
            cv2.fillPoly(mask, [points], (255, 255, 255))  # Fill the contour on the mask
        else:
            print("nothing is predicted!")

        output_path = output_dir / (img_path.stem + "_mask.jpg")
        Image.fromarray(img).save(output_path)


def train_yolo(input_folder : Path):
    model = YOLO("yolov8s.pt")
    
    model.train(
        data=input_folder / "dataset.yaml",
        project=CURRENT_DIR,
        name="experiment",      # default is 'train'
        epochs=30,               # default is 100
        imgsz=128,              # default is 640
        device=0 if torch.cuda.is_available() else "cpu",
        cache=True,             # default is False
        workers=4,              # default is 8
        single_cls=True,        # default is False
        mask_ratio=1,
        
        # Changed from defaults augmentation params
        hsv_h=0,               # default is 0.015
        hsv_s=0,               # default is 0.7
        hsv_v=0.1,             # default is 0.4
        overlap_mask = False,
        flipud=0.5,            # default is 0.0

        scale=0,
        shear=0,
        perspective=0,
        degrees=5.0,
        
        # Training specifics that differ from defaults
        close_mosaic=10,       # default is 0
        patience=100,          # default is 50
        seed=0,                # default is None
        deterministic=True     # default is False
    )


if __name__ == "__main__":

    input_folder = ROOT_DIR / "assets" / "datasets" / "LabelDetect_PM"
    output_folder = ROOT_DIR / "assets" / "datasets" / "LabelDetect_PM" / "for-training"
    coco_annotation = input_folder / "instances_default.json"

    #split_dataset(input_folder, output_folder, coco_annotation)

    # model = YOLO( ROOT_DIR / "assets" / "models" / "Yolo" / "yolov8s-cls.pt")
    
    
    #train_yolo(input_folder)
    
    # Example usage

    model_file = CURRENT_DIR / "experiment16" / "weights" / "best.pt"

    input_folder = ROOT_DIR / "assets" / "datasets" / "peanut" / "separated" / "for-training" / "test" /"images"

    #input_folder = ROOT_DIR / "tmp" / "2025-02-14-train-separated" / "input"
    output_folder = ROOT_DIR / "tmp" / "2025-02-14-train-separated" / "output"
    annotation_folder = ROOT_DIR / "assets" / "datasets" / "peanut" / "separated" / "for-training" / "test" /"labels"
    
    """
    predict_and_save(input_folder, output_folder, model_file, annotation_folder)

    experiment_output_folder = model_file.parent.parent / "images"
    experiment_output_folder.mkdir(parents=True, exist_ok=True)

    for file in output_folder.glob("*"):
        shutil.copy(file, experiment_output_folder)
    """
    