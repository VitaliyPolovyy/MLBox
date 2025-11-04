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
from ultralytics.models.yolo.segment import SegmentationValidator


from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

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

        iou = calculate_iou(mask[:, :, 0], true_mask[:, :, 0])
        iou_scores.append(iou)
        
        output_path = output_dir / (img_path.stem + "_mask.jpg")
        Image.fromarray(img).save(output_path)

    # Save IoU scores to a text file
    iou_file_path = output_dir / "IoU.txt"
    with open(iou_file_path, 'w') as f:
        sorted_iou_scores = sorted(zip(input_dir.glob("*.jpg"), iou_scores), key=lambda x: x[1], reverse=True)
        for img_path, iou in sorted_iou_scores:
            f.write(f"{img_path.name}: {iou:.4f}\n")
        avg_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        f.write(f"\nAverage IoU: {avg_iou:.4f}\n")


if __name__ == "__main__":

    
    # model = YOLO( ROOT_DIR / "assets" / "models" / "Yolo" / "yolov8s-cls.pt")
    
    model = YOLO("yolov8n-seg.pt")
    
    input_folder = ROOT_DIR / "assets" / "peanuts" / "datasets" / "separated" / "for-training"
    
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
    
    # Example usage

    model_file = CURRENT_DIR / "experiment16" / "weights" / "best.pt"

    input_folder = ROOT_DIR / "assets" / "peanuts" / "datasets" / "separated" / "for-training" / "test" /"images"

    #input_folder = ROOT_DIR / "tmp" / "2025-02-14-train-separated" / "input"
    output_folder = ROOT_DIR / "tmp" / "2025-02-14-train-separated" / "output"
    annotation_folder = ROOT_DIR / "assets" / "peanuts" / "datasets" / "separated" / "for-training" / "test" /"labels"
    
    predict_and_save(input_folder, output_folder, model_file, annotation_folder)

    experiment_output_folder = model_file.parent.parent / "images"
    experiment_output_folder.mkdir(parents=True, exist_ok=True)

    for file in output_folder.glob("*"):
        shutil.copy(file, experiment_output_folder)
    
    