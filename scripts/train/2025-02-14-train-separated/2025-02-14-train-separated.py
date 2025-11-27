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

    iou_data = []  # Store (img_path, mask_info_list) where mask_info_list contains (iou, confidence, area) for each mask

    # Colors in BGR format
    pred_color = (0, 0, 255)  # Red for predicted masks
    gt_color = (0, 255, 255)  # Yellow for ground truth

    for img_path in input_dir.glob("*.jpg"):  # Assuming images are in .jpg format
        img = np.array(Image.open(img_path))
        results = model.predict(img, verbose=False)
        results = results[0] if results else None

        # Convert RGB to BGR for OpenCV drawing
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result_img = img_bgr.copy()
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        true_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Load the true mask - accumulate ALL polygons
        annotation_path = annotation_folder / (img_path.stem + '.txt')
        gt_polygons = []
        
        if annotation_path.exists():
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    coords = list(map(float, parts[1:]))
                    points = np.array(coords).reshape(-1, 2)
                    points[:, 0] *= img.shape[1]  # Scale x coordinates
                    points[:, 1] *= img.shape[0]  # Scale y coordinates
                    points = points.astype(np.int32)
                    gt_polygons.append(points)
        
        # Draw all ground truth polygons in yellow (just contours)
        for points in gt_polygons:
            cv2.drawContours(result_img, [points], -1, gt_color, 2)
            cv2.fillPoly(true_mask, [points], 255)

        # Process all predicted masks
        mask_info_list = []  # Store (iou, confidence, area) for each predicted mask
        
        if results and results.masks is not None and len(results.masks.xy) > 0:
            # Get confidence scores
            confidences = results.boxes.conf.cpu().numpy() if results.boxes is not None else [0.5] * len(results.masks.xy)
            
            # Draw all predicted masks in red (just contours) and calculate IoU for each
            for idx, (mask_xy, conf) in enumerate(zip(results.masks.xy, confidences)):
                points = mask_xy.astype(np.int32)
                
                # Draw contour only (no fill)
                cv2.drawContours(result_img, [points], -1, pred_color, 2)
                
                # Create mask for IoU calculation
                temp_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
                cv2.fillPoly(temp_mask, [points], 255)
                
                # Calculate area for this mask
                mask_area = temp_mask.sum()
                
                # Calculate IoU for this mask
                iou = calculate_iou(temp_mask, true_mask)
                mask_info_list.append((iou, float(conf), mask_area))
        else:
            print(f"nothing is predicted for {img_path.name}!")
        
        # Store all mask information
        iou_data.append((img_path, mask_info_list))
        
        # Convert BGR back to RGB for saving with PIL
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        output_path = output_dir / (img_path.stem + "_mask.jpg")
        Image.fromarray(result_img_rgb).save(output_path)

    # Save IoU scores to a text file with confidence
    iou_file_path = output_dir / "IoU.txt"
    with open(iou_file_path, 'w') as f:
        # Sort by best IoU per image (highest IoU among all masks for that image)
        sorted_iou_data = sorted(
            iou_data, 
            key=lambda x: max((mask_info[0] for mask_info in x[1]), default=0.0), 
            reverse=True
        )
        
        all_ious = []
        all_confs = []
        
        for img_path, mask_info_list in sorted_iou_data:
            if not mask_info_list:
                # No predictions
                f.write(f"{img_path.name}: no predictions\n")
            else:
                # Sort masks by IoU (best first)
                mask_info_list_sorted = sorted(mask_info_list, key=lambda x: x[0], reverse=True)
                
                # Write all masks for this image
                mask_strings = []
                for mask_idx, (iou, conf, area) in enumerate(mask_info_list_sorted):
                    mask_strings.append(f"mask{mask_idx+1}: IoU={iou:.4f} conf={conf:.4f}")
                    all_ious.append(iou)
                    all_confs.append(conf)
                
                f.write(f"{img_path.name}: {' | '.join(mask_strings)}\n")
        
        if all_ious:
            avg_iou = sum(all_ious) / len(all_ious)
            avg_conf = sum(all_confs) / len(all_confs)
            f.write(f"\nAverage IoU (all masks): {avg_iou:.4f}\n")
            f.write(f"Average Confidence (all masks): {avg_conf:.4f}\n")


if __name__ == "__main__":

    
    # model = YOLO( ROOT_DIR / "assets" / "models" / "Yolo" / "yolov8s-cls.pt")
    
    model = YOLO("yolov8n-seg.pt")
    
    input_folder = ROOT_DIR / "assets" / "peanuts" / "datasets" / "separated" / "for-training"
    
    """
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
    """
    
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
    
    