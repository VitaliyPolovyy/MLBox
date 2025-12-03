import shutil
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent

# ==================== Metric Functions ====================

def calculate_iou(pred_mask, true_mask):
    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_dice(pred_mask, true_mask):
    """
    Calculate Dice coefficient.
    
    Args:
        pred_mask: Binary prediction (bool or 0/1)
        true_mask: Ground truth binary mask (bool or 0/1)
    
    Returns:
        Dice score (0.0 to 1.0)
    """
    intersection = np.logical_and(pred_mask, true_mask).sum()
    pred_sum = pred_mask.sum()
    true_sum = true_mask.sum()
    
    if pred_sum + true_sum == 0:
        return 1.0 if intersection == 0 else 0.0
    
    dice = (2.0 * intersection) / (pred_sum + true_sum)
    return float(dice)

# ==================== U-Net Evaluation Function ====================

def predict_and_save_unet(input_dir, output_dir, model_path, masks_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,  # Don't load pretrained, we'll load our weights
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Validation transform
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    masks_dir = Path(masks_dir) if masks_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_data = []
    pred_color = (0, 0, 255)  # Red for predicted masks
    gt_color = (0, 255, 255)  # Yellow for ground truth
    
    for img_path in input_dir.glob("*.jpg"):
        # Load and preprocess image
        img = np.array(Image.open(img_path).convert('RGB'))
        img_original = img.copy()
        h_orig, w_orig = img.shape[:2]

        transformed = transform(image=img)
        img_tensor = transformed['image'].unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            prob_mask = torch.sigmoid(output).cpu().numpy()[0, 0]  # float in [0, 1]

        # Save probability map as grayscale image (resized to original image size)
        prob_mask_resized = cv2.resize(prob_mask, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
        prob_vis = (prob_mask_resized * 255).astype(np.uint8)
        prob_path = output_dir / f"{img_path.stem}_prob.png"
        Image.fromarray(prob_vis).save(prob_path)

        # Binarize for metrics/contours (keep original logic)
        pred_mask = (prob_mask > 0.5).astype(np.uint8) * 255

        # Resize prediction back to original size
        pred_mask_resized = cv2.resize(pred_mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        
        # Load ground truth mask (optional)
        true_mask = None
        if masks_dir is not None:
            mask_path = masks_dir / (img_path.stem + '.png')
            if mask_path.exists():
                true_mask = np.array(Image.open(mask_path).convert('L'))
                true_mask = (true_mask > 127).astype(np.uint8) * 255
        
        # Calculate Dice and IoU only if ground truth is available
        if true_mask is not None:
            pred_binary = (pred_mask_resized > 127).astype(bool)
            true_binary = (true_mask > 127).astype(bool)
            dice = calculate_dice(pred_binary, true_binary)
            iou = calculate_iou(pred_binary, true_binary)
            metrics_data.append((img_path.name, dice, iou))
        
        # Create visualization
        img_bgr = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
        result_img = img_bgr.copy()
        
        # Draw contours
        pred_contours, _ = cv2.findContours(pred_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result_img, pred_contours, -1, pred_color, 2)

        # Draw ground truth contours only if mask is available
        if true_mask is not None:
            gt_contours, _ = cv2.findContours(true_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result_img, gt_contours, -1, gt_color, 2)
        
        # Save visualization
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        output_path = output_dir / (img_path.stem + "_mask.jpg")
        Image.fromarray(result_img_rgb).save(output_path)
    
    # Save Dice and IoU scores only if we had ground truth masks
    if metrics_data:
        iou_file_path = output_dir / "IoU.txt"
        with open(iou_file_path, 'w') as f:
            # Sort by Dice (best first) - Dice is primary metric
            sorted_metrics_data = sorted(metrics_data, key=lambda x: x[1], reverse=True)
            
            all_dice_scores = []
            all_ious = []
            for img_name, dice, iou in sorted_metrics_data:
                f.write(f"{img_name}: Dice={dice:.4f}, IoU={iou:.4f}\n")
                all_dice_scores.append(dice)
                all_ious.append(iou)
            
            if all_dice_scores and all_ious:
                avg_dice = sum(all_dice_scores) / len(all_dice_scores)
                min_dice = min(all_dice_scores)
                avg_iou = sum(all_ious) / len(all_ious)
                min_iou = min(all_ious)
                f.write(f"\nAverage Dice: {avg_dice:.4f}\n")
                f.write(f"Min Dice: {min_dice:.4f}\n")
                f.write(f"Average IoU: {avg_iou:.4f}\n")
                f.write(f"Min IoU: {min_iou:.4f}\n")
        
        return iou_file_path
    
    return None

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
    
    test_images_dir = ROOT_DIR / "artifacts" / "Peanuts" / "temp"
    test_masks_dir = None # dataset_dir / "test" / "masks"
    
    # Experiment folder (set manually)
    experiment_dir = CURRENT_DIR / "experiment1"
    
    # Evaluation on test set
    print("Starting evaluation on test set...")
    model_file = experiment_dir / "weights" / "best.pth"
    output_folder = ROOT_DIR / "tmp" / "2025-02-14-train-separated" / "output"
    
    predict_and_save_unet(test_images_dir, output_folder, model_file, test_masks_dir)
    
    # Copy results to experiment folder
    experiment_output_folder = experiment_dir / "images"
    experiment_output_folder.mkdir(parents=True, exist_ok=True)
    
    for file in output_folder.glob("*"):
        shutil.copy(file, experiment_output_folder)
    
    print(f"\nEvaluation complete!")
    print(f"Results saved to: {experiment_dir}")
    print(f"Model: {model_file}")
    print(f"Evaluation results: {experiment_output_folder / 'IoU.txt'}")

