import os
import cv2
import pandas as pd
import numpy as np
import traceback
from pathlib import Path
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from mlbox.models.peanuts.detection.yolo_detector_model import YOLOPeanutsDetector
from mlbox.models.peanuts.detection.unet_detector_model import UNetPeanutsDetector
from mlbox.utils.cvtools import preprocess_images_with_white_rectangle
from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent
DATA_DIR = CURRENT_DIR / "ground-truth-comparison"
OUTPUT_DIR = DATA_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables - same as peanuts.py
# Load from peanuts service directory first (where the HF variables are)
peanuts_env = ROOT_DIR / "mlbox" / "services" / "peanuts" / ".env"
if peanuts_env.exists():
    load_dotenv(peanuts_env, override=False)
    print(f"Loaded .env from: {peanuts_env}")

# Then load from project root and credentials
load_dotenv(ROOT_DIR / ".env.mlbox", override=False)
env_file = Path.home() / "credentials" / ".env.mlbox"
if env_file.exists():
    load_dotenv(env_file, override=False)

HF_TOKEN = os.getenv("HF_TOKEN")
HF_PEANUT_SEG_REPO_ID = os.getenv("HF_PEANUT_SEG_REPO_ID")
HF_PEANUT_SEG_FILE = os.getenv("HF_PEANUT_SEG_FILE")
HF_PEANUT_SEG_SEPARATED_REPO_ID = os.getenv("HF_PEANUT_SEG_SEPARATED_REPO_ID")
HF_PEANUT_SEG_SEPARATED_FILE = os.getenv("HF_PEANUT_SEG_SEPARATED_FILE")

# Debug: print loaded env vars (without values for security)
if not HF_PEANUT_SEG_REPO_ID:
    print("WARNING: HF_PEANUT_SEG_REPO_ID not found in environment")
    print(f"Checked .env files: ROOT_DIR/.env={Path(ROOT_DIR / '.env').exists()}, CURRENT_DIR/.env={Path(CURRENT_DIR / '.env').exists()}, credentials/.env.mlbox={env_file.exists()}")


def load_ground_truth(csv_path: Path) -> Dict[int, Tuple[float, float]]:
    """Load ground truth CSV with semicolon delimiter and comma decimals."""
    df = pd.read_csv(csv_path, sep=';', header=None, names=['peanut_index', 'length_mm', 'width_mm'])
    
    # Convert comma decimals to dots
    df['length_mm'] = df['length_mm'].str.replace(',', '.').astype(float)
    df['width_mm'] = df['width_mm'].str.replace(',', '.').astype(float)
    
    # Create dictionary: index -> (length, width)
    gt_dict = {}
    for _, row in df.iterrows():
        gt_dict[int(row['peanut_index'])] = (row['length_mm'], row['width_mm'])
    
    return gt_dict


# Removed convert_bbox_to_original - we now detect directly on original image


def calculate_ellipse_measurements(
    contour: np.ndarray,
    pixels_per_mm: float
) -> Tuple[float, float]:
    """Calculate length and width from ellipse fitting."""
    if len(contour) < 5:
        return None, None
    
    contour_reshaped = contour.reshape(-1, 2)
    center, axes, angle = cv2.fitEllipse(contour_reshaped)
    
    # Major axis = length, minor axis = width
    major_axis = max(axes)
    minor_axis = min(axes)
    
    length_mm = major_axis / pixels_per_mm
    width_mm = minor_axis / pixels_per_mm
    
    return length_mm, width_mm


def calculate_bbox_measurements(
    contour: np.ndarray,
    pixels_per_mm: float
) -> Tuple[float, float]:
    """Calculate length and width from oriented bounding box."""
    if len(contour) < 3:
        return None, None
    
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]  # rect[1] is (width, height)
    
    # Length = longer side, width = shorter side
    length_px = max(width, height)
    width_px = min(width, height)
    
    length_mm = length_px / pixels_per_mm
    width_mm = width_px / pixels_per_mm
    
    return length_mm, width_mm


def create_visualization(
    original_image: np.ndarray,
    yolo_bboxes: List[Tuple[int, int, int, int]],
    unet_contours: List[Optional[np.ndarray]],
    ellipse_params: List[Optional[Tuple]],
    bbox_params: List[Optional[Tuple]],
    peanut_indices: List[int],
    output_path: Path
):
    """Create visualization with all approaches.
    
    Colors (BGR format for OpenCV):
    - Red: YOLO bbox
    - Green: UNet mask contour
    - Yellow: Oriented bounding box
    - Blue: Ellipse
    """
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    vis_img = img_bgr.copy()
    
    # Draw each peanut
    for yolo_bbox, unet_contour, ellipse_param, bbox_param, peanut_idx in zip(
        yolo_bboxes, unet_contours, ellipse_params, bbox_params, peanut_indices
    ):
        # Red: YOLO bbox
        x1, y1, x2, y2 = yolo_bbox
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Green: UNet mask contour
        if unet_contour is not None and len(unet_contour) > 0:
            cv2.drawContours(vis_img, [unet_contour], -1, (0, 255, 0), 2)
        
        # Yellow: Oriented bounding box
        if bbox_param is not None:
            rect = bbox_param
            box = cv2.boxPoints(rect).astype(int)
            cv2.drawContours(vis_img, [box], 0, (0, 255, 255), 2)
        
        # Blue: Ellipse
        if ellipse_param is not None:
            center, axes, angle = ellipse_param
            center_int = (int(center[0]), int(center[1]))
            axes_int = (int(axes[0] / 2), int(axes[1] / 2))
            cv2.ellipse(vis_img, center_int, axes_int, angle, 0, 360, (255, 0, 0), 2)
        
        # Draw peanut index label
        label = str(peanut_idx)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Position label at top-left of YOLO bbox with background
        label_x = x1
        label_y = y1 - 10 if y1 > 30 else y1 + text_height + 10
        
        # Draw background rectangle for text
        cv2.rectangle(vis_img, 
                     (label_x - 2, label_y - text_height - 2),
                     (label_x + text_width + 2, label_y + baseline + 2),
                     (255, 255, 255), -1)
        
        # Draw text
        cv2.putText(vis_img, label, (label_x, label_y), 
                   font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # Convert BGR back to RGB for saving
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(vis_img_rgb).save(output_path)


def main():
    # Load ground truth
    csv_path = DATA_DIR / "2.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")
    
    gt_dict = load_ground_truth(csv_path)
    print(f"Loaded {len(gt_dict)} ground truth measurements")
    
    # Load image
    image_path = DATA_DIR / "2.jpeg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_image = np.array(Image.open(image_path).convert('RGB'))
    original_h, original_w = original_image.shape[:2]
    print(f"Loaded image: {original_w}x{original_h}")
    
    # Preprocess image (same as peanuts.py) - A4 detection, crop, rotate, resize
    preprocessed_results = preprocess_images_with_white_rectangle(
        input_images=[original_image],
        target_width=2000
    )
    preprocessed_image, pixels_per_mm = preprocessed_results[0]
    preprocessed_h, preprocessed_w = preprocessed_image.shape[:2]
    
    print(f"Preprocessed image: {preprocessed_w}x{preprocessed_h}")
    print(f"Pixels per mm (preprocessed): {pixels_per_mm:.4f}")
    
    # Load models
    print("\nLoading models...")
    
    # YOLO detector - use same model as peanuts.py (HuggingFace)
    yolo_model_path = None
    if not HF_PEANUT_SEG_REPO_ID or not HF_PEANUT_SEG_FILE:
        raise ValueError("HF_PEANUT_SEG_REPO_ID and HF_PEANUT_SEG_FILE must be set (same as peanuts.py uses)")
    
    try:
        yolo_model_path = hf_hub_download(
            repo_id=HF_PEANUT_SEG_REPO_ID,
            filename=HF_PEANUT_SEG_FILE,
            token=HF_TOKEN
        )
        print(f"YOLO model downloaded from HuggingFace: {yolo_model_path}")
    except Exception as e:
        error_trace = traceback.format_exc()
        raise ValueError(f"Failed to download YOLO model from HuggingFace: {e}\n{error_trace}")
    
    detector = YOLOPeanutsDetector(yolo_model_path)
    print("YOLO detector loaded")
    
    # UNet detector - try HuggingFace first, then local path
    unet_model_path = None
    if HF_PEANUT_SEG_SEPARATED_REPO_ID and HF_PEANUT_SEG_SEPARATED_FILE:
        try:
            unet_model_path = hf_hub_download(
                repo_id=HF_PEANUT_SEG_SEPARATED_REPO_ID,
                filename=HF_PEANUT_SEG_SEPARATED_FILE,
                token=HF_TOKEN
            )
            print(f"UNet model downloaded from HuggingFace: {unet_model_path}")
        except Exception as e:
            print(f"Failed to download UNet model from HuggingFace: {e}")
            unet_model_path = None
    
    # Try local path if HuggingFace failed
    if unet_model_path is None:
        # Try common local paths
        local_unet_paths = [
            CURRENT_DIR / "experiment1" / "weights" / "best.pth",
            ROOT_DIR / "assets" / "models" / "unet" / "separated_unet.pth",
        ]
        for path in local_unet_paths:
            if path.exists():
                unet_model_path = str(path)
                print(f"Using local UNet model: {unet_model_path}")
                break
    
    if unet_model_path is None:
        raise ValueError("UNet model not found. Set HF_PEANUT_SEG_SEPARATED_REPO_ID/HF_PEANUT_SEG_SEPARATED_FILE or provide local model path.")
    
    separated_detector = UNetPeanutsDetector(unet_model_path)
    print("UNet detector loaded")
    
    # Detect peanuts with YOLO on preprocessed image (same as peanuts.py)
    print("\nDetecting peanuts with YOLO on preprocessed image...")
    sv_detections = detector.detect(
        [preprocessed_image],
        verbose=True,
        imgsz=1024,
        conf=0.6
    )
    sv_detection = sv_detections[0]
    
    # Apply NMS
    nms_iou_threshold = 0.5
    sv_detection = sv_detection.with_nms(threshold=nms_iou_threshold)
    
    print(f"Detected {len(sv_detection.xyxy)} peanuts")
    
    # Sort detections (top-to-bottom, left-to-right) - same as peanuts.py
    sorted_indices = sorted(
        range(len(sv_detection.xyxy)),
        key=lambda idx: (sv_detection.xyxy[idx][1], sv_detection.xyxy[idx][0])
    )
    
    # Process each peanut
    results = []
    yolo_bboxes_orig = []
    unet_contours = []
    ellipse_params_list = []
    bbox_params_list = []
    peanut_indices = []
    
    print("\nProcessing each peanut...")
    for ordered_index, index in enumerate(sorted_indices):
        # Get YOLO bbox in preprocessed image coordinates (same as peanuts.py)
        xyxy = sv_detection.xyxy[index]
        x1, y1, x2, y2 = map(int, xyxy)
        bbox = (x1, y1, x2, y2)
        yolo_bboxes_orig.append(bbox)
        peanut_indices.append(ordered_index)
        
        # Crop peanut from preprocessed image (same as peanuts.py)
        one_peanut_image = preprocessed_image[y1:y2, x1:x2].copy()
        
        # Run UNet segmentation on cropped peanut (UNet expects RGB)
        cropped_detections = separated_detector.detect(
            [one_peanut_image],
            verbose=False
        )
        
        mask_separated = None
        contour_separated = None
        ellipse_length_mm = None
        ellipse_width_mm = None
        bbox_length_mm = None
        bbox_width_mm = None
        ellipse_param = None
        bbox_param = None
        
        if (cropped_detections and 
            len(cropped_detections) > 0 and 
            cropped_detections[0].mask is not None and 
            len(cropped_detections[0].mask) > 0):
            
            # Get mask from cropped detection
            cropped_mask = cropped_detections[0].mask[0]
            cropped_h, cropped_w = cropped_mask.shape
            crop_h_actual, crop_w_actual = y2 - y1, x2 - x1
            
            # Resize mask if needed
            if cropped_h != crop_h_actual or cropped_w != crop_w_actual:
                cropped_mask = cv2.resize(
                    cropped_mask.astype(np.uint8),
                    (crop_w_actual, crop_h_actual),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool) 
            
            # Create full-size mask in preprocessed image coordinates
            full_mask = np.zeros((preprocessed_h, preprocessed_w), dtype=bool)
            full_mask[y1:y2, x1:x2] = cropped_mask
            mask_separated = full_mask
            
            # Find contour from full mask
            contours_separated, _ = cv2.findContours(
                mask_separated.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_NONE
            )
            
            if contours_separated:
                contour_separated = max(contours_separated, key=cv2.contourArea)
                unet_contours.append(contour_separated)
                
                # Calculate ellipse measurements using preprocessed pixels_per_mm
                ellipse_length_mm, ellipse_width_mm = calculate_ellipse_measurements(
                    contour_separated,
                    pixels_per_mm
                )
                
                # Get ellipse parameters for visualization
                if len(contour_separated) >= 5:
                    contour_reshaped = contour_separated.reshape(-1, 2)
                    center, axes, angle = cv2.fitEllipse(contour_reshaped)
                    ellipse_param = (center, axes, angle)
                else:
                    ellipse_param = None
                
                # Calculate bbox measurements using preprocessed pixels_per_mm
                bbox_length_mm, bbox_width_mm = calculate_bbox_measurements(
                    contour_separated,
                    pixels_per_mm
                )
                
                # Get bbox parameters for visualization
                if len(contour_separated) >= 3:
                    bbox_param = cv2.minAreaRect(contour_separated)
                else:
                    bbox_param = None
            else:
                unet_contours.append(None)
                ellipse_param = None
                bbox_param = None
        else:
            unet_contours.append(None)
            ellipse_param = None
            bbox_param = None
        
        ellipse_params_list.append(ellipse_param)
        bbox_params_list.append(bbox_param)
        
        # Get ground truth
        gt_length_mm, gt_width_mm = gt_dict.get(ordered_index, (None, None))
        
        # Calculate errors (absolute for MAE/RMSE, signed for Bias)
        ellipse_length_error_signed = (ellipse_length_mm - gt_length_mm) if (ellipse_length_mm is not None and gt_length_mm is not None) else None
        ellipse_width_error_signed = (ellipse_width_mm - gt_width_mm) if (ellipse_width_mm is not None and gt_width_mm is not None) else None
        ellipse_length_error_mm = abs(ellipse_length_error_signed) if ellipse_length_error_signed is not None else None
        ellipse_width_error_mm = abs(ellipse_width_error_signed) if ellipse_width_error_signed is not None else None
        ellipse_length_error_pct = (ellipse_length_error_mm / gt_length_mm * 100) if (ellipse_length_error_mm is not None and gt_length_mm is not None and gt_length_mm > 0) else None
        ellipse_width_error_pct = (ellipse_width_error_mm / gt_width_mm * 100) if (ellipse_width_error_mm is not None and gt_width_mm is not None and gt_width_mm > 0) else None
        
        bbox_length_error_signed = (bbox_length_mm - gt_length_mm) if (bbox_length_mm is not None and gt_length_mm is not None) else None
        bbox_width_error_signed = (bbox_width_mm - gt_width_mm) if (bbox_width_mm is not None and gt_width_mm is not None) else None
        bbox_length_error_mm = abs(bbox_length_error_signed) if bbox_length_error_signed is not None else None
        bbox_width_error_mm = abs(bbox_width_error_signed) if bbox_width_error_signed is not None else None
        bbox_length_error_pct = (bbox_length_error_mm / gt_length_mm * 100) if (bbox_length_error_mm is not None and gt_length_mm is not None and gt_length_mm > 0) else None
        bbox_width_error_pct = (bbox_width_error_mm / gt_width_mm * 100) if (bbox_width_error_mm is not None and gt_width_mm is not None and gt_width_mm > 0) else None
        
        results.append({
            'peanut_index': ordered_index,
            'gt_length_mm': gt_length_mm,
            'gt_width_mm': gt_width_mm,
            'ellipse_length_mm': ellipse_length_mm,
            'ellipse_width_mm': ellipse_width_mm,
            'ellipse_length_error_mm': ellipse_length_error_mm,
            'ellipse_width_error_mm': ellipse_width_error_mm,
            'ellipse_length_error_signed': ellipse_length_error_signed,
            'ellipse_width_error_signed': ellipse_width_error_signed,
            'ellipse_length_error_pct': ellipse_length_error_pct,
            'ellipse_width_error_pct': ellipse_width_error_pct,
            'bbox_length_mm': bbox_length_mm,
            'bbox_width_mm': bbox_width_mm,
            'bbox_length_error_mm': bbox_length_error_mm,
            'bbox_width_error_mm': bbox_width_error_mm,
            'bbox_length_error_signed': bbox_length_error_signed,
            'bbox_width_error_signed': bbox_width_error_signed,
            'bbox_length_error_pct': bbox_length_error_pct,
            'bbox_width_error_pct': bbox_width_error_pct,
        })
        
        print(f"  Peanut {ordered_index}: ellipse=({ellipse_length_mm:.2f}, {ellipse_width_mm:.2f}), bbox=({bbox_length_mm:.2f}, {bbox_width_mm:.2f})")
    
    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    
    # Calculate summary statistics before saving
    print("\n=== Summary Statistics ===")
    
    # Ellipse approach
    if 'ellipse_length_error_mm' in df_results.columns:
        ellipse_length_errors = df_results['ellipse_length_error_mm'].dropna()
        ellipse_width_errors = df_results['ellipse_width_error_mm'].dropna()
        ellipse_length_errors_signed = df_results['ellipse_length_error_signed'].dropna()
        ellipse_width_errors_signed = df_results['ellipse_width_error_signed'].dropna()
        ellipse_length_pred = df_results['ellipse_length_mm'].dropna()
        ellipse_width_pred = df_results['ellipse_width_mm'].dropna()
        ellipse_length_gt = df_results['gt_length_mm'].dropna()
        ellipse_width_gt = df_results['gt_width_mm'].dropna()
    else:
        ellipse_length_errors = pd.Series(dtype=float)
        ellipse_width_errors = pd.Series(dtype=float)
        ellipse_length_errors_signed = pd.Series(dtype=float)
        ellipse_width_errors_signed = pd.Series(dtype=float)
        ellipse_length_pred = pd.Series(dtype=float)
        ellipse_width_pred = pd.Series(dtype=float)
        ellipse_length_gt = pd.Series(dtype=float)
        ellipse_width_gt = pd.Series(dtype=float)
    
    # Calculate all metrics for ellipse length
    ellipse_length_mae = ellipse_length_errors.mean() if len(ellipse_length_errors) > 0 else None
    ellipse_length_rmse = np.sqrt((ellipse_length_errors**2).mean()) if len(ellipse_length_errors) > 0 else None
    ellipse_length_max = ellipse_length_errors.max() if len(ellipse_length_errors) > 0 else None
    ellipse_length_bias = ellipse_length_errors_signed.mean() if len(ellipse_length_errors_signed) > 0 else None
    ellipse_length_std = ellipse_length_errors_signed.std() if len(ellipse_length_errors_signed) > 0 else None
    ellipse_length_p95 = np.percentile(ellipse_length_errors, 95) if len(ellipse_length_errors) > 0 else None
    ellipse_length_p99 = np.percentile(ellipse_length_errors, 99) if len(ellipse_length_errors) > 0 else None
    
    # Calculate correlation for ellipse length (need matching pairs)
    ellipse_length_corr = None
    if len(ellipse_length_pred) > 0 and len(ellipse_length_gt) > 0:
        # Align by index to get matching pairs
        common_idx = ellipse_length_pred.index.intersection(ellipse_length_gt.index)
        if len(common_idx) > 1:
            pred_aligned = ellipse_length_pred.loc[common_idx]
            gt_aligned = ellipse_length_gt.loc[common_idx]
            ellipse_length_corr = pred_aligned.corr(gt_aligned) if len(pred_aligned) > 1 else None
    
    # Calculate all metrics for ellipse width
    ellipse_width_mae = ellipse_width_errors.mean() if len(ellipse_width_errors) > 0 else None
    ellipse_width_rmse = np.sqrt((ellipse_width_errors**2).mean()) if len(ellipse_width_errors) > 0 else None
    ellipse_width_max = ellipse_width_errors.max() if len(ellipse_width_errors) > 0 else None
    ellipse_width_bias = ellipse_width_errors_signed.mean() if len(ellipse_width_errors_signed) > 0 else None
    ellipse_width_std = ellipse_width_errors_signed.std() if len(ellipse_width_errors_signed) > 0 else None
    ellipse_width_p95 = np.percentile(ellipse_width_errors, 95) if len(ellipse_width_errors) > 0 else None
    ellipse_width_p99 = np.percentile(ellipse_width_errors, 99) if len(ellipse_width_errors) > 0 else None
    
    # Calculate correlation for ellipse width
    ellipse_width_corr = None
    if len(ellipse_width_pred) > 0 and len(ellipse_width_gt) > 0:
        common_idx = ellipse_width_pred.index.intersection(ellipse_width_gt.index)
        if len(common_idx) > 1:
            pred_aligned = ellipse_width_pred.loc[common_idx]
            gt_aligned = ellipse_width_gt.loc[common_idx]
            ellipse_width_corr = pred_aligned.corr(gt_aligned) if len(pred_aligned) > 1 else None
    
    if len(ellipse_length_errors) > 0:
        print(f"\nEllipse Approach - Length:")
        print(f"  MAE: {ellipse_length_mae:.4f} mm")
        print(f"  RMSE: {ellipse_length_rmse:.4f} mm")
        print(f"  Bias: {ellipse_length_bias:.4f} mm" if ellipse_length_bias is not None else "  Bias: N/A")
        print(f"  Std Dev: {ellipse_length_std:.4f} mm" if ellipse_length_std is not None else "  Std Dev: N/A")
        print(f"  Max Error: {ellipse_length_max:.4f} mm")
        print(f"  P95: {ellipse_length_p95:.4f} mm" if ellipse_length_p95 is not None else "  P95: N/A")
        print(f"  P99: {ellipse_length_p99:.4f} mm" if ellipse_length_p99 is not None else "  P99: N/A")
        print(f"  Pearson r: {ellipse_length_corr:.4f}" if ellipse_length_corr is not None else "  Pearson r: N/A")
    
    if len(ellipse_width_errors) > 0:
        print(f"\nEllipse Approach - Width:")
        print(f"  MAE: {ellipse_width_mae:.4f} mm")
        print(f"  RMSE: {ellipse_width_rmse:.4f} mm")
        print(f"  Bias: {ellipse_width_bias:.4f} mm" if ellipse_width_bias is not None else "  Bias: N/A")
        print(f"  Std Dev: {ellipse_width_std:.4f} mm" if ellipse_width_std is not None else "  Std Dev: N/A")
        print(f"  Max Error: {ellipse_width_max:.4f} mm")
        print(f"  P95: {ellipse_width_p95:.4f} mm" if ellipse_width_p95 is not None else "  P95: N/A")
        print(f"  P99: {ellipse_width_p99:.4f} mm" if ellipse_width_p99 is not None else "  P99: N/A")
        print(f"  Pearson r: {ellipse_width_corr:.4f}" if ellipse_width_corr is not None else "  Pearson r: N/A")
    
    # Bbox approach
    if 'bbox_length_error_mm' in df_results.columns:
        bbox_length_errors = df_results['bbox_length_error_mm'].dropna()
        bbox_width_errors = df_results['bbox_width_error_mm'].dropna()
        bbox_length_errors_signed = df_results['bbox_length_error_signed'].dropna()
        bbox_width_errors_signed = df_results['bbox_width_error_signed'].dropna()
        bbox_length_pred = df_results['bbox_length_mm'].dropna()
        bbox_width_pred = df_results['bbox_width_mm'].dropna()
        bbox_length_gt = df_results['gt_length_mm'].dropna()
        bbox_width_gt = df_results['gt_width_mm'].dropna()
    else:
        bbox_length_errors = pd.Series(dtype=float)
        bbox_width_errors = pd.Series(dtype=float)
        bbox_length_errors_signed = pd.Series(dtype=float)
        bbox_width_errors_signed = pd.Series(dtype=float)
        bbox_length_pred = pd.Series(dtype=float)
        bbox_width_pred = pd.Series(dtype=float)
        bbox_length_gt = pd.Series(dtype=float)
        bbox_width_gt = pd.Series(dtype=float)
    
    # Calculate all metrics for bbox length
    bbox_length_mae = bbox_length_errors.mean() if len(bbox_length_errors) > 0 else None
    bbox_length_rmse = np.sqrt((bbox_length_errors**2).mean()) if len(bbox_length_errors) > 0 else None
    bbox_length_max = bbox_length_errors.max() if len(bbox_length_errors) > 0 else None
    bbox_length_bias = bbox_length_errors_signed.mean() if len(bbox_length_errors_signed) > 0 else None
    bbox_length_std = bbox_length_errors_signed.std() if len(bbox_length_errors_signed) > 0 else None
    bbox_length_p95 = np.percentile(bbox_length_errors, 95) if len(bbox_length_errors) > 0 else None
    bbox_length_p99 = np.percentile(bbox_length_errors, 99) if len(bbox_length_errors) > 0 else None
    
    # Calculate correlation for bbox length
    bbox_length_corr = None
    if len(bbox_length_pred) > 0 and len(bbox_length_gt) > 0:
        common_idx = bbox_length_pred.index.intersection(bbox_length_gt.index)
        if len(common_idx) > 1:
            pred_aligned = bbox_length_pred.loc[common_idx]
            gt_aligned = bbox_length_gt.loc[common_idx]
            bbox_length_corr = pred_aligned.corr(gt_aligned) if len(pred_aligned) > 1 else None
    
    # Calculate all metrics for bbox width
    bbox_width_mae = bbox_width_errors.mean() if len(bbox_width_errors) > 0 else None
    bbox_width_rmse = np.sqrt((bbox_width_errors**2).mean()) if len(bbox_width_errors) > 0 else None
    bbox_width_max = bbox_width_errors.max() if len(bbox_width_errors) > 0 else None
    bbox_width_bias = bbox_width_errors_signed.mean() if len(bbox_width_errors_signed) > 0 else None
    bbox_width_std = bbox_width_errors_signed.std() if len(bbox_width_errors_signed) > 0 else None
    bbox_width_p95 = np.percentile(bbox_width_errors, 95) if len(bbox_width_errors) > 0 else None
    bbox_width_p99 = np.percentile(bbox_width_errors, 99) if len(bbox_width_errors) > 0 else None
    
    # Calculate correlation for bbox width
    bbox_width_corr = None
    if len(bbox_width_pred) > 0 and len(bbox_width_gt) > 0:
        common_idx = bbox_width_pred.index.intersection(bbox_width_gt.index)
        if len(common_idx) > 1:
            pred_aligned = bbox_width_pred.loc[common_idx]
            gt_aligned = bbox_width_gt.loc[common_idx]
            bbox_width_corr = pred_aligned.corr(gt_aligned) if len(pred_aligned) > 1 else None
    
    if len(bbox_length_errors) > 0:
        print(f"\nBbox Approach - Length:")
        print(f"  MAE: {bbox_length_mae:.4f} mm")
        print(f"  RMSE: {bbox_length_rmse:.4f} mm")
        print(f"  Bias: {bbox_length_bias:.4f} mm" if bbox_length_bias is not None else "  Bias: N/A")
        print(f"  Std Dev: {bbox_length_std:.4f} mm" if bbox_length_std is not None else "  Std Dev: N/A")
        print(f"  Max Error: {bbox_length_max:.4f} mm")
        print(f"  P95: {bbox_length_p95:.4f} mm" if bbox_length_p95 is not None else "  P95: N/A")
        print(f"  P99: {bbox_length_p99:.4f} mm" if bbox_length_p99 is not None else "  P99: N/A")
        print(f"  Pearson r: {bbox_length_corr:.4f}" if bbox_length_corr is not None else "  Pearson r: N/A")
    
    if len(bbox_width_errors) > 0:
        print(f"\nBbox Approach - Width:")
        print(f"  MAE: {bbox_width_mae:.4f} mm")
        print(f"  RMSE: {bbox_width_rmse:.4f} mm")
        print(f"  Bias: {bbox_width_bias:.4f} mm" if bbox_width_bias is not None else "  Bias: N/A")
        print(f"  Std Dev: {bbox_width_std:.4f} mm" if bbox_width_std is not None else "  Std Dev: N/A")
        print(f"  Max Error: {bbox_width_max:.4f} mm")
        print(f"  P95: {bbox_width_p95:.4f} mm" if bbox_width_p95 is not None else "  P95: N/A")
        print(f"  P99: {bbox_width_p99:.4f} mm" if bbox_width_p99 is not None else "  P99: N/A")
        print(f"  Pearson r: {bbox_width_corr:.4f}" if bbox_width_corr is not None else "  Pearson r: N/A")
    
    # Add summary statistics as rows to the DataFrame
    summary_rows = [
        {
            'peanut_index': 'Ellipse_Length_MAE',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': ellipse_length_mae,
            'ellipse_width_mm': None,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': None,
            'bbox_width_mm': None,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Ellipse_Length_RMSE',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': ellipse_length_rmse,
            'ellipse_width_mm': None,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': None,
            'bbox_width_mm': None,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Ellipse_Length_MaxError',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': ellipse_length_max,
            'ellipse_width_mm': None,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': None,
            'bbox_width_mm': None,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Ellipse_Width_MAE',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': None,
            'ellipse_width_mm': ellipse_width_mae,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': None,
            'bbox_width_mm': None,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Ellipse_Width_RMSE',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': None,
            'ellipse_width_mm': ellipse_width_rmse,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': None,
            'bbox_width_mm': None,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Ellipse_Width_MaxError',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': None,
            'ellipse_width_mm': ellipse_width_max,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': None,
            'bbox_width_mm': None,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Bbox_Length_MAE',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': None,
            'ellipse_width_mm': None,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': bbox_length_mae,
            'bbox_width_mm': None,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Bbox_Length_RMSE',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': None,
            'ellipse_width_mm': None,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': bbox_length_rmse,
            'bbox_width_mm': None,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Bbox_Length_MaxError',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': None,
            'ellipse_width_mm': None,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': bbox_length_max,
            'bbox_width_mm': None,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Bbox_Width_MAE',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': None,
            'ellipse_width_mm': None,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': None,
            'bbox_width_mm': bbox_width_mae,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Bbox_Width_RMSE',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': None,
            'ellipse_width_mm': None,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': None,
            'bbox_width_mm': bbox_width_rmse,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
        {
            'peanut_index': 'Bbox_Width_MaxError',
            'gt_length_mm': None,
            'gt_width_mm': None,
            'ellipse_length_mm': None,
            'ellipse_width_mm': None,
            'ellipse_length_error_mm': None,
            'ellipse_width_error_mm': None,
            'ellipse_length_error_pct': None,
            'ellipse_width_error_pct': None,
            'bbox_length_mm': None,
            'bbox_width_mm': bbox_width_max,
            'bbox_length_error_mm': None,
            'bbox_width_error_mm': None,
            'bbox_length_error_pct': None,
            'bbox_width_error_pct': None,
        },
    ]
    
    # Append summary rows to results
    df_results = pd.concat([df_results, pd.DataFrame(summary_rows)], ignore_index=True)
    
    csv_output_path = OUTPUT_DIR / "comparison_results.csv"
    df_results.to_csv(csv_output_path, index=False)
    print(f"\nResults saved to: {csv_output_path}")
    
    if len(df_results) == 0:
        print("\nWARNING: No peanuts were detected/processed. Results CSV is empty.")
        return
    
    # Save preprocessed image
    print("\nSaving preprocessed image...")
    preprocessed_path = OUTPUT_DIR / "preprocessed_image.jpg"
    Image.fromarray(preprocessed_image).save(preprocessed_path)
    print(f"Preprocessed image saved to: {preprocessed_path}")
    
    # Create visualization on preprocessed image (same coordinate space as detection)
    print("\nCreating visualization...")
    vis_path = OUTPUT_DIR / "visualization.jpg"
    create_visualization(
        preprocessed_image,
        yolo_bboxes_orig,
        unet_contours,
        ellipse_params_list,
        bbox_params_list,
        peanut_indices,
        vis_path
    )
    print(f"Visualization saved to: {vis_path}")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

