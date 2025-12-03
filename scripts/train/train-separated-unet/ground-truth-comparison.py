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
    for yolo_bbox, unet_contour, ellipse_param, bbox_param in zip(
        yolo_bboxes, unet_contours, ellipse_params, bbox_params
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
    
    # Convert BGR back to RGB for saving
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    Image.fromarray(vis_img_rgb).save(output_path)


def main():
    # Load ground truth
    csv_path = DATA_DIR / "1.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground truth CSV not found: {csv_path}")
    
    gt_dict = load_ground_truth(csv_path)
    print(f"Loaded {len(gt_dict)} ground truth measurements")
    
    # Load image
    image_path = DATA_DIR / "1.jpeg"
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    original_image = np.array(Image.open(image_path).convert('RGB'))
    original_h, original_w = original_image.shape[:2]
    print(f"Loaded image: {original_w}x{original_h}")
    
    # Preprocess image to get pixels_per_mm (but we'll detect on original)
    preprocessed_results = preprocess_images_with_white_rectangle(
        input_images=[original_image],
        target_width=2000
    )
    preprocessed_image, pixels_per_mm_preprocessed = preprocessed_results[0]
    preprocessed_h, preprocessed_w = preprocessed_image.shape[:2]
    
    # Calculate pixels_per_mm for original image
    # The preprocessing detects A4 paper in original, then resizes
    # We need pixels_per_mm for the original image before resize
    # Get the A4 detection from original to calculate pixels_per_mm
    from mlbox.utils.cvtools import detect_white_rectangles
    rectangles = detect_white_rectangles(
        original_image,
        aspect_ratio=297/210,
        sheet_width=210,
        sheet_height=297
    )
    if rectangles:
        # Calculate pixels_per_mm from original image A4 detection
        rect_points = np.array([point[0] for point in rectangles[0]["rectangle"]])
        d1 = np.linalg.norm(rect_points[0] - rect_points[1])
        d2 = np.linalg.norm(rect_points[1] - rect_points[2])
        d3 = np.linalg.norm(rect_points[2] - rect_points[3])
        d4 = np.linalg.norm(rect_points[3] - rect_points[0])
        sides = sorted([d1, d2, d3, d4])
        sheet_width_px_original = (sides[0] + sides[1]) / 2
        pixels_per_mm = sheet_width_px_original / 210.0
    else:
        # Fallback: estimate from preprocessed
        scale_x = preprocessed_w / original_w
        pixels_per_mm = pixels_per_mm_preprocessed / scale_x
    
    print(f"Preprocessed image: {preprocessed_w}x{preprocessed_h}")
    print(f"Pixels per mm (original): {pixels_per_mm:.4f}")
    
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
    
    # Detect peanuts with YOLO on original image (to avoid coordinate conversion issues)
    print("\nDetecting peanuts with YOLO on original image...")
    sv_detections = detector.detect(
        [original_image],
        verbose=True,
        imgsz=1024,
        conf=0.6
    )
    sv_detection = sv_detections[0]
    
    # Apply NMS
    nms_iou_threshold = 0.5
    sv_detection = sv_detection.with_nms(threshold=nms_iou_threshold)
    
    print(f"Detected {len(sv_detection.xyxy)} peanuts")
    
    # Sort detections (top-to-bottom, left-to-right)
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
    
    print("\nProcessing each peanut...")
    for ordered_index, index in enumerate(sorted_indices):
        # Get YOLO bbox in original image coordinates (already in original!)
        xyxy = sv_detection.xyxy[index]
        x1_orig, y1_orig, x2_orig, y2_orig = map(int, xyxy)
        bbox_orig = (x1_orig, y1_orig, x2_orig, y2_orig)
        yolo_bboxes_orig.append(bbox_orig)
        
        # Crop peanut from original image (already RGB from PIL)
        one_peanut_image = original_image[y1_orig:y2_orig, x1_orig:x2_orig].copy()
        
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
            crop_h_actual, crop_w_actual = y2_orig - y1_orig, x2_orig - x1_orig
            
            # Resize mask if needed
            if cropped_h != crop_h_actual or cropped_w != crop_w_actual:
                cropped_mask = cv2.resize(
                    cropped_mask.astype(np.uint8),
                    (crop_w_actual, crop_h_actual),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            
            # Create full-size mask in original image coordinates
            full_mask = np.zeros((original_h, original_w), dtype=bool)
            full_mask[y1_orig:y2_orig, x1_orig:x2_orig] = cropped_mask
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
                
                # Calculate ellipse measurements
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
                
                # Calculate bbox measurements
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
        
        # Calculate errors
        ellipse_length_error_mm = abs(ellipse_length_mm - gt_length_mm) if (ellipse_length_mm is not None and gt_length_mm is not None) else None
        ellipse_width_error_mm = abs(ellipse_width_mm - gt_width_mm) if (ellipse_width_mm is not None and gt_width_mm is not None) else None
        ellipse_length_error_pct = (ellipse_length_error_mm / gt_length_mm * 100) if (ellipse_length_error_mm is not None and gt_length_mm is not None and gt_length_mm > 0) else None
        ellipse_width_error_pct = (ellipse_width_error_mm / gt_width_mm * 100) if (ellipse_width_error_mm is not None and gt_width_mm is not None and gt_width_mm > 0) else None
        
        bbox_length_error_mm = abs(bbox_length_mm - gt_length_mm) if (bbox_length_mm is not None and gt_length_mm is not None) else None
        bbox_width_error_mm = abs(bbox_width_mm - gt_width_mm) if (bbox_width_mm is not None and gt_width_mm is not None) else None
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
            'ellipse_length_error_pct': ellipse_length_error_pct,
            'ellipse_width_error_pct': ellipse_width_error_pct,
            'bbox_length_mm': bbox_length_mm,
            'bbox_width_mm': bbox_width_mm,
            'bbox_length_error_mm': bbox_length_error_mm,
            'bbox_width_error_mm': bbox_width_error_mm,
            'bbox_length_error_pct': bbox_length_error_pct,
            'bbox_width_error_pct': bbox_width_error_pct,
        })
        
        print(f"  Peanut {ordered_index}: ellipse=({ellipse_length_mm:.2f}, {ellipse_width_mm:.2f}), bbox=({bbox_length_mm:.2f}, {bbox_width_mm:.2f})")
    
    # Create DataFrame and save
    df_results = pd.DataFrame(results)
    csv_output_path = OUTPUT_DIR / "comparison_results.csv"
    df_results.to_csv(csv_output_path, index=False)
    print(f"\nResults saved to: {csv_output_path}")
    
    if len(df_results) == 0:
        print("\nWARNING: No peanuts were detected/processed. Results CSV is empty.")
        return
    
    # Calculate and print summary statistics
    print("\n=== Summary Statistics ===")
    
    # Ellipse approach
    if 'ellipse_length_error_mm' in df_results.columns:
        ellipse_length_errors = df_results['ellipse_length_error_mm'].dropna()
        ellipse_width_errors = df_results['ellipse_width_error_mm'].dropna()
    else:
        ellipse_length_errors = pd.Series(dtype=float)
        ellipse_width_errors = pd.Series(dtype=float)
    
    if len(ellipse_length_errors) > 0:
        print(f"\nEllipse Approach - Length:")
        print(f"  MAE: {ellipse_length_errors.mean():.4f} mm")
        print(f"  RMSE: {np.sqrt((ellipse_length_errors**2).mean()):.4f} mm")
        print(f"  Max Error: {ellipse_length_errors.max():.4f} mm")
    
    if len(ellipse_width_errors) > 0:
        print(f"\nEllipse Approach - Width:")
        print(f"  MAE: {ellipse_width_errors.mean():.4f} mm")
        print(f"  RMSE: {np.sqrt((ellipse_width_errors**2).mean()):.4f} mm")
        print(f"  Max Error: {ellipse_width_errors.max():.4f} mm")
    
    # Bbox approach
    if 'bbox_length_error_mm' in df_results.columns:
        bbox_length_errors = df_results['bbox_length_error_mm'].dropna()
        bbox_width_errors = df_results['bbox_width_error_mm'].dropna()
    else:
        bbox_length_errors = pd.Series(dtype=float)
        bbox_width_errors = pd.Series(dtype=float)
    
    if len(bbox_length_errors) > 0:
        print(f"\nBbox Approach - Length:")
        print(f"  MAE: {bbox_length_errors.mean():.4f} mm")
        print(f"  RMSE: {np.sqrt((bbox_length_errors**2).mean()):.4f} mm")
        print(f"  Max Error: {bbox_length_errors.max():.4f} mm")
    
    if len(bbox_width_errors) > 0:
        print(f"\nBbox Approach - Width:")
        print(f"  MAE: {bbox_width_errors.mean():.4f} mm")
        print(f"  RMSE: {np.sqrt((bbox_width_errors**2).mean()):.4f} mm")
        print(f"  Max Error: {bbox_width_errors.max():.4f} mm")
    
    # Create visualization
    print("\nCreating visualization...")
    vis_path = OUTPUT_DIR / "visualization.jpg"
    create_visualization(
        original_image,
        yolo_bboxes_orig,
        unet_contours,
        ellipse_params_list,
        bbox_params_list,
        vis_path
    )
    print(f"Visualization saved to: {vis_path}")
    
    print("\n=== Done ===")


if __name__ == "__main__":
    main()

