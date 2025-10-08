"""
Computer Vision Tools Module

Public Utility Functions:
    - vector_angle: Calculate angle between vectors
    - euclidean_distance: Calculate distance between points

Public API Functions:
    - detect_white_rectangles: Detect rectangles with optional aspect ratio filtering
    - masks_to_coco_annotations: Convert masks to COCO annotation format
    - combine_coco_annotation_files: Combine multiple COCO annotation files
"""

import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as coco_mask
from PIL import Image as PILImage

from mlbox.settings import ROOT_DIR

# Type aliases
Point = Tuple[int, int]
Rectangle = np.ndarray
Contour = np.ndarray
CURRENT_DIR = Path(__file__).parent

def vector_angle(pt1: Point, pt2: Point, pt0: Point) -> float:
    """Calculate cosine of angle between vectors pt1->pt0 and pt2->pt0."""
    dx1 = pt1[0] - pt0[0]
    dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]
    dy2 = pt2[1] - pt0[1]
    norm1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    norm2 = math.sqrt(dx2 * dx2 + dy2 * dy2)

    if norm1 == 0 or norm2 == 0:
        return 0
    return (dx1 * dx2 + dy1 * dy2) / (norm1 * norm2)


def euclidean_distance(pt1: Point, pt2: Point) -> float:
    """Calculate Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def masks_to_coco_annotations(
    masks: List[np.ndarray],
    image_id: int,
    filename: str,
    categories: List[Dict],
    segmentation_type: str = "polygon",
    approximation_threshold: float = 0.001,
) -> str:

    if not masks:
        return json.dumps({"images": [], "annotations": [], "categories": categories})
    """Convert binary masks to COCO JSON annotation format."""
    annotations = []
    annotation_id = 1

    image_width = masks[0].shape[1]
    epsilon = image_width * approximation_threshold

    for mask in masks:
        if segmentation_type == "polygon":
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                # cv2.CHAIN_APPROX_SIMPLE
                cv2.CHAIN_APPROX_TC89_L1,
            )
            polygons = []

            for contour in contours:
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                segmentation = approx_contour.flatten().tolist()
                if len(segmentation) >= 6:
                    polygons.append(segmentation)

            if not polygons:
                continue

            segmentation = polygons
        elif segmentation_type == "raster":
            try:
                rle = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")
                segmentation = rle
            except ImportError:
                raise ImportError("pycocotools is required for raster segmentation")
        else:
            raise ValueError("Invalid segmentation type. Use 'polygon' or 'raster'.")

        y_indices, x_indices = np.where(mask == 1)
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]

        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": segmentation,
                "area": int(np.sum(mask)),
                "bbox": bbox,
                "iscrowd": 0,
            }
        )
        annotation_id += 1

    coco_json = {
        "images": [
            {
                "file_name": filename,
                "height": mask.shape[0],
                "width": mask.shape[1],
                "id": image_id,
            }
        ],
        "annotations": annotations,
        "categories": categories,
    }

    return json.dumps(coco_json)


def combine_coco_annotation_files(coco_files_directory: str) -> str:
    """Combine multiple COCO annotation files into single file."""
    combined_coco = {"images": [], "annotations": [], "categories": []}
    category_ids_set = set()
    current_image_id = 1
    current_annotation_id = 1

    for filename in os.listdir(coco_files_directory):
        if filename.endswith("_coco.json") and filename != "annotation_coco.json":
            with open(os.path.join(coco_files_directory, filename), "r") as f:
                coco_data = json.load(f)

                for category in coco_data["categories"]:
                    if category["id"] not in category_ids_set:
                        combined_coco["categories"].append(category)
                        category_ids_set.add(category["id"])

                image_id_map = {}
                for image in coco_data["images"]:
                    image_id_map[image["id"]] = current_image_id
                    image["id"] = current_image_id
                    combined_coco["images"].append(image)
                    current_image_id += 1

                for annotation in coco_data["annotations"]:
                    annotation["id"] = current_annotation_id
                    annotation["image_id"] = image_id_map[annotation["image_id"]]
                    combined_coco["annotations"].append(annotation)
                    current_annotation_id += 1

    return combined_coco


def detect_white_rectangles(
    image: np.ndarray,
    aspect_ratio: Optional[float] = 297 / 210,
    angle_tolerance: Optional[float] = 5,
    aspect_ratio_tolerance: Optional[float] = 0.05,
    sheet_width: Optional[float] = 210,
    sheet_height: Optional[float] = 297,
    debug_mode: Optional[bool] = False,
    output_dir: Optional[Path] = None,
) -> List[Dict]:
    """
    Detect white rectangles in image. If aspect_ratio provided, filter and sort by ratio match.

    Args:
        image: Input BGR image
        aspect_ratio: Expected height/width ratio (optional)
        angle_tolerance: Maximum angle deviation from 90°
        aspect_ratio_tolerance: Maximum aspect ratio deviation
        sheet_width: Width of the reference sheet in mm (optional)
        sheet_height:   Height of the reference sheet in mm (optional)

    Returns:
        List of rectangles sorted by total score in descending order
    """

    def is_distinguishable(
        rect: Rectangle, rectangles: List[Rectangle], tolerance_distance: float = 10
    ) -> bool:
        def calculate_similarity_score(rect1, rect2):
            rect1_points = [pt[0] for pt in rect1]
            rect2_points = [pt[0] for pt in rect2]
            total_distance = 0
            matched_indices = set()

            # Find minimum distance for each vertex in rect1 to rect2, without repeating matches
            for pt1 in rect1_points:
                min_distance = float("inf")
                min_index = -1
                for i, pt2 in enumerate(rect2_points):
                    if i in matched_indices:
                        continue
                    distance = euclidean_distance(pt1, pt2)
                    if distance < min_distance:
                        min_distance = distance
                        min_index = i
                total_distance += min_distance
                matched_indices.add(min_index)

            return total_distance

        for existing_rect in rectangles:
            similarity_score = calculate_similarity_score(rect, existing_rect)
            if similarity_score < tolerance_distance:
                return False

        return True

    try:
        rectangles = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        min_area = image.shape[0] * image.shape[1] / 9

        # Find all rectangles
        for threshold in range(int(ret), 256, 10):
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [
                contour for contour in contours if cv2.contourArea(contour) > min_area
            ]

            if debug_mode:
                save_image(
                    image=thresh,
                    contours=contours,
                    file_name=output_dir / f"thresh_{threshold}.jpg",
                )

            for i, cnt in enumerate(contours):

                approx = cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.02, True)

                if debug_mode:
                    save_image(
                        image=thresh,
                        contours=[cnt],
                        approx=approx,
                        file_name=output_dir / f"thresh_approx_{i}_{threshold}.jpg",
                    )

                if len(approx) == 4:
                    # Check angles
                    max_angle_deviation = 0
                    for j in range(4):
                        pt1 = approx[j % 4][0]
                        pt2 = approx[(j - 2) % 4][0]
                        pt0 = approx[(j - 1) % 4][0]
                        cosine = abs(vector_angle(pt1, pt2, pt0))
                        angle_degrees = math.degrees(math.acos(cosine))
                        angle_deviation = min(
                            abs(angle_degrees - 90), abs(angle_degrees - 270)
                        )
                        max_angle_deviation = max(max_angle_deviation, angle_deviation)

                    # Get width and height of rectangle
                    d1 = euclidean_distance(approx[0][0], approx[1][0])
                    d2 = euclidean_distance(approx[1][0], approx[2][0])
                    d3 = euclidean_distance(approx[2][0], approx[3][0])
                    d4 = euclidean_distance(approx[3][0], approx[0][0])

                    width = min(d1, d2, d3, d4)
                    height = max(d1, d2, d3, d4)

                    rect_aspect_ratio = (
                        height / width if height > width else width / height
                    )
                    aspect_ratio_deviation = (
                        abs(rect_aspect_ratio - aspect_ratio)
                        if aspect_ratio is not None
                        else 0
                    )

                    # Calculate aspect ratio score
                    if aspect_ratio is not None:
                        aspect_ratio_score = 1 - abs(
                            rect_aspect_ratio - aspect_ratio
                        ) / (aspect_ratio + aspect_ratio_tolerance)
                        aspect_ratio_score = max(0, aspect_ratio_score)
                    else:
                        aspect_ratio_score = 1

                    # Calculate angle score
                    angle_score = 1 - abs(max_angle_deviation) / angle_tolerance
                    angle_score = max(0, angle_score)

                    # Calculate whiteness score
                    mask = np.zeros_like(gray)
                    cv2.fillPoly(mask, [approx], 255)
                    whiteness_score = cv2.mean(gray, mask=mask)[0] / 255.0

                    # Calculate total score (weighted average)
                    total_score = (
                        (0.3 * aspect_ratio_score)
                        + (0.3 * angle_score)
                        + (0.4 * whiteness_score)
                    )

                    # Calculate pixels per mm if sheet dimensions are provided
                    pixels_per_mm = None
                    if sheet_width is not None and sheet_height is not None:
                        pixels_per_mm = (
                            width / sheet_width
                            if width > height
                            else height / sheet_height
                        )

                    rect_data = {
                        "rectangle": approx,
                        "max_angle_deviation": max_angle_deviation,
                        "width": width,
                        "height": height,
                        "aspect_ratio_deviation": aspect_ratio_deviation,
                        "aspect_ratio_score": aspect_ratio_score,
                        "angle_score": angle_score,
                        "whiteness_score": whiteness_score,
                        "total_score": total_score,
                        "pixels_per_mm": pixels_per_mm,
                    }

                    if (
                        max_angle_deviation <= angle_tolerance
                        and aspect_ratio_deviation <= aspect_ratio_tolerance
                        and is_distinguishable(
                            approx, [r["rectangle"] for r in rectangles]
                        )
                    ):
                        if debug_mode:
                            save_image(
                                image=thresh,
                                contours=[cnt],
                                approx=approx,
                                file_name=output_dir
                                / f"thresh_result_{i}_{threshold}.jpg",
                            )
                        rectangles.append(rect_data)

        # Sort rectangles by total score in descending order
        rectangles.sort(key=lambda x: x["total_score"], reverse=True)
        return rectangles

    except Exception as e:
        raise


def save_image(
    image: np.ndarray,
    file_name: Path,
    contours: Optional[List[Contour]] = None,
    approx: Optional[Contour] = None,
):

    if len(image.shape) == 2:
        image_copy = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_copy = image.copy()

    if contours is not None and len(contours) > 0:
        cv2.drawContours(image_copy, contours, -1, (0, 255, 100), 5)
    if approx is not None:
        cv2.drawContours(image_copy, [approx], -1, (0, 100, 255), 3)

    cv2.imwrite(str(file_name), image_copy)


def preprocess_images_with_white_rectangle(
    input_images: List[np.ndarray],
    a4_ratio: float = 297 / 210,
    target_width: Optional[int] = None,
    #padding_percent: float = 0.005,
    padding_percent: float = 0.01,
    sheet_width: Optional[float] = 210,
    sheet_height: Optional[float] = 297,

    ) -> List[Tuple[np.ndarray, float]]:
    """
    Preprocess a batch of images containing white rectangles (e.g., A4 paper) for further analysis.

    This function:
    - detects a white rectangle in each image
    - rotates it so that the wider side is horizontal
    - the rectangle with optional padding
    - resizes it to the specified width while maintaining  aspect ratio.

    Parameters:
        input_images (List[PILImage]): A list of input images to preprocess.
        a4_ratio (float): The aspect ratio of the white rectangle to detect. Default is A4 paper ratio (297/210).
        target_width (Optional[int]): The desired width of the output images. If None, no resizing is performed.
        padding_percent (float): The percentage of padding to add around the detected rectangle. Default is 0.01 (1%).

    Returns:
        Tuple[List[np.ndarray], List[float]]:
            - A list of preprocessed images as NumPy arrays.
            - A list of pixel-to-mm conversion factors for each image.
    """

    def rotate_image(
        image: np.ndarray, rect_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Find the longer side of the rectangle
        d1 = np.linalg.norm(rect_points[0] - rect_points[1])
        d2 = np.linalg.norm(rect_points[1] - rect_points[2])
        pts = (
            (rect_points[0], rect_points[1])
            if d1 > d2
            else (rect_points[1], rect_points[2])
        )
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        
        # Calculate the angle of the longer side relative to horizontal
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        
        # Normalize angle to [-90, 90] to ensure minimum rotation
        # This makes the longer side horizontal with minimal rotation
        if angle_deg > 90:
            angle_deg -= 180
        elif angle_deg < -90:
            angle_deg += 180
        
        # Rotate by the angle to make horizontal (positive angle rotates counter-clockwise)
        rotation_angle = angle_deg

        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (width, height))
        rect_points_rotated = cv2.transform(np.array([rect_points]), M)[0]
        return rotated, rect_points_rotated

    def crop_and_resize(image: np.ndarray, rect_points: np.ndarray) -> np.ndarray:
        min_x = np.min(rect_points[:, 0])
        max_x = np.max(rect_points[:, 0])
        min_y = np.min(rect_points[:, 1])
        max_y = np.max(rect_points[:, 1])
        padding_x = int(padding_percent * (max_x - min_x))
        padding_y = int(padding_percent * (max_y - min_y))
        x1 = max(0, int(min_x) - padding_x)
        x2 = min(image.shape[1], int(max_x) + padding_x)
        y1 = max(0, int(min_y) - padding_y)
        y2 = min(image.shape[0], int(max_y) + padding_y)
        cropped = image[y1:y2, x1:x2]
        new_rect_points = rect_points.copy()

        if target_width is not None:
            aspect_ratio = cropped.shape[1] / cropped.shape[0]
            new_height = int(target_width / aspect_ratio)
            resized = cv2.resize(cropped, (target_width, new_height))
            
            # Calculate new rect_points for resized image
            scale_x = target_width / cropped.shape[1]
            scale_y = new_height / cropped.shape[0]
            
            # Transform original rect_points to new coordinates
            new_rect_points[:, 0] = (new_rect_points[:, 0] - x1) * scale_x
            new_rect_points[:, 1] = (new_rect_points[:, 1] - y1) * scale_y
            
            return resized, new_rect_points

        return cropped

    processed_data = []

    for np_image in input_images:
        # Detect A4 paper
        rectangles = detect_white_rectangles(np_image, aspect_ratio=a4_ratio, sheet_width=sheet_width, sheet_height=sheet_height)
        if not rectangles:
            raise ValueError(f"No A4 paper detected. Image size: {np_image.shape}")
            
        rect_points = np.array([point[0] for point in rectangles[0]["rectangle"]])
        pixels_per_mm = rectangles[0]["pixels_per_mm"]

        # Rotate image
        rotated_image, rotated_rect_points = rotate_image(np_image, rect_points)

        # Crop and resize
        processed_image, rotated_rect_points = crop_and_resize(rotated_image, rotated_rect_points)

        # Calculate all four side lengths of the rotated rectangle
        d1 = np.linalg.norm(rotated_rect_points[0] - rotated_rect_points[1])
        d2 = np.linalg.norm(rotated_rect_points[1] - rotated_rect_points[2])
        d3 = np.linalg.norm(rotated_rect_points[2] - rotated_rect_points[3])
        d4 = np.linalg.norm(rotated_rect_points[3] - rotated_rect_points[0])
        
        # Sort the side lengths and take the average of the two smallest
        # This handles imperfect rectangles where opposite sides aren't exactly equal
        sides = sorted([d1, d2, d3, d4])
        sheet_width_px = (sides[0] + sides[1]) / 2
        
        # Calculate pixels per mm using the actual width
        pixels_per_mm = sheet_width_px / sheet_width


        # Pair processed image with pixels_per_mm
        processed_data.append((processed_image, pixels_per_mm))

    return processed_data


if __name__ == "__main__":
    
    input_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/tests")
    specific_file = "POCO_5200_10_WHITE.jpg"
    image_path = input_folder / specific_file

    # Load the image
    np_image = cv2.imread(str(image_path))
    
    # Process the single image using the preprocessing function
    processed_data = preprocess_images_with_white_rectangle([np_image])
    
    # Extract the processed image and pixels_per_mm
    processed_image, pixels_per_mm = processed_data[0]
    
    print(f"Image: {image_path.name}, pixels_per_mm: {pixels_per_mm}")
    print(f"Original image shape: {np_image.shape}")
    print(f"Processed image shape: {processed_image.shape}")
    
    # Optionally save the processed image
    output_path = input_folder / f"processed_{specific_file}"
    cv2.imwrite(str(output_path), processed_image)
    print(f"Processed image saved to: {output_path}")

