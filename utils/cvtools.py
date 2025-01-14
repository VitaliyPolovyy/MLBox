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

from typing import List, Dict, Union, Tuple
import cv2
import numpy as np
import math
import json
import os

# Type aliases
Point = Tuple[int, int]
Rectangle = np.ndarray
Contour = np.ndarray


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
        approximation_threshold: float = 0.001
) -> str:

    if not masks:
        return json.dumps({
            "images": [],
            "annotations": [],
            "categories": categories
        })
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
                #cv2.CHAIN_APPROX_SIMPLE
                cv2.CHAIN_APPROX_TC89_L1
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
                import pycocotools.mask as coco_mask
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

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "segmentation": segmentation,
            "area": int(np.sum(mask)),
            "bbox": bbox,
            "iscrowd": 0
        })
        annotation_id += 1

    coco_json = {
        "images": [{
            "file_name": filename,
            "height": mask.shape[0],
            "width": mask.shape[1],
            "id": image_id
        }],
        "annotations": annotations,
        "categories": categories
    }

    return json.dumps(coco_json)


def combine_coco_annotation_files(coco_files_directory: str) -> str:
    """Combine multiple COCO annotation files into single file."""
    combined_coco = {
        "images": [],
        "annotations": [],
        "categories": []
    }
    category_ids_set = set()
    current_image_id = 1
    current_annotation_id = 1

    for filename in os.listdir(coco_files_directory):
        if filename.endswith("_coco.json") and filename != "annotation_coco.json":
            with open(os.path.join(coco_files_directory, filename), 'r') as f:
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


if __name__ == "__main__":
    # Module can be run directly for testing
    pass


def detect_white_rectangles(
        image: np.ndarray,
        aspect_ratio: float = 297/210,
        angle_tolerance: float = 5,
        aspect_ratio_tolerance: float = 0.05,
        sheet_width: float = 297,
        sheet_height: float = 210
) -> List[Dict]:
    """
    Detect white rectangles in image. If aspect_ratio provided, filter and sort by ratio match.

    Args:
        image: Input BGR image
        aspect_ratio: Expected height/width ratio (optional)
        angle_tolerance: Maximum angle deviation from 90°
        aspect_ratio_tolerance: Maximum aspect ratio deviation
        sheet_width: Width of the reference sheet in mm (optional)
        sheet_height: Height of the reference sheet in mm (optional)

    Returns:
        List of rectangles sorted by total score in descending order
    """

    def is_distinguishable(rect: Rectangle, rectangles: List[Rectangle], tolerance_distance: float = 10) -> bool:
        def calculate_similarity_score(rect1, rect2):
            rect1_points = [pt[0] for pt in rect1]
            rect2_points = [pt[0] for pt in rect2]
            total_distance = 0
            matched_indices = set()

            # Find minimum distance for each vertex in rect1 to rect2, without repeating matches
            for pt1 in rect1_points:
                min_distance = float('inf')
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
        image_area = image.shape[0] * image.shape[1]
        min_area = image_area / 9

        # Find all rectangles
        for threshold in range(int(ret), 256, 10):
            _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < min_area:
                    continue

                approx = cv2.approxPolyDP(cnt, cv2.arcLength(cnt, True) * 0.02, True)
                if len(approx) == 4:
                    # Check angles
                    max_angle_deviation = 0
                    for j in range(4):
                        pt1 = approx[j % 4][0]
                        pt2 = approx[(j - 2) % 4][0]
                        pt0 = approx[(j - 1) % 4][0]
                        cosine = abs(vector_angle(pt1, pt2, pt0))
                        angle_degrees = math.degrees(math.acos(cosine))
                        angle_deviation = min(abs(angle_degrees - 90), abs(angle_degrees - 270))
                        max_angle_deviation = max(max_angle_deviation, angle_deviation)

                    # Get dimensions
                    d1 = euclidean_distance(approx[0][0], approx[1][0])
                    d2 = euclidean_distance(approx[1][0], approx[2][0])
                    d3 = euclidean_distance(approx[2][0], approx[3][0])
                    d4 = euclidean_distance(approx[3][0], approx[0][0])

                    width = min(d1, d2, d3, d4)
                    height = max(d1, d2, d3, d4)

                    rect_aspect_ratio = height / width if height > width else width / height
                    aspect_ratio_deviation = abs(rect_aspect_ratio - aspect_ratio) if aspect_ratio is not None else 0

                    # Calculate aspect ratio score
                    if aspect_ratio is not None:
                        aspect_ratio_score = 1 - abs(rect_aspect_ratio - aspect_ratio) / (aspect_ratio + aspect_ratio_tolerance)
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
                    total_score = (0.3 * aspect_ratio_score) + (0.3 * angle_score) + (0.4 * whiteness_score)

                    # Calculate pixels per mm if sheet dimensions are provided
                    pixels_per_mm = None
                    if sheet_width is not None and sheet_height is not None:
                        pixels_per_mm = width / sheet_width if width > height else height / sheet_height

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
                        "pixels_per_mm": pixels_per_mm
                    }

                    if max_angle_deviation <= angle_tolerance and aspect_ratio_deviation <= aspect_ratio_tolerance and is_distinguishable(
                            approx, [r['rectangle'] for r in rectangles]
                    ):
                        rectangles.append(rect_data)

        # Sort rectangles by total score in descending order
        rectangles.sort(key=lambda x: x["total_score"], reverse=True)
        return rectangles

    except Exception as e:
        return []

