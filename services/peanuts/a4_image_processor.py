import cv2
import os
import numpy as np
import math
from typing import Optional, Tuple
from utils.cvtools import detect_white_rectangles
from tqdm import tqdm


def process_images(
        input_folder: str,
        output_folder: str,
        target_width: Optional[int] = None,
        padding_percent: float = 0.01,
        a4_ratio: float = 297 / 210
) -> None:
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not os.path.exists(os.path.join(output_folder, f))]

    with tqdm(total=len(filenames), desc="Processing Images") as pbar:
        for filename in filenames:
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                pbar.set_postfix_str(f"Couldn't load {filename}")
                pbar.update(1)
                continue
            result = process_single_image(image, target_width, padding_percent, a4_ratio)
            if result is None:
                pbar.set_postfix_str(f"No A4 paper found in {filename}")
                pbar.update(1)
                continue
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, result)
            pbar.set_postfix_str(f"Processed and saved: {filename}")
            pbar.update(1)


def process_single_image(
        image: np.ndarray,
        target_width: Optional[int],
        padding_percent: float,
        aspect_ratio: float
) -> Optional[np.ndarray]:
    rectangles = detect_white_rectangles(image, aspect_ratio=aspect_ratio)
    if not rectangles:
        return None
    best_rectangle = rectangles[0]["rectangle"]
    rect_points = best_rectangle.reshape(4, 2)
    angle = get_rotation_angle(rect_points)
    rotated, rect_points_rotated = rotate_image(image, rect_points, angle)
    result = crop_and_resize(rotated, rect_points_rotated, target_width, padding_percent)
    return result


def get_rotation_angle(rect_points: np.ndarray) -> float:
    # Find the longest side and calculate its angle with horizontal
    d1 = np.linalg.norm(rect_points[0] - rect_points[1])
    d2 = np.linalg.norm(rect_points[1] - rect_points[2])
    pts = (rect_points[0], rect_points[1]) if d1 > d2 else (rect_points[1], rect_points[2])
    dx = pts[1][0] - pts[0][0]
    dy = pts[1][1] - pts[0][1]
    angle = math.degrees(math.atan2(dy, dx))
    return 180 - angle


def rotate_image(
        image: np.ndarray,
        rect_points: np.ndarray,
        angle: float
) -> Tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
    rect_points_rotated = cv2.transform(np.array([rect_points]), M)[0]
    return rotated, rect_points_rotated


def crop_and_resize(
        image: np.ndarray,
        rect_points: np.ndarray,
        target_width: Optional[int],
        padding_percent: float
) -> np.ndarray:
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

    if target_width is not None:
        aspect_ratio = cropped.shape[1] / cropped.shape[0]
        new_height = int(target_width / aspect_ratio)
        return cv2.resize(cropped, (target_width, new_height))
    return cropped


if __name__ == "__main__":
    INPUT_FOLDER = r"C:\My storage\Python projects\DataSets\peanuts\raw"
    OUTPUT_FOLDER = r"C:\My storage\Python projects\DataSets\peanuts"
    TARGET_WIDTH = None
    PADDING_PERCENT = 0.01
    A4_RATIO = 297 / 210
    process_images(INPUT_FOLDER, OUTPUT_FOLDER, TARGET_WIDTH, PADDING_PERCENT, A4_RATIO)
