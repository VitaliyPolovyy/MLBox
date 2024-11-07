import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def generate_random_color():
    """Generate a random color for each mask."""
    return tuple(random.randint(0, 255) for _ in range(3))


import cv2


def apply_mask(image, mask, color, alpha):
    # Ensure the mask matches the size of the image
    mask_resized = cv2.resize(
        mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    overlay = image.copy()
    overlay[mask_resized == 1] = color  # Apply color where mask is True
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def process_image(model, image_path, result_folder, alpha):
    """Process a single image and save the result."""
    image = cv2.imread(str(image_path))
    results = model.predict(str(image_path), verbose=False)

    # print#(results[0].masks.data)

    # Iterate over masks in the first result
    for mask in results[0].masks.data:
        color = generate_random_color()
        image = apply_mask(image, mask.to(torch.uint8).cpu().numpy(), color, alpha)

    # Save result image
    result_path = result_folder / f"{image_path.stem}_result{image_path.suffix}"
    cv2.imwrite(str(result_path), image)
    print(f"Saved: {result_path}")


def main(image_folder, model_path, result_folder, alpha=0.5):
    """Main function to process all images sequentially."""
    # Load YOLO model
    model = YOLO(model_path)

    # Ensure result folder exists
    Path(result_folder).mkdir(parents=True, exist_ok=True)

    # Get all JPG images in folder
    image_paths = list(Path(image_folder).glob("*.jpg"))

    # Process images sequentially
    for img in image_paths:
        process_image(model, img, Path(result_folder), alpha)


if __name__ == "__main__":
    # Parameters
    image_folder = Path(
        r"C:\My storage\Python projects\DataSets\peanuts\task3\test_model"
    )  # Path to the images folder
    model_path = Path(
        r"C:\My storage\Python projects\MLBox\models\Yolo\best.pt"
    )  # Path to the YOLOv8 model file
    result_folder = image_folder / "result"
    alpha = 0.7  # Transparency level of the mask overlay

    main(image_folder, model_path, result_folder, alpha)
