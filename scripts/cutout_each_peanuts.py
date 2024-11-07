import json
import os
from pathlib import Path

import cv2
import numpy as np

# Paths
input_folder = r"C:\My storage\Python projects\DataSets\peanuts\task1"
annotations_file = os.path.join(input_folder, "instances_default.json")
output_folder = os.path.join(input_folder, "dataset2")
os.makedirs(output_folder, exist_ok=True)

# Load annotations
with open(annotations_file) as f:
    coco_data = json.load(f)

# Load images
images_info = {img["id"]: img for img in coco_data["images"]}
annotations = coco_data["annotations"]

# Track current image ID and peanut index
current_image_id = None
peanut_index = 0

# Process each annotation
for annotation in annotations:
    image_id = annotation["image_id"]
    if image_id != current_image_id:
        current_image_id = image_id
        peanut_index = 1  # Restart peanut index for new image
    else:
        peanut_index += 1

    image_info = images_info[image_id]

    # Method 1: Using startswith()
    print(image_info["file_name"])
    if image_info["file_name"].startswith("D21010101000001__14972__V223-011__004"):
        continue

    image_path = os.path.join(input_folder, image_info["file_name"])

    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        continue

    # Get mask from annotation
    mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
    segmentation = annotation["segmentation"][0]
    points = np.array(segmentation).reshape((-1, 2))
    cv2.fillPoly(mask, [np.int32(points)], 255)

    # Extract the peanut using the mask
    peanut_image = cv2.bitwise_and(image, image, mask=mask)

    # Calculate the average grey value for the peanut area
    gray_image = cv2.cvtColor(peanut_image, cv2.COLOR_BGR2GRAY)
    peanut_pixels = gray_image[mask == 255]
    avg_gray_value = int(np.mean(peanut_pixels))

    # Get bounding box
    x, y, w, h = annotation["bbox"]
    x, y, w, h = int(x), int(y), int(w), int(h)
    cropped_peanut = peanut_image[y : y + h, x : x + w]

    # Generate output file name
    original_name = Path(image_info["file_name"]).stem
    output_file_name = f"{avg_gray_value}_{original_name}__{peanut_index:03}.png"
    output_path = os.path.join(output_folder, output_file_name)

    # Save the cropped peanut image
    cv2.imwrite(output_path, cropped_peanut)
    print(f"Saved: {output_path}")
