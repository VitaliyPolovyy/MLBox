import base64
import json
from dotenv import load_dotenv
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from openpyxl.drawing.image import Image as OpenPyxlImage
from PIL import Image as PILImage
from PIL import ImageDraw
from ultralytics import YOLO  # Import YOLO for model loading and prediction

from mlbox.settings import ROOT_DIR
from mlbox.utils.cvtools import detect_white_rectangles

CURRENT_DIR = Path(__file__).parent
os.environ["CURL_CA_BUNDLE"] = ""


load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("PEANUT_HF_REPO_ID")
HF_MODEL_FILE = os.getenv("PEANUT_HF_MODEL_FILE")
IMAGE_SIZE = int(os.getenv("PEANUT_IMAGE_SIZE"))


image_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "input"
result_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "output"


def process_image(input_image_file, alias, alias_key) -> str:
    input_image = PILImage.open(input_image_file).convert("RGB")

    # Step 1: detect a4 and crop it
    preprocessed_image, pixels_per_mm = preprocessing_image_for_detection(input_image)

    # TODO: add flag to enable / disable saving preprocessed image
    preprocessed_image.save(
        result_folder
        / f"{input_image_file.stem}_preprocessed_image{input_image_file.suffix}"
    )

    # Step 2: Run predictions using instance segmentation
    detection_result = peanut_detection(preprocessed_image)

    # Step 3: Assign random class IDs and accuracy (placeholder)
    for i in range(len(detection_result[0].boxes)):
        detection_result[0].boxes[i]["class_id"] = random.randint(0, 2)
        detection_result[0].boxes[i]["class_accuracy"] = round(
            random.uniform(0.5, 1.0), 2
        )

    # Step 4: Prepare results
    result_table, data, result_image = prepare_result(
        detection_result, pixels_per_mm, input_photo=preprocessed_image
    )

    # Step 5: Create the Excel report
    excel_file = prepare_excel(result_table, data, result_image)

    # Step 6: Send results to ERP (optional)
    send_result_to_ERP(excel_file, alias, alias_key)

    return "Processing complete"


def preprocessing_image_for_detection(input_photo):
    def rotate_image(
        image: np.ndarray, rect_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        d1 = np.linalg.norm(rect_points[0] - rect_points[1])
        d2 = np.linalg.norm(rect_points[1] - rect_points[2])
        pts = (
            (rect_points[0], rect_points[1])
            if d1 > d2
            else (rect_points[1], rect_points[2])
        )
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        angle = 180 - np.degrees(np.arctan2(dy, dx))

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
        padding_percent: float,
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

    # Convert Pillow image to NumPy array for OpenCV processing
    image = np.array(input_photo)
    a4_ratio = 297 / 210  # A4 aspect ratio
    target_width = None  # No resizing in this step
    padding_percent = 0.01  # Padding percentage

    # Detect A4 paper
    rectangles = detect_white_rectangles(image, aspect_ratio=a4_ratio)
    if not rectangles:
        raise ValueError("No A4 paper detected in the image.")

    rect_points = np.array([point[0] for point in rectangles[0]["rectangle"]])
    pixels_per_mm = rectangles[0]["pixels_per_mm"]

    # Rotate image
    rotated, rect_points_rotated = rotate_image(image, rect_points)

    # Crop and resize
    result = crop_and_resize(
        rotated, rect_points_rotated, target_width, padding_percent
    )

    return PILImage.fromarray(result), pixels_per_mm


def calculate_ellipse_parameters(mask):
    """Calculate ellipse parameters from a binary mask."""
    # Convert mask to binary
    binary_mask = (mask > 0).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Get the largest contour (assuming the main object)
    contour = max(contours, key=cv2.contourArea)

    # Fit ellipse
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (center, (minor_axis, major_axis), angle) = ellipse
        return {
            "center": center,
            "minor_diameter": minor_axis,
            "major_diameter": major_axis,
            "angle": angle,
        }
    return None


def peanut_detection(image):
    if not all([HF_TOKEN, HF_REPO_ID, HF_MODEL_FILE]):
        raise ValueError("Missing required environment variables (huggingface hub)")

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_MODEL_FILE, token=HF_TOKEN
    )

    confidence_threshold = 0.5  # Confidence threshold for predictions

    # Initialize YOLO model with the specified model path
    model = YOLO(model_path)

    # Process image size and padding
    image_width, image_height = image.size
    aspect_ratio = min(IMAGE_SIZE / image_width, IMAGE_SIZE / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)
    resized_image = image.resize((new_width, new_height))

    # Create padded image
    padded_image = PILImage.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    top_left_x = (IMAGE_SIZE - new_width) // 2
    top_left_y = (IMAGE_SIZE - new_height) // 2
    padded_image.paste(resized_image, (top_left_x, top_left_y))

    # Run prediction
    results = model.predict(
        np.array(padded_image), conf=confidence_threshold, verbose=False
    )

    # Create custom detection results
    custom_boxes = []

    if results and len(results) > 0 and results[0].masks is not None:
        for i, (box, mask) in enumerate(
            zip(results[0].boxes.xyxy, results[0].masks.data)
        ):
            # Extract bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, box)

            # Crop the bounding box from the padded image
            cropped_image = padded_image.crop((x_min, y_min, x_max, y_max))

            # Convert mask to numpy array and crop to bounding box
            mask_np = mask.cpu().numpy()
            mask_cropped = mask_np[y_min:y_max, x_min:x_max]

            # Calculate area and ellipse parameters
            area = (x_max - x_min) * (y_max - y_min)

            # Get ellipse parameters
            ellipse_params = calculate_ellipse_parameters(mask_cropped)

            # Create detection item
            detection_item = {
                "xyxy": box,
                "mask": mask_cropped,
                "cropped_image": cropped_image,
                "area": area,
                "class_id": random.randint(0, 2),  # Placeholder for classification
                "detection_accuracy": float(results[0].boxes.conf[i]),
                "class_accuracy": round(
                    random.uniform(0.5, 1.0), 2
                ),  # Placeholder for accuracy
                "minor_diameter": (
                    ellipse_params["minor_diameter"] if ellipse_params else 0
                ),
                "major_diameter": (
                    ellipse_params["major_diameter"] if ellipse_params else 0
                ),
                "ellipse_angle": ellipse_params["angle"] if ellipse_params else 0,
            }

            custom_boxes.append(detection_item)
    else:
        print("No masks found for image")

    # Wrap custom_boxes in a list to match the original structure
    return [type("DetectionResult", (), {"boxes": custom_boxes})]


def prepare_result(
    detection_result: List[Dict], pixels_per_mm: float, input_photo: PILImage
) -> Tuple[pd.DataFrame, Dict, PILImage.Image]:
    # Initialize lists to store data for the result table
    peanut_index = []
    areas_mm2 = []
    classification_accuracies = []

    # Create a copy of the input photo for annotation
    annotated_image = input_photo.copy()
    draw = ImageDraw.Draw(annotated_image)

    # Iterate through the detection results to extract data
    for index, detection in enumerate(detection_result[0].boxes, start=1):
        class_id = detection["class_id"]
        class_accuracy = detection["class_accuracy"]
        areas_mm2.append(detection["area"] / (pixels_per_mm**2))
        peanut_index.append(index)

        box = detection["xyxy"]  # Bounding box coordinates (x_min, y_min, x_max, y_max)
        classification_accuracies.append(class_accuracy)

    # Prepare the result table as a pandas DataFrame
    result_table = pd.DataFrame(
        {
            "Peanut Index": peanut_index,
            "Area (mm^2)": areas_mm2,
            "Classification Accuracy": classification_accuracies,
        }
    )

    # Prepare additional data dictionary
    data = {
        "total_objects": len(peanut_index),
        "average_area_mm2": np.mean(areas_mm2) if areas_mm2 else 0,
    }

    return result_table, data, annotated_image


def prepare_excel(result_table, data, result_image):
    # Create Excel writer
    excel_file = result_folder / f"peanut_analysis.xlsx"

    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        # Write result table to first sheet
        result_table.to_excel(writer, sheet_name="Peanut Analysis", index=False)

        # Write summary data to 'Peanut Analysis' sheet
        summary_df = pd.DataFrame.from_dict(data, orient="index", columns=["Value"])

        summary_df.to_excel(
            writer,
            sheet_name="Peanut Analysis",
            index=True,
            startcol=len(result_table.columns) + 2,
        )

        # Save result image as a worksheet (optional)
        try:
            img_sheet = writer.book.create_sheet(title="Result Image")
            img_path = result_folder / "result_image.jpg"
            result_image.save(img_path, format="JPEG")

            # Add image to Excel
            img_obj = OpenPyxlImage(img_path)
            img_sheet.add_image(img_obj, "A1")
        except Exception as e:
            print(f"Could not add image to Excel: {e}")

    return excel_file


# def send_result_to_ERP(result_table: pd.DataFrame, data: Dict, excel_file: str):
def send_result_to_ERP(excel_file: Path, alias : str, alias_key: str):
    url = "https://ite.roshen.com:4433/WS/api/_MLBOX_HANDLE_RESPONSE?call_in_async_mode=false"
    headers = {"accept": "application/xml", "Content-Type": "application/json"}

    # result_table_json = result_table.to_dict(orient="records")

    result_json = {
        "Status": "Success",
        "Message": "Processing completed successfully.",
        "ServiceName": "peanut",
        "alias": alias,
        "alias_key": alias_key,
        "Timestamp": datetime.now().isoformat(),
    }

    # Prepare the result_service_json payload
    result_service_json = {
        "alias": alias,
        "key": alias_key,
        "excel_file": base64.b64encode(excel_file.read_bytes()).decode("utf-8"),
    }

    # Prepare the payload similar to the working cURL command
    payload = {
        "resultjson": json.dumps(result_json),
        "serviceresponsejson": json.dumps(result_service_json),
    }

    # Send the request
    response = httpx.post(url, headers=headers, data=json.dumps(payload))

    # Check response status 
    if response.status_code == 200:
        print("Successfully sent the result to ERP.")
    else:
        print(
            f"Failed to send the result to ERP. Status code: {response.status_code}, Response: {response.text}"
        )


"""
if __name__ == "__main__":
    result_folder.mkdir(parents=True, exist_ok=True)

    image_files = list(image_folder.glob("*.jpg"))

    if not image_files:
        exit(1)

    for image_file in image_files:
        process_image(image_file)
"""