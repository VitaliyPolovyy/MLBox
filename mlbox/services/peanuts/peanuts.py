import base64
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from loguru import logger
from openpyxl.drawing.image import Image as OpenPyxlImage
from PIL import Image as PILImage
from PIL import ImageDraw
from ultralytics import YOLO  # Import YOLO for model loading and prediction

from mlbox.models.peanuts.cls.yolo_cls_model import YOLOPeanutsClassifier
from mlbox.models.peanuts.detection.yolo_detector_model import \
    YOLOPeanutsDetector
from mlbox.services.peanuts.datatype import (Ellipse,
                                             OnePeanutProcessingResult,
                                             PeanutProcessingRequest,
                                             PeanutProcessingResult)
from mlbox.settings import DEBUG_MODE, ROOT_DIR
from mlbox.utils.cvtools import preprocess_images_with_white_rectangle

CURRENT_DIR = Path(__file__).parent

load_dotenv()

os.environ["CURL_CA_BUNDLE"] = ""
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("PEANUT_HF_REPO_ID")
HF_YOLO_DETECTION_FILE = os.getenv("PEANUT_HF_MODEL_FILE")
HF_YOLO_CLS_FILE = os.getenv("PEANUT_HF_MODEL_FILE")
# IMAGE_SIZE = int(os.getenv("PEANUT_IMAGE_SIZE"))
ERP_ENDPOINT = (
    "https://ite.roshen.com:4433/WS/api/_MLBOX_HANDLE_RESPONSE?call_in_async_mode=false"
)
DEBUG_MODE = os.getenv("DEBUG_MODE")

# Configure loguru
logger.remove()


input_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "input"
result_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "output"


def process_requests(requests: List[PeanutProcessingRequest]) -> None:

    # Step 1: Process peanut requests and get results
    peanut_processing_results = process_peanuts_images(requests)

    for request, peanut_processing_result in zip(requests, peanut_processing_results):
        # Step 2: Format result based on client needs
        if request.service_code == "PEANUT":
            excel_file = prepare_excel(peanut_processing_result)
            post_rest_request_to_client(
                ERP_ENDPOINT, excel_file, request.alias, request.key
            )


def process_peanuts_images(
    requests: List[PeanutProcessingRequest],
) -> List[PeanutProcessingResult]:

    peanut_processing_results: List[PeanutProcessingResult] = []

    input_images = [np.array(request.image) for request in requests]

    # prepare images for processing
    preprocessed_results = preprocess_images_with_white_rectangle(input_images)

    preprocessed_images, pixels_per_mm_values = zip(*preprocessed_results)

    if DEBUG_MODE:
        save_images_with_annotations(
            preprocessed_images, step_name="preprocessing", output_folder=result_folder
        )

    # load weights file and detect peanuts on the images
    cls_model_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_YOLO_CLS_FILE, token=HF_TOKEN
    )
    classifier = YOLOPeanutsClassifier(cls_model_path)

    # load weights file and detect peanuts on the images
    det_model_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_YOLO_DETECTION_FILE, token=HF_TOKEN
    )
    detector = YOLOPeanutsDetector(det_model_path)
    sv_detections = detector.detect(preprocessed_images, verbose=False)

    if DEBUG_MODE:
        bounding_boxes = [detection.xyxy for detection in sv_detections]
        masks = [detection.mask for detection in sv_detections]
        save_images_with_annotations(
            preprocessed_images,
            step_name="detection",
            output_folder=result_folder,
            bounding_boxes=bounding_boxes,
            masks=masks,
        )

    # for each image and each peanut: crop peanut, fit elipse, classify, fill result
    for preprocessed_image, sv_detection, pixels_per_mm in zip(
        preprocessed_images, sv_detections, pixels_per_mm_values
    ):

        peanuts: List[OnePeanutProcessingResult] = []
        one_peanut_images: List[np.ndarray] = []

        for index in range(len(sv_detection.xyxy)):

            xyxy = sv_detection.xyxy[index]
            mask = sv_detection.mask[index]
            x1, y1, x2, y2 = map(int, xyxy)
            # Crop peanut image
            one_peanut_image = preprocessed_image[y1:y2, x1:x2].copy()

            inverted_mask = cv2.bitwise_not(mask[y1:y2, x1:x2].astype(np.uint8) * 255)

            # Set outside pixels to white
            one_peanut_image[mask[y1:y2, x1:x2] == False] = 255

            # bit AND operation between mask and peanut image
            # one_peanut_image = cv2.bitwise_and(one_peanut_image, one_peanut_image, mask = mask[y1:y2, x1:x2].astype(np.uint8)*255)

            # Save the cropped peanut image
            pil_image = PILImage.fromarray(one_peanut_image)
            file_name = result_folder / f"{index}.jpg"
            pil_image.save(file_name, format="JPEG")

            one_peanut_images.append(one_peanut_image)

            # find contour and fit elipse
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = max(contours, key=cv2.contourArea)
            center, axes, angle = cv2.fitEllipse(contour)
            elipse = Ellipse(center=center, axes=axes, angle=angle)
            rotated_bbox = cv2.minAreaRect(contour)  # (center), (width, height), angle
            peanut = OnePeanutProcessingResult(
                index=index,
                xyxy=xyxy,
                mask=mask,
                contour=contour,
                det_confidence=sv_detection.confidence[index],
                image=PILImage.fromarray(one_peanut_image),
                rotated_bbox=rotated_bbox,
                ellipse=elipse,
            )

            peanuts.append(peanut)

        """
        sv_cls = classifier.classify(one_peanut_images)

        for index in range(len(sv_cls)):
            # Assuming sv_cls contains Detections with class_id and confidence
            peanuts[index].class_id = sv_cls[index].class_id
            peanuts[index].class_confidence = sv_cls[index].confidence
        """

        peanut_processing_results.append(
            PeanutProcessingResult(peanuts=peanuts, pixels_per_mm=pixels_per_mm)
        )
    if DEBUG_MODE:
        classes: List[List[str]] = []
        bounding_boxes: List[List[Tuple[int, int, int, int]]] = []
        masks: List[List[np.ndarray]] = []
        ellipses: List[List[Ellipse]] = []
        contours: List[List[np.ndarray]] = []
        indexes: List[List[int]] = []
        rotated_bboxes: List[cv2.typing.RotatedRect] = []

        for peanut_result in peanut_processing_results:
            bounding_boxes.append([peanut.xyxy for peanut in peanut_result.peanuts])
            masks.append([peanut.mask for peanut in peanut_result.peanuts])
            classes.append([peanut.class_id for peanut in peanut_result.peanuts])
            ellipses.append([peanut.ellipse for peanut in peanut_result.peanuts])
            contours.append([peanut.contour for peanut in peanut_result.peanuts])
            indexes.append([peanut.index for peanut in peanut_result.peanuts])
            rotated_bboxes.append(
                [peanut.rotated_bbox for peanut in peanut_result.peanuts]
            )

        save_images_with_annotations(
            images=preprocessed_images,
            step_name="result",
            indexes=indexes,
            bounding_boxes=bounding_boxes,
            masks=masks,
            classes=classes,
            contours=contours,
            ellipses=ellipses,
            rotated_bboxes=rotated_bboxes,
            output_folder=result_folder,
        )

    return peanut_processing_results


def prepare_excel(peanut_processing_result: PeanutProcessingResult) -> Path:
    """
    Create an Excel file for the given PeanutProcessingResult.

    Args:
        peanut_processing_result (PeanutProcessingResult): The result to prepare the Excel file for.

    Returns:
        Path: The path to the generated Excel file.
    """
    # Define output file path
    excel_file = (
        result_folder
        / f"peanut_analysis_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
    )

    # Create Excel writer
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        # Prepare result table
        result_table = pd.DataFrame(
            [
                {
                    "Peanut Index": idx + 1,
                    "Major Axis (mm)": (
                        peanut.ellipse.axes[0] / peanut_processing_result.pixels_per_mm
                        if peanut.ellipse
                        else None
                    ),
                    "Minor Axis (mm)": (
                        peanut.ellipse.axes[1] / peanut_processing_result.pixels_per_mm
                        if peanut.ellipse
                        else None
                    ),
                    "Angle (degrees)": peanut.ellipse.angle if peanut.ellipse else None,
                }
                for idx, peanut in enumerate(peanut_processing_result.peanuts)
            ]
        )

        # Write the result table to the first sheet
        result_table.to_excel(writer, sheet_name="Peanut Analysis", index=False)

    return excel_file


def post_rest_request_to_client(endpoint: str, excel_file: Path, alias: str, key: str):
    """
    Send the generated Excel file to the ERP endpoint.

    Args:
        excel_file (Path): Path to the Excel file to be sent.
        communication_data (Dict[str, str]): Communication details, including alias and alias_key.
    """
    headers = {"accept": "application/xml", "Content-Type": "application/json"}

    # Prepare result JSON payload
    result_json = {
        "Status": "Success",
        "Message": "Processing completed successfully.",
        "ServiceName": "peanut",
        "alias": alias,
        "alias_key": key,
        "Timestamp": datetime.now().isoformat(),
    }

    # Prepare the service response payload
    result_service_json = {
        "alias": alias,
        "key": key,
        "excel_file": base64.b64encode(excel_file.read_bytes()).decode("utf-8"),
    }

    payload = {
        "resultjson": json.dumps(result_json),
        "serviceresponsejson": json.dumps(result_service_json),
    }

    # Send the request
    response = httpx.post(endpoint, headers=headers, json=payload)

    # Check response status
    if response.status_code == 200:
        print("Successfully sent the result to ERP.")
    else:
        print(
            f"Failed to send the result to ERP. Status code: {response.status_code}, Response: {response.text}"
        )


def save_images_with_annotations(
    images: List[np.ndarray],
    step_name: str,
    output_folder: Path,
    bounding_boxes: Optional[List[List[Tuple[int, int, int, int]]]] = None,
    masks: Optional[List[List[np.ndarray]]] = None,
    contours: Optional[List[List[np.ndarray]]] = None,
    ellipses: Optional[List[List[Ellipse]]] = None,
    indexes: Optional[List[List[int]]] = None,
    classes: Optional[List[List[str]]] = None,
    rotated_bboxes: Optional[List[List[cv2.typing.RotatedRect]]] = None,
) -> None:

    output_folder.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        cv_image = img.copy()

        if bounding_boxes is not None and len(bounding_boxes) > i and False:
            for j, box in enumerate(bounding_boxes[i]):
                cv2.rectangle(
                    cv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2
                )
                label = (
                    f"{indexes[i][j]}"
                    if indexes is not None and len(indexes) > i
                    else ""
                )
                if (
                    classes is not None
                    and len(classes) > i
                    and classes[i][j] is not None
                ):
                    label += f" {classes[i][j]}"
                cv2.putText(
                    cv_image,
                    label,
                    (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        if masks is not None and len(masks) > i:
            for mask in masks[i]:
                # Create an RGBA image with the mask and set the color to green with 50% transparency
                pil_mask = PILImage.fromarray(mask).convert("L")
                green_mask = PILImage.new(
                    "RGBA", pil_mask.size, (245, 0, 70, 255)
                )  # Green with 50% transparency
                pil_mask_rgba = PILImage.composite(
                    green_mask, PILImage.new("RGBA", pil_mask.size), pil_mask
                )

                # Convert RGBA mask to BGR format for OpenCV
                mask_bgr = np.array(pil_mask_rgba.convert("RGB"))
                mask_bgr = cv2.cvtColor(mask_bgr, cv2.COLOR_RGB2BGR)

                # Overlay the mask on the original image
                cv_image = cv2.addWeighted(cv_image, 1, mask_bgr, 0.9, 0)

        if ellipses is not None and len(ellipses) > i and False:
            for ellipse in ellipses[i]:
                center = (int(ellipse.center[0]), int(ellipse.center[1]))
                axes = (int(ellipse.axes[0] / 2), int(ellipse.axes[1] / 2))
                angle = ellipse.angle
                cv2.ellipse(
                    cv_image, center, axes, angle, 0, 360, (255, 0, 0), 2
                )  # Blue color with thickness 2

        if rotated_bboxes is not None and len(rotated_bboxes) > i:
            for rotated_bbox in rotated_bboxes[i]:
                box = cv2.boxPoints(rotated_bbox)
                box = np.int0(box)
                cv2.drawContours(cv_image, [box], 0, (0, 155, 155), 2)

        if contours is not None and len(contours) > i:
            for contour in contours[i]:
                cv2.drawContours(
                    cv_image, [contour], -1, (0, 0, 255), 2
                )  # Hard blue color with thickness 2

        # Convert NumPy array back to PIL image
        pil_image = PILImage.fromarray(cv_image)

        file_name = output_folder / f"{step_name}_{i}.jpg"
        pil_image.save(file_name, format="JPEG")


def test_process_requests():

    # Gather all image files from the input folder
    image_files = list(
        input_folder.glob("*.*")
    )  # You can filter by extension, e.g. "*.jpg" if needed

    # Prepare requests by reading each image
    requests = []
    for img_file in image_files:
        img = PILImage.open(img_file)
        requests.append(
            PeanutProcessingRequest(
                service_code="PEANUT", image=img, alias="DMS", key="11"
            )
        )

    # Call the main processing function with the prepared requests
    process_requests(requests)


if __name__ == "__main__":
    # Run the test
    test_process_requests()
