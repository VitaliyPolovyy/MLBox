import os
import sys
from pathlib import Path
from typing import List

import numpy as np
import PIL.Image as PILImage
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from mlbox.models.peanuts.detection.yolo_detector_model import YOLOPeanutsDetector
from mlbox.settings import DEBUG_MODE, ROOT_DIR
from mlbox.utils.cvtools import (
    detect_white_rectangles,
    preprocess_images_with_white_rectangle,
)

CURRENT_DIR = Path(__file__).parent

load_dotenv()

os.environ["CURL_CA_BUNDLE"] = ""
HF_TOKEN = os.getenv("HF_TOKEN")
HF_REPO_ID = os.getenv("PEANUT_HF_REPO_ID")
HF_YOLO_DETECTION_FILE = os.getenv("PEANUT_HF_MODEL_FILE")


input_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "input"
result_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "output"

if __name__ == "__main__":

    image_files = list(input_folder.glob("*.*"))

    input_images = [np.array(PILImage.open(image)) for image in image_files]

    # prepare images for processing
    preprocessed_results = preprocess_images_with_white_rectangle(input_images)

    preprocessed_images, pixels_per_mm_values = zip(*preprocessed_results)

    # load weights file and detect peanuts on the images
    det_model_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_YOLO_DETECTION_FILE, token=HF_TOKEN
    )
    detector = YOLOPeanutsDetector(det_model_path)
    sv_detections = detector.detect(preprocessed_images, verbose=False)

    # for each image and each peanut: crop peanut, fit elipse, classify, fill result
    for preprocessed_image, sv_detection, image_file in zip(
        preprocessed_images, sv_detections, image_files
    ):

        one_peanut_images: List[np.ndarray] = []

        for index in range(len(sv_detection.xyxy)):

            xyxy = sv_detection.xyxy[index]
            mask = sv_detection.mask[index]
            x1, y1, x2, y2 = map(int, xyxy)
            # Crop peanut image
            one_peanut_image = preprocessed_image[y1:y2, x1:x2].copy()

            # inverted_mask = cv2.bitwise_not(mask[y1:y2, x1:x2].astype(np.uint8) * 255)

            # Set outside pixels to white
            # one_peanut_image[mask[y1:y2, x1:x2] == False] = 255

            # bit AND operation between mask and peanut image
            # one_peanut_image = cv2.bitwise_and(one_peanut_image, one_peanut_image, mask = mask[y1:y2, x1:x2].astype(np.uint8)*255)

            # Save the cropped peanut image
            pil_image = PILImage.fromarray(one_peanut_image)
            file_name = result_folder / f"{image_file.stem}_{index}.jpg"
            pil_image.save(file_name, format="JPEG")
