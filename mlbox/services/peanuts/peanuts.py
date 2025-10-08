import base64  # Pylint warning: Missing module docstring
import os
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from loguru import logger
from openpyxl.drawing.image import Image as OpenPyxlImage
from openpyxl.styles import Alignment
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage
from mlbox.models.peanuts.cls.yolo_cls_model import YOLOPeanutsClassifier
from mlbox.models.peanuts.detection.yolo_detector_model import YOLOPeanutsDetector
from mlbox.services.peanuts.datatype import Ellipse, Status, PeanutProcessingRequest, PeanutProcessingResult, OnePeanutProcessingResult, BaseResponseJson, PeanutDataResponseJson
from mlbox.settings import ROOT_DIR, LOG_LEVEL
from mlbox.utils.cvtools import preprocess_images_with_white_rectangle
from mlbox.utils.logger import get_logger, get_artifact_service

CURRENT_DIR = Path(__file__).parent
SERVICE_NAME = "Peanuts"
app_logger = get_logger(ROOT_DIR)
artifact_service = get_artifact_service(ROOT_DIR)

# Load environment variables from the peanuts service directory
load_dotenv(CURRENT_DIR / ".env")

#os.environ["CURL_CA_BUNDLE"] = ""
HF_TOKEN = os.getenv("HF_TOKEN")
HF_PEANUT_SEG_REPO_ID = os.getenv("HF_PEANUT_SEG_REPO_ID")
HF_PEANUT_SEG_FILE = os.getenv("HF_PEANUT_SEG_FILE")
HF_PEANUT_CLS_REPO_ID = os.getenv("HF_PEANUT_CLS_REPO_ID")
HF_PEANUT_CLS_FILE = os.getenv("HF_PEANUT_CLS_FILE")
ERP_ENDPOINT = (
    "https://ite.roshen.com:4433/WS/api/_MLBOX_HANDLE_RESPONSE?call_in_async_mode=false"
)



input_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "input"


def process_requests(
    requests: List[PeanutProcessingRequest],
) -> List[PeanutProcessingResult]:

    """
    Processes a list of peanut processing requests and returns the results.

    Args:
        requests (List[PeanutProcessingRequest]): A list of peanut processing requests.

    Returns:
        List[PeanutProcessingResult]: A list of results after processing the peanut requests.

    The function performs the following steps:
    1. Processes the peanut images from the requests.
    2. Formats the result based on client needs.
    3. Prepares an Excel file for each processed result.
    4. Updates the status and error message of each result.
    5. Posts the result to the client via a REST request.

    Each result includes the status, error message, and the filename of the generated Excel file.
    """

    peanut_processing_results = process_peanuts_images(requests)

    for request, peanut_processing_result in zip(requests, peanut_processing_results):

        # Step 2: Format result based on client needs
        excel_file = prepare_excel(peanut_processing_result)
        peanut_processing_result.status = Status.SUCCESS
        peanut_processing_result.error_message = "Successfully processed the request."
        peanut_processing_result.excel_filename = excel_file
        post_rest_request_to_client(
            ERP_ENDPOINT, excel_file, request.alias, request.key, request.image_filename
        )

    return peanut_processing_results
    

def process_peanuts_images(
    requests: List[PeanutProcessingRequest],
) -> List[PeanutProcessingResult]:

    app_logger.info(SERVICE_NAME, f"Processing peanuts images: {len(requests)}")
    peanut_processing_results: List[PeanutProcessingResult] = []

    input_images, input_images_filename = zip(
        *[(np.array(request.image), request.image_filename) for request in requests]
    )

    # prepare images for processing
    try:
        preprocessed_results = preprocess_images_with_white_rectangle(input_images= input_images, target_width = 2000)
        preprocessed_images, pixels_per_mm_values = zip(*preprocessed_results)
    except ValueError as e:
        # If A4 paper detection fails, use simple resizing instead
        print(f"A4 paper detection failed: {e}. Using simple resizing.")
        preprocessed_images = []
        pixels_per_mm_values = []
        for image in input_images:
            # Simple resize to target width while maintaining aspect ratio
            height, width = image.shape[:2]
            aspect_ratio = width / height
            new_height = int(2000 / aspect_ratio)
            resized_image = cv2.resize(image, (2000, new_height))
            preprocessed_images.append(resized_image)
            # Use a default pixels_per_mm value (this is approximate)
            pixels_per_mm_values.append(5.0)  # Default value, may need adjustment

    if app_logger.level == "DEBUG":
        save_images_with_annotations(
            preprocessed_images, step_name="preprocessing", output_folder=artifact_service.get_service_dir(SERVICE_NAME) )

    # load weights file and detect peanuts on the images
    cls_model_path = hf_hub_download(
        repo_id=HF_PEANUT_CLS_REPO_ID, filename=HF_PEANUT_CLS_FILE, token=HF_TOKEN
    )
    classifier = YOLOPeanutsClassifier(cls_model_path)

    # load weights file and detect peanuts on the images
    det_model_path = hf_hub_download(
        repo_id=HF_PEANUT_SEG_REPO_ID, filename=HF_PEANUT_SEG_FILE, token=HF_TOKEN
    )

    detector = YOLOPeanutsDetector(det_model_path)

    sv_detections = detector.detect(
        preprocessed_images, verbose=True, imgsz=1024, conf=0.6
    )

    if LOG_LEVEL == "DEBUG":
        bounding_boxes = [detection.xyxy for detection in sv_detections]
        masks = [detection.mask for detection in sv_detections]
        save_images_with_annotations(
            preprocessed_images,
            step_name="detection",
            output_folder=artifact_service.get_service_dir(SERVICE_NAME),
            bounding_boxes=bounding_boxes,
            masks=masks,
        )

    # for each image and each peanut: crop peanut, fit elipse, classify, fill result
    for input_image_filename, preprocessed_image, sv_detection, pixels_per_mm in zip(
        input_images_filename, preprocessed_images, sv_detections, pixels_per_mm_values
    ):

        peanuts: List[OnePeanutProcessingResult] = []
        one_peanut_images: List[np.ndarray] = []

        sorted_indices = sorted(
            range(len(sv_detection.xyxy)),
            key=lambda idx: (sv_detection.xyxy[idx][1], sv_detection.xyxy[idx][0]),
        )
        ordered_index = 0
        for index in sorted_indices:

            xyxy = sv_detection.xyxy[index]
            mask = sv_detection.mask[index]
            x1, y1, x2, y2 = map(int, xyxy)
            # Crop peanut image
            one_peanut_image = preprocessed_image[y1:y2, x1:x2].copy()

            # Set outside pixels to white
            one_peanut_image[mask[y1:y2, x1:x2] is False] = 255

            # Save the cropped peanut image
            if LOG_LEVEL == "DEBUG":
                pil_image = PILImage.fromarray(one_peanut_image)
                artifact_service.save_artifact(
                    service=SERVICE_NAME,
                    file_name=f"{index}.jpg",
                    data=pil_image
                )

            one_peanut_images.append(one_peanut_image)

            # find contour and fit elipse
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            contour = max(contours, key=cv2.contourArea)
            contour_reshaped = contour.reshape(-1, 2)
            center, axes, angle = cv2.fitEllipse(contour_reshaped)
            axes = (axes[0] * 0.9, axes[1] * 0.9)
            elipse = Ellipse(center=center, axes=axes, angle=angle)
            peanut = OnePeanutProcessingResult(
                index=ordered_index,
                xyxy=xyxy,
                mask=mask,
                contour=contour,
                det_confidence=sv_detection.confidence[index],
                image=PILImage.fromarray(one_peanut_image),
                ellipse=elipse,
            )
            ordered_index = ordered_index + 1

            peanuts.append(peanut)

        sv_cls = classifier.classify(one_peanut_images, verbose=False)

        for index, classification in enumerate(sv_cls):
            peanuts[index].class_id = classification.class_id
            peanuts[index].class_confidence = classification.confidence

        peanut_processing_results.append(
            PeanutProcessingResult(
                peanuts=peanuts,
                weight_g=100,
                pixels_per_mm=pixels_per_mm,
                original_image=PILImage.fromarray(preprocessed_image),
                original_image_filename=input_image_filename,
                status=Status.SUCCESS,
                message="Successfully processed the request",
            )
        )

        # Ensure result folder exists
        
        for result in peanut_processing_results:
            result.result_image.save(
                artifact_service.get_service_dir(SERVICE_NAME) / f"result_{result.original_image_filename}",
                format="JPEG",
            )

    if LOG_LEVEL == "DEBUG":
        classes: List[List[str]] = []
        bounding_boxes: List[List[Tuple[int, int, int, int]]] = []
        masks: List[List[np.ndarray]] = []
        ellipses: List[List[Ellipse]] = []
        contours: List[List[np.ndarray]] = []
        indexes: List[List[int]] = []

        for peanut_result in peanut_processing_results:
            bounding_boxes.append([peanut.xyxy for peanut in peanut_result.peanuts])
            masks.append([peanut.mask for peanut in peanut_result.peanuts])
            classes.append([peanut.class_id for peanut in peanut_result.peanuts])
            ellipses.append([peanut.ellipse for peanut in peanut_result.peanuts])
            contours.append([peanut.contour for peanut in peanut_result.peanuts])
            indexes.append([peanut.index for peanut in peanut_result.peanuts])

        save_images_with_annotations(
            images=preprocessed_images,
            step_name="result",
            indexes=indexes,
            bounding_boxes=bounding_boxes,
            masks=masks,
            classes=classes,
            contours=contours,
            ellipses=ellipses,
            output_folder=artifact_service.get_service_dir(SERVICE_NAME),
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
        artifact_service.get_service_dir(SERVICE_NAME)
        / f"{Path(peanut_processing_result.original_image_filename).stem}.xlsx"
    )

    if excel_file.exists():
        excel_file.unlink()

    # Create Excel writer
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        # Prepare result table
        result_table = pd.DataFrame(
            [
                {
                    "№ п/п": idx,
                    "max діаметр, мм": (
                        round(
                            peanut.ellipse.axes[1]
                            / peanut_processing_result.pixels_per_mm,
                            1,
                        )
                        if peanut.ellipse
                        else None
                    ),
                    "min діаметр, мм": (
                        round(
                            peanut.ellipse.axes[0]
                            / peanut_processing_result.pixels_per_mm,
                            1,
                        )
                        if peanut.ellipse
                        else None
                    ),
                    "клас": YOLOPeanutsClassifier.class_names[peanut.real_class[0]],
                    "впевненність (клас)": round(peanut.real_class[1], 2),
                    "впевненність (маска)": round(peanut.det_confidence, 2),
                }
                for idx, peanut in enumerate(peanut_processing_result.peanuts)
            ]
        )

        # Write the result table to the first sheet
        result_table.to_excel(writer, sheet_name="Peanut Analysis", index=False)

        # Create a DataFrame for the calculated indicators
        indicators_table = pd.DataFrame(
            {
                "Показник": [
                    "Вага зразку, г",
                    "середньо кв.відх. (по 'меньша осі')",
                    "коефіціент варіації (по 'меньша осі'), %",
                    "середньо кв.відх. (по 'більша ось / меньша ось')",
                    "коефіціент варіації (по 'більша ось / меньша ось'), %",
                    "Шт в 1 унції",
                ],
                "Значення": [
                    peanut_processing_result.weight_g,
                    peanut_processing_result.standard_deviation_minor_axe,
                    peanut_processing_result.coefficient_variation_minor_axe,
                    peanut_processing_result.standard_deviation_ratio_axes,
                    peanut_processing_result.coefficient_variation_ratio_axes,
                    round(len(peanut_processing_result.peanuts) * 28.35 / 100, 2),
                ],
            }
        )

        indicators_table_start_col = result_table.shape[1] + 1
        # Write the indicators table to the second sheet starting from column E (5th column)
        indicators_table.to_excel(
            writer,
            sheet_name="Peanut Analysis",
            index=False,
            startcol=indicators_table_start_col,
        )

        # Access the workbook and worksheet
        worksheet = writer.sheets["Peanut Analysis"]

        # Set the alignment for all cells to wrap text

        worksheet.column_dimensions[
            get_column_letter(indicators_table_start_col + 1)
        ].width = 50

        for row in worksheet.iter_rows():
            for cell in row:

                if cell.column == indicators_table_start_col + 1:
                    cell.alignment = Alignment(wrap_text=True)
                else:
                    cell.alignment = Alignment(
                        wrap_text=True, horizontal="center", vertical="center"
                    )

        # Insert the result image into the second sheet
        result_image_sheet = writer.book.create_sheet(title="Result Image")
        image_stream = BytesIO()
        peanut_processing_result.result_image.save(image_stream, format="PNG")
        image_stream.seek(0)
        openpyxl_image = OpenPyxlImage(image_stream)
        openpyxl_image.width = openpyxl_image.width // 2
        openpyxl_image.height = openpyxl_image.height // 2
        result_image_sheet.add_image(openpyxl_image, "A1")

    return excel_file


def post_rest_request_to_client(
    endpoint: str, excel_file: Path, alias: str, key: str, image_filename: str
) -> None:
    """
    Send the generated Excel file to the ERP endpoint.

    Args:
        excel_file (Path): Path to the Excel file to be sent.
        communication_data (Dict[str, str]): Communication details, including alias and alias_key.
    """
    headers = {"accept": "application/xml", "Content-Type": "application/json"}

    peanut_data = PeanutDataResponseJson(
        alias=alias,
        key=key,
        image_filename=image_filename,
        excel_file=base64.b64encode(excel_file.read_bytes()).decode("utf-8"),
    )

    peanut_response = BaseResponseJson(
        status="Success",
        message="Processing completed successfully.",
        service_name="peanuts",
        timestamp=pd.Timestamp.now().isoformat(),
        data=peanut_data.to_json()  # pylint: disable=no-member
    )
    
    # Send the request
    response = httpx.post(endpoint, headers=headers, json={ "responseJson": peanut_response.to_json() }) # pylint: disable=no-member

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

        if masks is not None and len(masks) > i and False:
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

        if ellipses is not None and len(ellipses) > i:
            for ellipse in ellipses[i]:
                center = (int(ellipse.center[0]), int(ellipse.center[1]))
                axes = (int(ellipse.axes[0] / 2), int(ellipse.axes[1] / 2))
                angle = ellipse.angle
                cv2.ellipse(
                    cv_image, center, axes, angle, 0, 360, (255, 0, 0), 2
                )  # Blue color with thickness 2

        if contours is not None and len(contours) > i and False and False:
            for contour in contours[i]:
                cv2.drawContours(
                    cv_image, [contour], -1, (0, 0, 255), 2
                )  # Hard blue color with thickness 2

        # Convert NumPy array back to PIL image
        pil_image = PILImage.fromarray(cv_image)

        #pil_image.save(file_name, format="JPEG")
        #artifact_service.save_artifact(service=SERVICE_NAME,file_name=f"{step_name}_{i}.jpg",data=pil_image)


def preprocessing_images_for_dataset(input_folder: Path, output_folder: Path) -> None:
    """
    Preprocess images from the input folder and save the preprocessed images to the output folder.

    Args:
        input_folder (Path): The folder containing the raw input images.
        output_folder (Path): The folder to save the preprocessed images.
    """
    # Gather all image files from the input folder
    output_files = {file.name for file in output_folder.glob("*.*")}
    image_files = [file for file in input_folder.glob("*.*") if file.name not in output_files]

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each image one at a time
    for img_file in image_files:
        # Read the image and convert to numpy array
        input_image = np.array(PILImage.open(img_file))

        # Preprocess the image
        try:
            preprocessed_result = preprocess_images_with_white_rectangle(input_images=[input_image], target_width=2000)
        except ValueError as e:
            print(f"{e}")
            continue

        preprocessed_image = preprocessed_result[0][0]

        # Save the preprocessed image to the output folder
        output_path = output_folder / img_file.name
        PILImage.fromarray(preprocessed_image).save(output_path, format="JPEG")



def test_process_requests(input_folder: Path, output_folder: Path):
    """
    Test the process_requests function by gathering image files from the input folder,
    preparing PeanutProcessingRequest objects, and calling the main processing function.
    """

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
                image=img,
                image_filename=img_file.name,
                alias="DMS",
                key="   9127673     1",
                response_method="HTTP_POST_REQUEST",
                response_endpoint="https://ite.roshen.com:4433/WS/api/_MLBOX_HANDLE_RESPONSE?call_in_async_mode=false",
            )
        )

    # Call the main processing function with the prepared requests
    process_requests(requests)




if __name__ == "__main__":
    
    #output_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/preprocessed")
    #input_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/Experiments-photo-lab/2025-02-19-phones")
    #preprocessing_images_for_dataset(input_folder, output_folder)




    input_folder = Path(r"/home/polovyi/projects/mlbox/assets/datasets/peanut/2025-10-08-test-measurement")
    output_folder = input_folder / "output"
    
    
    test_process_requests(input_folder, output_folder)
