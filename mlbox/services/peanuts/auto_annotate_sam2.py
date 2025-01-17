import gc
import json
import math
import os
import random

import cv2
import numpy as np
import torch
from matplotlib.path import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tqdm import tqdm

from mlbox.utils.cvtools import (
    combine_coco_annotation_files,
    detect_white_rectangles,
    masks_to_coco_annotations,
)


def save_image_with_result(
    image, detected_ellipses, grid_points, image_path, output_path
):
    image_with_all_ellipses = image.copy()

    for obj in detected_ellipses:
        ellipse = obj["ellipse"]
        score = obj["score"]
        accuracy = obj["accuracy"]
        point = obj["point"]
        # Generate a random color with transparency for each ellipse
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            100,
        )  # Add alpha for transparency
        overlay = image_with_all_ellipses.copy()
        # Draw and fill the ellipse on the overlay
        cv2.ellipse(
            overlay, ellipse, color[:3], -1
        )  # Use only RGB values for OpenCV drawing
        # Add transparency to the filled ellipse
        cv2.addWeighted(
            overlay, 0.4, image_with_all_ellipses, 0.6, 0, image_with_all_ellipses
        )
        # Draw the outline of the ellipse
        cv2.ellipse(
            image_with_all_ellipses, ellipse, (0, 0, 0), 2
        )  # Draw the ellipse outline in black
        # Draw the point as a cross
        cross_size = 5
        cv2.line(
            image_with_all_ellipses,
            (point[0] - cross_size, point[1]),
            (point[0] + cross_size, point[1]),
            (0, 0, 255),
            2,
        )
        cv2.line(
            image_with_all_ellipses,
            (point[0], point[1] - cross_size),
            (point[0], point[1] + cross_size),
            (0, 0, 255),
            2,
        )
        # Display the score and accuracy on the image
        center = (int(ellipse[0][0]), int(ellipse[0][1]))
        cv2.putText(
            image_with_all_ellipses,
            f"Score: {score:.2f}, Acc: {accuracy:.2f}",
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    for processed_point in grid_points:
        cv2.circle(
            image_with_all_ellipses,
            processed_point,
            radius=3,
            color=(0, 255, 0),
            thickness=-1,
        )  # Draw a green filled circle

    for processed_point in grid_points:
        cv2.circle(
            image_with_all_ellipses,
            processed_point,
            radius=3,
            color=(0, 0, 255),
            thickness=-1,
        )  # Draw a green filled circle

    # Save the resulting image to the result folder inside the folder_path
    original_name = os.path.splitext(os.path.basename(image_path))[0]
    output_file_name = f"{original_name}.png"
    output_path = os.path.join(output_path, output_file_name)
    cv2.imwrite(output_path, image_with_all_ellipses)
    print(f"Image saved as: {output_path}")


def detect_ellipses_sam2(
    image_path=None,
    image_data=None,
    predictor=None,
    grid_points=None,
    min_diameter_of_object_px=None,
    max_diameter_of_object_px=None,
    accuracy_of_reliability=0.6,
    threshold_value=200,
    ellipse_accuracy=0.9,
):
    if image_data is not None:
        image = image_data
    elif image_path is not None:
        image = cv2.imread(image_path)
    else:
        raise ValueError("Either image_path or image_data must be provided")

    if image is None:
        raise ValueError("Failed to load image")

    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded_image = cv2.threshold(
        gray_image, threshold_value, 255, cv2.THRESH_BINARY
    )

    predictor.set_image(blur_image)

    processed_points = []
    objects = (
        []
    )  # List to store suitable ellipses along with their masks, accuracy, and score
    combined_mask = np.zeros_like(
        gray_image, dtype=np.float32
    )  # Combined mask for all objects

    # Iterate through each grid point separately with a progress bar
    for idx, point in enumerate(tqdm(grid_points, desc="Processing grid points")):
        # Check if the current point is contained by any existing mask
        point_contained = any(
            point[1] < existing_mask["mask"].shape[0]
            and point[0] < existing_mask["mask"].shape[1]
            and existing_mask["mask"][point[1], point[0]]
            for existing_mask in objects
        )
        if point_contained:
            continue

        if thresholded_image[point[1], point[0]] > 0:
            continue

        input_points = np.array([point])
        point_labels = np.array([1])  # Each point gets its own label

        # Create mask for the current point
        masks, scores, _ = predictor.predict(
            point_coords=input_points, point_labels=point_labels, multimask_output=True
        )
        processed_points.append(point)

        # Use only the mask with the highest score if greater than threshold and area within specified range
        if len(scores) > 0 and scores[0] > accuracy_of_reliability:
            mask = masks[0]
            mask_area = (mask > 0).sum()

            if not (
                math.pi * (min_diameter_of_object_px / 2) ** 2
                <= mask_area
                <= math.pi * (max_diameter_of_object_px / 2) ** 2
            ):
                continue

            score = scores[0]

            # Compare with all existing masks to avoid duplicates
            intersection = np.logical_and(combined_mask, mask).sum()
            overlap_area = intersection / mask.sum()

            if overlap_area < 0.1:
                # Find contours of the mask
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Fit an ellipse to the largest contour if it has enough points
                    contour = max(contours, key=cv2.contourArea)
                    if (
                        len(contour) >= 5
                    ):  # At least 5 points are needed to fit an ellipse
                        ellipse = cv2.fitEllipse(contour)
                        major_diameter = max(ellipse[1][0], ellipse[1][1])
                        minor_diameter = min(ellipse[1][0], ellipse[1][1])

                        # Check if the major diameter is within the specified range (5 to 30 mm)
                        if (
                            min_diameter_of_object_px
                            <= minor_diameter
                            <= max_diameter_of_object_px
                            and min_diameter_of_object_px
                            <= major_diameter
                            <= max_diameter_of_object_px
                        ):
                            # Calculate the area of the contour
                            contour_area = cv2.contourArea(contour)
                            # Calculate the area of the ellipse
                            ellipse_area = (
                                np.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)
                            )
                            # Calculate the accuracy as the ratio of ellipse area to contour area
                            accuracy = (
                                ellipse_area / contour_area if contour_area > 0 else 0
                            )

                            if ellipse_accuracy <= accuracy <= 2 - ellipse_accuracy:
                                # Store the ellipse, mask, accuracy, and score in the list
                                objects.append(
                                    {
                                        "ellipse": ellipse,
                                        "mask": mask,
                                        "accuracy": scores[0],
                                        "score": score,
                                        "point": point,
                                    }
                                )
                                combined_mask = cv2.bitwise_or(combined_mask, mask)

                        del contours, contour, mask, scores

        del input_points, point_labels

    del thresholded_image, combined_mask
    gc.collect()

    return objects


def process_all_images_in_folder(folder_path):
    sam2_path = sam2_checkpoint = os.path.abspath(
        r"C:\My storage\Python projects\MLBox\models\sam2.1"
    )
    sam2_checkpoint = os.path.join(sam2_path, "sam2.1_hiera_large.pt")
    model_cfg = os.path.join(sam2_path, "sam2.1_hiera_l.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)

    result_folder_path = os.path.join(folder_path, "result")
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            try:
                image_path = os.path.join(folder_path, filename)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Failed to load image: {image_path}")
                    continue

                coco_output_path = os.path.join(
                    result_folder_path, f"{os.path.splitext(filename)[0]}_coco.json"
                )
                # Skip if corresponding JSON exists
                if os.path.exists(coco_output_path):
                    continue

                a4_rectangls = detect_white_rectangles(image)

                if not a4_rectangls:  # Check if list is empty
                    print(f"No white rectangles detected in image: {filename}")
                    continue

                # Process image
                a4_rectangle = a4_rectangls[0]

                pixels_per_mm = a4_rectangle["pixels_per_mm"]

                step_size = round(pixels_per_mm * MIN_SIZE_OF_OBJECT_MM)
                height, width = image.shape[:2]

                polygon_path = Path(a4_rectangle["rectangle"].reshape(-1, 2))
                grid_points = [
                    (x, y)
                    for y in range(0, height, step_size)
                    for x in range(0, width, step_size)
                    if polygon_path.contains_point((x, y))
                ]

                detected_ellipses = detect_ellipses_sam2(
                    image_path=image_path,
                    predictor=predictor,
                    grid_points=grid_points,
                    min_diameter_of_object_px=MIN_SIZE_OF_OBJECT_MM * pixels_per_mm,
                    max_diameter_of_object_px=10
                    * MIN_SIZE_OF_OBJECT_MM
                    * pixels_per_mm,
                    accuracy_of_reliability=0.6,
                    threshold_value=200,
                    ellipse_accuracy=0.9,
                )

                # Generate and save COCO JSON
                all_masks = [obj["mask"] for obj in detected_ellipses]
                coco_json = masks_to_coco_annotations(
                    masks=all_masks,
                    image_id=1,
                    filename=filename,
                    categories=categories,
                )

                with open(coco_output_path, "w") as json_file:
                    json_file.write(coco_json)

                save_image_with_result(
                    image,
                    detected_ellipses,
                    grid_points,
                    image_path,
                    output_path=result_folder_path,
                )

                # Cleanup
                cv2.destroyAllWindows()
                del image, a4_rectangle, grid_points, detected_ellipses, all_masks
                del coco_json

                # Force garbage collection
                gc.collect()

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                raise e
                continue
    del sam2_model
    del predictor
    gc.collect()


MIN_SIZE_OF_OBJECT_MM = 4
categories = [{"id": 1, "name": "peanut"}]

folder_path = r"C:\My storage\Python projects\DataSets\peanuts\task2"  # Replace with your folder path
process_all_images_in_folder(folder_path)
# annotation_data =combine_coco_annotation_files(os.path.join(folder_path, "result"))


# output_path = os.path.join(folder_path, "result", "annotation_coco.json")

# Write the JSON data to the specified file
# with open(output_path, 'w') as f:
#    json.dump(annotation_data, f)
