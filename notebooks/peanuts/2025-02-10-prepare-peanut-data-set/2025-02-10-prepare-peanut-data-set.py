from pathlib import Path
from PIL import Image as PILImage
import numpy as np
import cv2
from mlbox.settings import DEBUG_MODE, ROOT_DIR
from mlbox.utils.cvtools import preprocess_images_with_white_rectangle
import json
import random

CURRENT_DIR = Path(__file__).parent
input_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "input"
result_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "output"

def preprocessing_images_for_dataset(input_folder: Path, output_folder: Path) -> None:
    """
    Preprocess images from the input folder and save the preprocessed images to the output folder.

    Args:
        input_folder (Path): The folder containing the raw input images.
        output_folder (Path): The folder to save the preprocessed images.
    """
    # Gather all image files from the input folder
    output_files = {file.name for file in output_folder.glob("*.*")}
    image_files = [file for file in input_folder.glob("*.jpg") if file.name not in output_files]

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each image one at a time
    for img_file in image_files:
        # Read the image and convert to numpy array
        input_image = np.array(PILImage.open(img_file))

        # Preprocess the image
        try:
            preprocessed_result = preprocess_images_with_white_rectangle(input_images=[input_image])
        except ValueError as e:
            print(f"{e}")
            continue

        preprocessed_image = preprocessed_result[0][0]

        # Save the preprocessed image to the output folder
        output_path = output_folder / img_file.name
        PILImage.fromarray(preprocessed_image).save(output_path, format="JPEG")

def draw_contours_from_coco_annotation(input_folder: Path, output_folder: Path, coco_annotation: Path) -> None:
    """
    Draw contours from COCO annotation on images and save the result.

    Args:
        input_folder (Path): The folder containing the raw input images.
        output_folder (Path): The folder to save the images with contours.
        coco_annotation (Path): The path to the COCO annotation file.
    """
    # Load COCO annotation
    with open(coco_annotation, 'r') as f:
        annotations = json.load(f)

    # Create a mapping from image_id to file_name
    image_id_to_file_name = {image['id']: image['file_name'] for image in annotations['images']}

    # Ensure the output folder exists
    output_folder.mkdir(parents=True, exist_ok=True)

    # Process each image in the input folder
    for img_file in input_folder.glob("*.jpg"):
        # Read the image
        image = cv2.imread(str(img_file))

        # Find annotations for the current image
        image_id = next((id for id, file_name in image_id_to_file_name.items() if file_name == img_file.name), None)
        if image_id is None:
            continue

        image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]

        # Create a mask for the current image
        mask = np.zeros(image.shape, dtype=np.uint8)

        # Draw filled polygons for each annotation
        for ann in image_annotations:
            segmentation = ann['segmentation']
            for seg in segmentation:
                points = np.array(seg).reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [points], color=(0, 255, 0))

        # Blend the mask with the original image
        alpha = 0.3
        image = cv2.addWeighted(mask, alpha, image, 1 - alpha, 0)

        # Save the image with contours
        output_path = output_folder / f"{img_file.stem}_contour.jpg"
        cv2.imwrite(str(output_path), image)

def shuffle_and_rename_files(input_folder: Path) -> None:
    """
    Shuffle the files in the input folder and rename each file by adding a random index with leading zeros.

    Args:
        input_folder (Path): The folder containing the files to be shuffled and renamed.
    """
    # Gather all files in the input folder
    files = list(input_folder.glob("*.*"))

    # Generate a list of random indices
    indices = list(range(len(files)))
    random.shuffle(indices)

    # Rename each file with the random index
    for idx, file in zip(indices, files):
        new_name = f"{idx:04d}_{file.name}"
        

        new_path = input_folder / new_name
        file.rename(new_path)



if __name__ == "__main__":
    output_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/preprocessed")
    input_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/raw")
    #preprocessing_images_for_dataset(input_folder, output_folder)

    coco_annotation = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/raw/1/annotation_coco.json")
    output_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/preprocessed/1")
    input_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/preprocessed/1")
    #draw_contours_from_coco_annotation(input_folder=output_folder, output_folder=output_folder, coco_annotation=coco_annotation)

    input_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/preprocessed/")
    shuffle_and_rename_files(input_folder)
