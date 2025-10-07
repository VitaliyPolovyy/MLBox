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


def preprocess_images_for_dataset(
    input_dir: Path, output_dir: Path, coco_annotation: Path
) -> None:
    """
    Preprocess images by cutting each bounding box and creating separate images 
    with YOLO instance segmentation files.

    Args:
        input_dir (Path): The folder containing the raw input images.
        output_dir (Path): The folder to save the processed images and YOLO files.
        coco_annotation (Path): The path to the COCO annotation file.
    """
    # Load COCO annotation
    with open(coco_annotation, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    # Create a mapping from image_id to file_name
    image_id_to_file_name = {
        image['id']: image['file_name'] for image in annotations['images']
    }

    # Ensure the output folder exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the input folder
    for img_file in input_dir.glob("*.jpg"):
        # Read the image
        image = cv2.imread(str(img_file))

        # Find annotations for the current image
        image_id = next(
            (id for id, file_name in image_id_to_file_name.items() 
             if file_name == img_file.name), 
            None
        )
        if image_id is None:
            continue

        image_annotations = [
            ann for ann in annotations['annotations'] 
            if ann['image_id'] == image_id
        ]

        # Process each bounding box in the annotations
        for idx, ann in enumerate(image_annotations):
            bbox = ann['bbox']
            x, y, w, h = map(int, bbox)
            cropped_image = image[y:y+h, x:x+w]
            # Enlarge the bounding box by 20 pixels in each direction

            width_padding = 2
            height_padding = 2

            x = max(0, x - width_padding)
            y = max(0, y - height_padding)
            w = min(image.shape[1] - width_padding, w + width_padding*2)
            h = min(image.shape[0] - y, h + height_padding*2)
            cropped_image = image[y:y+h, x:x+w]

            # Save the cropped image
            cropped_image_path = output_dir / f"{img_file.stem}_{idx}.jpg"
            cv2.imwrite(str(cropped_image_path), cropped_image)

            # Create YOLO instance segmentation file
            yolo_file_path = output_dir / f"{img_file.stem}_{idx}.txt"
            with open(yolo_file_path, 'w', encoding='utf-8') as yolo_file:
                for seg in ann['segmentation']:
                    points = np.array(seg).reshape((-1, 2))
                    # Correct points according to the bounding box
                    corrected_points = points - [x, y]
                    # Normalize points
                    normalized_points = corrected_points / [w, h]
                    yolo_file.write(
                    "0 " + " ".join(map(str, normalized_points.flatten())) + "\n"
                    )

def draw_contours_on_images(input_dir: Path, output_dir: Path) -> None:
    """
    Draw contours on images using YOLO segmentation files.

    Args:
        input_dir (Path): The folder containing the input images and YOLO files.
        output_dir (Path): The folder to save the images with drawn contours.
    """
    # Ensure the output folder exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each image in the input folder
    for img_file in input_dir.glob("*.jpg"):
        # Read the image
        image = cv2.imread(str(img_file))

        # Read the corresponding YOLO file
        yolo_file_path = img_file.with_suffix('.txt')
        if not yolo_file_path.exists():
            continue

        with open(yolo_file_path, 'r', encoding='utf-8') as yolo_file:
            for line in yolo_file:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                points = np.array(parts[1:], dtype=float).reshape((-1, 2))
                points = (points * [image.shape[1], image.shape[0]]).astype(int)
                cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        # Save the image with contours
        output_image_path = output_dir / f"{img_file.stem}_.jpg"
        cv2.imwrite(str(output_image_path), image)


def split_dataset(input_folder: Path, output_folder: Path, train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1) -> None:
    """
    Split images and their annotations into train, val, and test sets for YOLO segmentation training.

    Args:
        input_folder (Path): The folder containing the input images and annotation files.
        output_folder (Path): The folder to save the split datasets.
        train_ratio (float): The ratio of the training set.
        val_ratio (float): The ratio of the validation set.
        test_ratio (float): The ratio of the test set.
    """
    # Ensure the output folder structure exists
    for split in ['train', 'val', 'test']:
        (output_folder / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_folder / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Get all image files and shuffle them
    image_files = list(input_folder.glob("*.jpg"))
    random.shuffle(image_files)

    # Calculate split indices
    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Split the dataset
    train_files = image_files[:train_end]
    val_files = image_files[train_end:val_end]
    test_files = image_files[val_end:]

    # Function to copy files to the respective folders
    def copy_files(files, split):
        for img_file in files:
            label_file = img_file.with_suffix('.txt')
            if not label_file.exists():
                continue
            img_dest = output_folder / split / 'images' / img_file.name
            label_dest = output_folder / split / 'labels' / label_file.name
            img_dest.write_bytes(img_file.read_bytes())
            label_dest.write_bytes(label_file.read_bytes())

    # Copy files to train, val, and test folders
    copy_files(train_files, 'train')
    copy_files(val_files, 'val')
    copy_files(test_files, 'test')
if __name__ == "__main__":
    
    output_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/tmp/2025-02-14-prepare-separated-peanuts-dataset/output")
    input_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/tmp/2025-02-14-prepare-separated-peanuts-dataset/input")
    annotation_coco =  input_folder / "annotation_coco.json"

    #preprocess_images_for_dataset(input_folder, result_folder, annotation_coco)
    #draw_contours_on_images(output_folder, output_folder / "masks")

    
    #C:\My storage\Python projects\MLBox\assets\datasets\peanut\separated
    input_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/separated/all")
    output_folder = Path(r"/mnt/c/My storage/Python projects/MLBox/assets/datasets/peanut/separated/for-training")

    split_dataset (in   put_folder, output_folder)
    
