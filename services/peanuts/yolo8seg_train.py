import os
import json
import shutil
import random
from pathlib import Path
import logging
import torch
from ultralytics import YOLO
import yaml
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_device():
    if torch.cuda.is_available():
        device = 0
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logging.info("No GPU available, using CPU")
    return device


def convert_coco_to_yolo(annotation_path: Path, img_dir: Path, label_dir: Path):
    """Convert COCO format annotations to YOLO format"""
    # Create labels directory
    label_dir.mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    with open(annotation_path) as f:
        coco_data = json.load(f)

    # Create image_id to image info mapping
    image_info = {img['id']: img for img in coco_data['images']}

    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Process each image
    for image_id, annotations in annotations_by_image.items():
        image = image_info[image_id]
        img_width = image['width']
        img_height = image['height']

        # Create YOLO format label file
        label_file = label_dir / Path(image['file_name']).with_suffix('.txt').name

        with open(label_file, 'w') as f:
            for ann in annotations:
                # Get segmentation polygons
                polygons = ann['segmentation']
                # For each polygon in the annotation
                for polygon in polygons:
                    # Convert to normalized coordinates
                    normalized = []
                    for i in range(0, len(polygon), 2):
                        x = polygon[i] / img_width
                        y = polygon[i + 1] / img_height
                        normalized.extend([x, y])

                    # Write to file: class_id x1 y1 x2 y2 ...
                    f.write(f"0 {' '.join(map(str, normalized))}\n")


def create_yaml_config(base_path: Path):
    """Create YAML configuration file for training"""
    config = {
        'path': str(base_path),
        'train': str(base_path / 'train'),  # Directory containing images and labels
        'val': str(base_path / 'val'),  # Directory containing images and labels
        'test': str(base_path / 'test'),  # Directory containing images and labels
        'names': {0: 'peanut'},
        'nc': 1
    }

    yaml_path = base_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(config, f, default_flow_style=False)

    logging.info("Created dataset.yaml configuration file")
    return yaml_path


def process_dataset(base_path: Path):
    test_dir = base_path / 'test'

    # If test directory exists, skip processing
    if test_dir.exists():
        logging.info("Using existing dataset split")
        return base_path / 'dataset.yaml'

    logging.info("Creating new dataset split...")

    # Create directories
    for split in ['train', 'val', 'test']:
        split_dir = base_path / split
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)

    # Load COCO annotations
    with open(base_path / 'annotation_coco.json', 'r') as f:
        coco_data = json.load(f)

    # Get all image IDs and shuffle
    image_ids = [img['id'] for img in coco_data['images']]
    random.shuffle(image_ids)

    # Split into train/val/test
    total = len(image_ids)
    train_size = int(0.7 * total)
    val_size = int(0.2 * total)

    splits = {
        'train': image_ids[:train_size],
        'val': image_ids[train_size:train_size + val_size],
        'test': image_ids[train_size + val_size:]
    }

    # Process each split
    for split_name, split_ids in splits.items():
        # Create new COCO structure
        split_data = {
            'info': coco_data['info'],
            'licenses': coco_data['licenses'],
            'categories': coco_data['categories'],
            'images': [],
            'annotations': []
        }

        # Filter images and annotations
        id_set = set(split_ids)
        split_data['images'] = [img for img in coco_data['images']
                                if img['id'] in id_set]
        split_data['annotations'] = [anno for anno in coco_data['annotations']
                                     if anno['image_id'] in id_set]

        # Copy images and save annotations
        split_dir = base_path / split_name
        img_dir = split_dir / 'images'
        for img in split_data['images']:
            shutil.copy2(base_path / 'images' / img['file_name'],
                         img_dir / img['file_name'])

        # Save COCO annotations
        annotation_path = split_dir / 'annotation_coco.json'
        with open(annotation_path, 'w') as f:
            json.dump(split_data, f)

        # Convert to YOLO format
        label_dir = split_dir / 'labels'
        convert_coco_to_yolo(annotation_path, img_dir, label_dir)

        logging.info(f"Created {split_name} split with {len(split_ids)} images")

    # Create YAML configuration file
    return create_yaml_config(base_path)


def split_and_train(base_path: str, weights_path: str):
    # Convert base_path to Path object
    base_path = Path(base_path)

    # Process dataset and get yaml path
    yaml_path = process_dataset(base_path)

    # Train YOLOv8 model
    logging.info("Starting model training...")
    model = YOLO(weights_path)

    # Get available device
    device = get_device()

    model.train(
        data=str(yaml_path),
        epochs=2,
        imgsz=1024,
        batch=16,
        device=device,
        workers=8,
        task='segment'
    )
    logging.info("Training completed successfully")


if __name__ == "__main__":
    base_path = r"C:\My storage\Python projects\DataSets\peanuts\task1"
    weights_path = r"C:\My storage\Python projects\MLBox\models\Yolo\yolov8s-seg.pt"


    def process_dataset(base_path: Path):
        test_dir = base_path / 'test'

        # If test directory exists, skip processing
        if test_dir.exists():
            logging.info("Using existing dataset split")
            return base_path / 'dataset.yaml'


    split_and_train(base_path, weights_path)