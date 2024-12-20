import os
import random
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import dotenv
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw, ImageFilter
from ultralytics import YOLO  # Import YOLO for model loading and prediction

from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent

# Configure SSL at the start to avoid SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""


def process_image(model, image_path, result_folder, alpha, confidence_threshold=0.5):
    # Load the image using Pillow and convert to RGB format
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size

    # Calculate new size to maintain aspect ratio and fit within 1024x1024
    aspect_ratio = min(1024 / image_width, 1024 / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)
    resized_image = image.resize((new_width, new_height))

    # Create a new blank image (1024x1024) and paste the resized image onto the center
    padded_image = Image.new("RGB", (1024, 1024), (0, 0, 0))
    top_left_x = (1024 - new_width) // 2
    top_left_y = (1024 - new_height) // 2
    padded_image.paste(resized_image, (top_left_x, top_left_y))

    # Convert the image to a NumPy array and use the model to make predictions with the confidence threshold
    results = model.predict(
        np.array(padded_image), conf=confidence_threshold, verbose=False
    )

    # Only process if we have valid results and masks are found
    if results and len(results) > 0 and results[0].masks is not None:
        # Create a copy of the padded image to draw overlays on
        overlay = padded_image.copy()
        draw = ImageDraw.Draw(overlay)

        # Iterate over each mask and its corresponding confidence score
        for i, (mask, score) in enumerate(
            zip(results[0].masks.data, results[0].boxes.conf)
        ):
            # Resize the mask back to the size of the padded image (1024x1024)
            mask_resized = cv2.resize(
                mask.cpu().numpy(), (1024, 1024), interpolation=cv2.INTER_NEAREST
            )
            # Generate a random color for the mask
            color = tuple(random.randint(0, 255) for _ in range(3))
            # Use NumPy to create a binary mask and apply it efficiently
            mask_indices = np.where(mask_resized == 1)
            for y, x in zip(mask_indices[0], mask_indices[1]):
                draw.point((x, y), fill=color)

            # Display the confidence score next to the mask
            score_text = f"{score:.2f}"
            draw.text(
                (mask_indices[1][0], mask_indices[0][0]), score_text, fill="white"
            )

        # Blend the overlay with the padded image to add transparency to the masks
        result_image = Image.blend(padded_image, overlay, alpha)

        # Save the result image to the specified result folder
        result_path = result_folder / f"{image_path.stem}_result{image_path.suffix}"
        result_image.save(result_path)
        print(f"Saved: {result_path}")
    else:
        # Print a message if no masks are found for the image
        print(f"No masks found for image: {image_path}")


if __name__ == "__main__":
    # Load environment variables from a .env file
    dotenv.load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    hf_repo_id = os.getenv("HF_REPO_ID")
    hf_model_file = os.getenv("HF_MODEL_FILE")

    # Ensure all necessary environment variables are set
    if not all([hf_token, hf_repo_id, hf_model_file]):
        raise ValueError("Missing required environment variables (huggingface hub)")

    model_path = hf_hub_download(
        repo_id=hf_repo_id, filename=hf_model_file, token=hf_token
    )

    # model_path = ROOT_DIR / "models" / "Yolo" / "yolov8m-seg_v1.pt"
    image_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "input"
    result_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "output"

    alpha = 0.5  # Transparency level of the mask overlay

    # Initialize YOLO model with the specified model path
    model = YOLO(model_path)

    # Ensure the result folder exists, create if it does not
    result_folder.mkdir(parents=True, exist_ok=True)

    # Get list of all jpg images in the specified folder
    image_files = list(image_folder.glob("*.jpg"))
    if not image_files:
        # Exit if no images are found in the folder
        print("No jpg images found in the specified folder")
        exit(1)

    # Process each image in the folder
    for img in image_files:
        try:
            process_image(model, img, result_folder, alpha)
        except Exception as e:
            # Print an error message if processing fails for an image
            print(f"Error processing {img}: {str(e)}")
