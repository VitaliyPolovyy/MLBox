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

# Configure SSL at the start to avoid SSL certificate verification issues
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""


def process_image(model, image_path, result_folder, alpha):
    """Process a single image and save the result.
    Args:
        model: The YOLO model to use for prediction.
        image_path: Path to the input image.
        result_folder: Folder to save the result images.
        alpha: Transparency level for blending the overlay with the original image.
    """
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

    # Convert the image to a NumPy array and use the model to make predictions
    results = model.predict(np.array(padded_image), verbose=False)

    # Only process if we have valid results and masks are found
    if results and len(results) > 0 and results[0].masks is not None:
        # Create a copy of the padded image to draw overlays on
        overlay = padded_image.copy()
        draw = ImageDraw.Draw(overlay)

        # Iterate over each mask in the results and draw it on the overlay
        for i, mask in enumerate(results[0].masks.data):
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
        """
        # Draw bounding boxes around detected objects
        for box in results[0].boxes.data:
            # Extract coordinates of the bounding box
            x1, y1, x2, y2 = map(int, box[:4])
            # Draw a rectangle on the overlay with green color
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

        """

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

    # Define paths for the model, input images, and results
    model_path = ROOT_DIR / "models" / "Yolo" / "yolov8m-seg_v1.pt"
    image_folder = ROOT_DIR / "Assets" / "test_peanuts_images"
    result_folder = image_folder / "result"
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
