import os
import ssl
from pathlib import Path
import dotenv
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO  # Import YOLO for model loading and prediction
from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent
os.environ["CURL_CA_BUNDLE"] = ""
# Configure SSL at the start to avoid SSL certificate verification issues

# Constants for image size

if __name__ == "__main__":
    # Load environment variables from a .env file
    dotenv.load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_REPO_ID = os.getenv("PEANUT_HF_REPO_ID")
    HF_MODEL_FILE = os.getenv("PEANUT_HF_MODEL_FILE")
    IMAGE_SIZE = int(os.getenv("PEANUT_IMAGE_SIZE"))

    # Ensure all necessary environment variables are set
    if not all([HF_TOKEN, HF_REPO_ID, HF_MODEL_FILE]):
        raise ValueError("Missing required environment variables (huggingface hub)")

    model_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_MODEL_FILE, token=HF_TOKEN
    )

    # model_path = ROOT_DIR / "models" / "Yolo" / "yolov8m-seg_v1.pt"
    image_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "input"
    result_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "output"

    confidence_threshold = 0.5  # Confidence threshold for predictions

    # Initialize YOLO model with the specified model path
    model = YOLO(model_path)

    # Ensure the result folder exists, create if it does not
    result_folder.mkdir(parents=True, exist_ok=True)

    # Get list of all jpg images in the specified folder
    image_files = list(image_folder.glob("*.jpg"))
    if not image_files:
        exit(1)

    # Process each image in the folder
    for img in image_files:
        # Load the image using Pillow and convert to RGB format
        image = Image.open(img).convert("RGB")
        image_width, image_height = image.size

        # Calculate new size to maintain aspect ratio and fit within IMAGE_SIZExIMAGE_SIZE
        aspect_ratio = min(IMAGE_SIZE / image_width, IMAGE_SIZE / image_height)
        new_width = int(image_width * aspect_ratio)
        new_height = int(image_height * aspect_ratio)
        resized_image = image.resize((new_width, new_height))

        # Create a new blank image (IMAGE_SIZExIMAGE_SIZE) and paste the resized image onto the center
        padded_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
        top_left_x = (IMAGE_SIZE - new_width) // 2
        top_left_y = (IMAGE_SIZE - new_height) // 2
        padded_image.paste(resized_image, (top_left_x, top_left_y))

        # Convert the image to a NumPy array and use the model to make predictions with the confidence threshold
        results = model.predict(
            np.array(padded_image), conf=confidence_threshold, verbose=False
        )

        # Only process if we have valid results and masks are found
        if results and len(results) > 0 and results[0].masks is not None:
            # Iterate over each bounding box and save the cropped image
            for i, box in enumerate(results[0].boxes.xyxy):
                # Extract the coordinates of the bounding box
                x_min, y_min, x_max, y_max = map(int, box)

                # Crop the bounding box from the padded image
                cropped_image = padded_image.crop((x_min, y_min, x_max, y_max))

                # Save the cropped image to the result folder, keeping the original size
                cropped_image_path = result_folder / f"{img.stem}_{i}{img.suffix}"
                cropped_image.save(cropped_image_path)
                print(f"Saved cropped image: {cropped_image_path}")
        else:
            # Print a message if no masks are found for the image
            print(f"No masks found for image: {img}")
