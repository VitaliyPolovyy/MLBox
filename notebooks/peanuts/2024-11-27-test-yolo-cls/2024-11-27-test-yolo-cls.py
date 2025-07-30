import os
from collections import defaultdict
from pathlib import Path

import dotenv
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent

os.environ["CURL_CA_BUNDLE"] = ""

if __name__ == "__main__":

    dotenv.load_dotenv()

    HF_TOKEN = os.getenv("HF_TOKEN")
    HF_REPO_ID = os.getenv("PEANUT_CLS_HF_REPO_ID")
    HF_MODEL_FILE = os.getenv("PEANUT_CLS_HF_MODEL_FILE")

    # Ensure all necessary environment variables are set
    if not all([HF_TOKEN, HF_REPO_ID, HF_MODEL_FILE]):
        raise ValueError("Missing required environment variables (huggingface hub)")

    print(HF_MODEL_FILE)
    model_path = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_MODEL_FILE, token=HF_TOKEN
    )

    model = YOLO(model_path)

    output_folder = ROOT_DIR / "tmp" / CURRENT_DIR.name / "output"
    input_folder = ROOT_DIR / "assets" / "DataSet" / "peanuts_class" / "test" / "images"

    # Run predictions
    results = model.predict(
        source=input_folder, task="classify", save=True, save_dir=output_folder
    )

    print(results[0].probs.top1)
    print(results[0].names)

    # Prepare summary variables
    total_images = 0
    correct_predictions = 0
    class_summary = defaultdict(lambda: {"total": 0, "correct": 0})

    # Process results
    for result in results:
        image_path = Path(result.path)  # Path to the image
        top_class = result.probs.top1  # Predicted class index
        confidence = result.probs.top1conf  # Confidence score

        real_class_index = int(image_path.stem[0]) + 1
        predicted_class_index = int(top_class)

        total_images += 1
        class_summary[real_class_index]["total"] += 1

        if real_class_index == predicted_class_index:
            correct_predictions += 1
            class_summary[real_class_index]["correct"] += 1

    # Print overall summary
    accuracy = correct_predictions / total_images * 100 if total_images > 0 else 0
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Print per-class summary
    for class_index, summary in class_summary.items():
        class_accuracy = (
            summary["correct"] / summary["total"] * 100 if summary["total"] > 0 else 0
        )
        print(
            f"Class {class_index} - Total: {summary['total']}, Correct: {summary['correct']}, Accuracy: {class_accuracy:.2f}%"
        )
