import argparse

import cv2


def crop_image(image_path, xyxy, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("Image not found. Check the path.")
    crop = image[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
    cv2.imwrite(output_path, crop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop image based on xyxy coordinates")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument(
        "xyxy", type=int, nargs=4, help="Bounding box coordinates in xyxy format"
    )
    parser.add_argument("output_path", type=str, help="Path to save the cropped image")
    args = parser.parse_args()

    crop_image(args.image_path, args.xyxy, args.output_path)
