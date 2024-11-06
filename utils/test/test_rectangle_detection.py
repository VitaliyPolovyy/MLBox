import cv2
import os
import numpy as np
from utils.cvtools import detect_white_rectangles

if __name__ == "__main__":
    INPUT_FOLDER = r"C:\My storage\Python projects\MLBox\utils\test"
    # Get the first image from the input folder
    filenames = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Load the first image
    image_path = os.path.join(INPUT_FOLDER, filenames[0])
    image = cv2.imread(image_path)

    rectangles = detect_white_rectangles(image)

    # Output the number of detected rectangles
    print(f"Number of detected rectangles: {len(rectangles)}")

    # Draw the first detected rectangle on the image
    result_image = image.copy()

    for rect in rectangles[:10]:
        rect_points = rect["rectangle"].reshape(4, 2).astype(int)
        color = tuple(np.random.randint(0, 256, 3).tolist())  # Random color for each rectangle
        cv2.polylines(result_image, [rect_points], isClosed=True, color=color, thickness=20)

        # Draw scores
        score_text = f"Aspect Ratio: {rect['aspect_ratio_score']:.2f}, Angle: {rect['angle_score']:.2f}, Whiteness: {rect['whiteness_score']:.2f}, Total: {rect['total_score']:.2f}"
        text_position = tuple(rect_points[0])
        color = (0, 255, 255)
        cv2.putText(result_image, score_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3, cv2.LINE_AA)

    screen_width = 1280  # Adjust this value based on your screen
    height, width = result_image.shape[:2]
    scaling_factor = screen_width / width
    result_image = cv2.resize(result_image, (int(width * scaling_factor), int(height * scaling_factor)))
    # Show the result image
    cv2.imshow("Detected Rectangle", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the result image in the same folder
    output_path = os.path.join(INPUT_FOLDER, "detected_rectangle_result.jpg")
    cv2.imwrite(output_path, result_image)
    print(f"Result image saved at: {output_path}")

    # Output details of detected rectangles
    print(rectangles)