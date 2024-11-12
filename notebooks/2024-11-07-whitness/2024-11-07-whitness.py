import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = r"C:\My storage\Python projects\MLBox\utils\test\D21010101000007__12282__JALMO4325686-624__0090.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Image not found. Check the path.")

# Rectangle points (using the points you provided)
points = np.array([[925, 566], [776, 2638], [3577, 2845], [3751, 767]], dtype=np.int32)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a mask for the rectangle
mask = np.zeros_like(gray_image)

# Draw the rectangle in white on the mask
cv2.fillPoly(mask, [points], 255)

# Apply the mask to the grayscale image to highlight the area of interest
highlighted_area = cv2.bitwise_and(gray_image, mask)

# Display the original image with the rectangle overlay
image_with_rectangle = image.copy()
cv2.polylines(
    image_with_rectangle, [points], isClosed=True, color=(0, 255, 0), thickness=5
)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.title("Original Image with Rectangle")
plt.imshow(cv2.cvtColor(image_with_rectangle, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Highlighted Area for Whiteness Calculation")
plt.imshow(highlighted_area, cmap="gray")
plt.axis("off")

plt.show()
