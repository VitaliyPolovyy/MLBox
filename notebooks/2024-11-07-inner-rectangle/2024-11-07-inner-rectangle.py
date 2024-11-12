import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image
image_path = r"C:\My storage\Python projects\MLBox\utils\test\D21010101000007__12282__JALMO4325686-624__0090.jpg"
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError("Image not found. Check the path.")

# Rectangle points (using the points you provided)
points = np.array(
    [[925, 566], [776, 2638], [3577, 2845], [3751, 767]], dtype=np.float32
)

# Calculate center point of the rectangle
center_x = np.mean(points[:, 0])
center_y = np.mean(points[:, 1])
center = np.array([center_x, center_y])

# Shift value
shift = 750

# Calculate new rectangle points by moving each point inward by `shift` pixels
new_points = []
for point in points:
    direction = center - point
    unit_direction = (direction / np.linalg.norm(direction)) * shift
    new_point = point + unit_direction
    new_points.append(new_point)

new_points = np.array(new_points, dtype=np.int32)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create a mask for the area between the rectangles
mask = np.zeros_like(gray_image)

# Draw the outer rectangle in white on the mask
cv2.fillPoly(mask, [points.astype(int)], 255)

# Draw the inner rectangle in black to "cut out" from the mask
cv2.fillPoly(mask, [new_points.astype(int)], 0)

# Apply the mask to the grayscale image to highlight the area between rectangles
highlighted_area = cv2.bitwise_and(gray_image, mask)

# Display the result
plt.figure(figsize=(10, 10))
plt.imshow(highlighted_area, cmap="gray")
plt.axis("off")
plt.show()
