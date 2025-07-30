from pathlib import Path
import cv2
import numpy as np
from mlbox.settings import ROOT_DIR

def detect_blocks(img, thresh_img, min_area=500*500, max_area=6000*6000):
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1,9))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    joined = cv2.dilate(thresh_img, kernel_v, iterations=1)
    joined = cv2.dilate(joined, kernel_h, iterations=1)

    contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blocks = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if min_area < area < max_area:
            blocks.append((x, y, x + w, y + h))
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 8)
    return blocks

def mask_blocks(thresh_img, blocks):
    mask = np.ones_like(thresh_img) * 255
    for x1, y1, x2, y2 in blocks:
        mask[y1:y2, x1:x2] = 0
    return cv2.bitwise_and(thresh_img, thresh_img, mask=mask)

# Завантаження зображення
CURRENT_DIR = Path(__file__).parent
img_file = ROOT_DIR / "assets" / "LabelGuard" / "input" / "source" / "Lovita_CC_Glazur_150g_UNI_v181224E копія.jpg"
output_img_file = img_file.with_name(img_file.stem + ".annotated.jpg")
img = cv2.imread(str(img_file))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Бінаризація
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, blockSize=31, C=15)

# Перша ітерація
blocks = detect_blocks(img, thresh)

# Маскування знайдених блоків
#thresh_cleaned = mask_blocks(thresh, blocks)

# Друга ітерація
#blocks += detect_blocks(img, thresh_cleaned)

# Збереження результату
cv2.imwrite(str(output_img_file), img)
print(f"Збережено: {output_img_file}")
print(f"Кількість блоків: {len(blocks)}")
