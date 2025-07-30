from paddleocr import PaddleOCR
import cv2
from pathlib import Path
from mlbox.settings import ROOT_DIR

CURRENT_DIR = Path(__file__).parent
img_path = ROOT_DIR / "assets" / "LabelGuard" / "input" / "source" / "Lovita_CC_Glazur_150g_UNI_v181224E__tt__curv_.jpg"
output_img_path = img_path.with_name(img_path.stem + ".annotated.jpg")


img = cv2.imread(str(img_path))
scale_ratio = 1#4000 / img.shape[1]
resized = cv2.resize(img, None, fx=scale_ratio, fy=scale_ratio)

# Збереження тимчасового зображення
temp_path = img_path.with_name("resized_temp.jpg")
cv2.imwrite(str(temp_path), resized)
# 1. Ініціалізація OCR
# Якщо потрібен звичайний OCR:
ocr = PaddleOCR(lang='uk')

# Якщо потрібен структурний аналіз (layout/table extraction):
# from paddleocr import PPStructure, save_structure_res
# table_engine = PPStructure(lang='en')

# 2. Завантаження зображення
# img = cv2.imread(str(img_path))
# Для звичайного OCR:
result = ocr.predict(str(temp_path))


for text, score in zip(result[0]['rec_texts'], result[0]['rec_scores']):
    print(f"{text} ({score:.2f})")


exit()
# Для структурного аналізу (розкоментуй якщо треба):
# result = table_engine(img)

# 3. Виклик структури
blocks = []
if result and isinstance(result, list) and 'layout' in result[0]:
    blocks = result[0]['layout']
    for block in blocks:
        x1, y1, x2, y2 = block['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
else:
    print("❗ Layout not found.")

# 4. Отримання блоків
# 5. Збереження
cv2.imwrite(str(output_img_path), img)
print(f"✅ Saved: {output_img_path}, blocks: {len(blocks)}")
