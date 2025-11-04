"""Test LayoutDetector for overlapping blocks."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mlbox.settings import ROOT_DIR
from mlbox.services.LabelGuard.layout_detector import LayoutDetector

# Test image
dataset_dir = ROOT_DIR / "assets" / "labelguard" / "datasets"
image_path = dataset_dir / "Lovita_CC_Glazur_150g_UNI_v181224E.jpg"

# Run layout detector
original_image = Image.open(image_path)
layout_detector = LayoutDetector()
layout_blocks = layout_detector.extract_blocks(image_path, score_thresh=0.4)

print(f"\nDetected {len(layout_blocks)} blocks:")
for block in layout_blocks:
    print(f"  Block {block.index}: bbox={block.bbox}, score={block.score:.3f}")

# Check for intersecting blocks
def get_intersection_percent(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection_area = x_overlap * y_overlap
    
    if intersection_area == 0:
        return 0, 0, 0
    
    # Calculate areas
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersection_area
    
    # Percentages
    pct_of_box1 = (intersection_area / area1 * 100) if area1 > 0 else 0
    pct_of_box2 = (intersection_area / area2 * 100) if area2 > 0 else 0
    pct_of_union = (intersection_area / union_area * 100) if union_area > 0 else 0
    
    return pct_of_box1, pct_of_box2, pct_of_union

print("\nChecking for intersections:")
intersections = []
for i, block1 in enumerate(layout_blocks):
    for j, block2 in enumerate(layout_blocks[i+1:], start=i+1):
        pct1, pct2, pct_union = get_intersection_percent(block1.bbox, block2.bbox)
        if pct_union > 0:
            intersections.append((block1.index, block2.index))
            print(f"  ⚠️  Block {block1.index} ∩ Block {block2.index}: {pct1:.1f}% of B{block1.index}, {pct2:.1f}% of B{block2.index}, IoU={pct_union:.1f}%")

if not intersections:
    print("  ✓ No intersections found")

# Visualize blocks
vis_image = original_image.copy()
draw = ImageDraw.Draw(vis_image)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
except:
    font = None

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

for block in layout_blocks:
    color = colors[block.index % len(colors)]
    draw.rectangle(block.bbox, outline=color, width=5)
    
    # Draw block number in top-right corner with large red text
    x1, y1, x2, y2 = block.bbox
    text = str(block.index)
    
    if font:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
    else:
        tw = len(text) * 40
    
    text_x = x2 - tw - 10
    text_y = y1 + 5
    
    draw.text((text_x, text_y), text, fill=(255, 0, 0), font=font)

# Save visualization
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / f"{image_path.stem}_layout_debug.jpg"
vis_image.save(output_path, quality=95)
print(f"Saved: {output_path}")

