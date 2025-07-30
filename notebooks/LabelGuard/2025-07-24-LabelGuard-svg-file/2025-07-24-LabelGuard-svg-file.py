from pathlib import Path
import re
import fitz  # PyMuPDF
from mlbox.settings import ROOT_DIR
from bs4 import BeautifulSoup

CURRENT_DIR = Path(__file__).parent

PDF_FILE = ROOT_DIR / "assets" / "LabelGuard" / "input" / "source" / "Lovita_CC_Glazur_150g_UNI_v181224E копія.pdf"


from pdfminer.high_level import extract_text

text = extract_text(PDF_FILE)

print(text)


exit()

SVG_FILE = ROOT_DIR / "assets" / "LabelGuard" / "input" / "source" / "Lovita_CC_Glazur_150g_UNI_v181224E.svg"

with open(SVG_FILE, encoding="utf-8") as f:
    soup = BeautifulSoup(f, "xml")

 
style_text = "\n".join(s.string for s in soup.find_all("style") if s.string)

bold_classes = set()
for selector_block, rule_block in re.findall(r"([^{]+)\{([^}]+)\}", style_text):
    if "font-family" in rule_block.lower() and "bold" in rule_block.lower():
        selectors = [sel.strip().lstrip(".") for sel in selector_block.split(",")]
        bold_classes.update(selectors)



# Витягнути текст з жирних класів
bold_words = []
for tspan in soup.find_all("tspan"):
    cls = tspan.get("class")
    if cls in bold_classes:
        word = tspan.text.strip()
        if word:
            bold_words.append((word, cls))

extracted = []
for t in soup.find_all("tspan"):
    cls = t.get("class")
    text = t.get_text(strip=True)
    if cls and text:
        extracted.append((text))

full_text = "".join(extracted)



# Зчитування SVG-файлу у байтах
with open(SVG_FILE, "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "xml")

lines = []
for text_tag in soup.find_all("text"):
    line = []
    last_x = None
    for tspan in text_tag.find_all("tspan"):
        text = tspan.get_text().strip()
        if not text:
            continue
        x = tspan.get("x")
        try:
            x_val = float(x) if x else None
        except:
            x_val = None
        if last_x is not None and x_val is not None and (x_val - last_x) > 5:
            line.append(" ")
        elif last_x is not None and x_val is None:
            line.append(" ")
        line.append(text)
        if x_val is not None:
            last_x = x_val
    if line:
        lines.append("".join(line))

text = "\n".join(lines)
print(text)