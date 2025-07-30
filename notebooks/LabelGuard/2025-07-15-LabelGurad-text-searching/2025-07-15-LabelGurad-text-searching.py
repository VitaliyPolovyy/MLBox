import pandas as pd
import re
from pathlib import Path
from mlbox.settings import ROOT_DIR
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString

CURRENT_DIR = Path(__file__).parent

ETALON_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "Lovita_CC_Glazur_150g_UNI_v181224E_etalon.txt"
TARGET_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "Lovita_CC_Glazur_150g_UNI_v181224E_label.txt"
OUTPUT_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "Lovita_CC_Glazur_150g_UNI_v181224E_label.out"

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Розгортаємо всі <span>, <b>, <i> тощо (залишаємо тільки текст)
    for tag in soup.find_all(['span', 'b', 'i']):
        tag.unwrap()

    # Збираємо текст з <div> і <p> без переносу всередині
    blocks = []
    for tag in soup.find_all(['div', 'p']):
        text = tag.get_text(separator='', strip=True)
        if text:
            blocks.append(text)

    return '\n'.join(blocks)



with open(ETALON_PATH, encoding='utf-8') as f:
    etalon_raw = f.read()
    text = html_to_text(etalon_raw)
    # Save the extracted text to ETALON_PATH + '_'
print(text)

exit()



# Функція для очищення тексту
def clean_text(text: str) -> str:
    text = text.replace('–', '-')              # довге тире → звичайне
    text = text.replace('՝', ':')              # вірменська двокрапка → стандартна
    text = text.replace('։', '.')              # вірменська крапка → стандартна
    text = text.replace('“', '"').replace('”', '"')  # лапки → звичайні

    text = re.sub(r'[.,;:()\""]', ' ', text)    # розділові знаки → пробіли
    text = re.sub(r'\s+', ' ', text)            # подвійні пробіли → один
    return text.strip()





# Завантаження та підготовка
with open(ETALON_PATH, encoding='utf-8') as f:
    etalon_raw = f.read()
    if '<body' in etalon_raw or '<html' in etalon_raw:
        etalon_raw = html_to_text(etalon_raw)
    # Save the extracted text to ETALON_PATH + '_'
    etalon_blocks = [clean_text(b) for b in re.split(r'\n{2,}|\n', etalon_raw) if b.strip()]

    etalon_save_path = ETALON_PATH.parent / (ETALON_PATH.name + '_')
    with open(etalon_save_path, 'w', encoding='utf-8') as etalon_out:
        etalon_out.write('\n'.join([clean_text(b) for b in etalon_blocks]))


with open(TARGET_PATH, encoding='utf-8') as f:
    target_text = clean_text(f.read())
    # Зберігаємо почищений текст у TARGET_PATH + '_'
    target_save_path = TARGET_PATH.parent / (TARGET_PATH.name + '_')
    with open(target_save_path, 'w', encoding='utf-8') as target_out:
        target_out.write(target_text)

results = []

# Пошук точних входжень
for idx, block in enumerate(tqdm(etalon_blocks, desc="Точний пошук"), 1):
    if block in target_text:
        status = 'found'
        matched_text = block
        target_text = target_text.replace(block, ' ' * len(block), 1)  # "вирізаємо"
    else:
        status = 'not found'
        matched_text = ''

    results.append({
        'block_id': idx,
        'searched_text': block,
        'matched_text': matched_text,
        'status': status
    })

# Запис у файл
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for row in results:
        f.write(f"Block ID: {row['block_id']}\n")
        f.write(f"Status: {row['status']}\n\n")
        f.write(f"Searched text:\n{row['searched_text']}\n\n")
        f.write(f"Matched text:\n{row['matched_text']}\n")
        f.write("-" * 40 + "\n\n")

# --- Output unmatched text from TARGET_PATH ---
# Clean up the remaining text (preserve line breaks, collapse only spaces/tabs)
unmatched_text = re.sub(r'[ \t]+', ' ', target_text)
# Remove lines that are only spaces
unmatched_text = re.sub(r'^[ ]+$', '', unmatched_text, flags=re.MULTILINE)
# Remove leading/trailing whitespace on each line and skip empty lines
unmatched_text = '\n'.join(line.strip() for line in unmatched_text.splitlines() if line.strip())

# Print to console
print("\n--- Unmatched text in TARGET_PATH ---\n")
print(unmatched_text)

# Save to a file
UNMATCHED_OUTPUT_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "Lovita_CC_Glazur_150g_UNI_v181224E_label_unmatched.txt"
with open(UNMATCHED_OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(unmatched_text)

# Вивід у консоль
pd.set_option('display.max_colwidth', None)
print(pd.DataFrame(results))



"""
def find_exact_matches(search_text, target_text):
    matches = []
    used_ranges = []

    def is_overlapping(start, end):
        for u_start, u_end in used_ranges:
            if not (end <= u_start or start >= u_end):
                return True
        return False

    words = search_text.split()
    n = len(words)

    for length in range(n, 0, -1):
        for i in range(n - length + 1):
            candidate = ' '.join(words[i:i+length])
            start_idx = target_text.find(candidate)
            if start_idx != -1:
                end_idx = start_idx + len(candidate)
                if not is_overlapping(start_idx, end_idx):
                    matches.append(candidate)
                    used_ranges.append((start_idx, end_idx))
    return matches


# приклад використання
search_text = """Склад: цукор, борошно пшеничне, жир рослинний (негідрогенізована пальмова олія, повністю гідрогенізована пальмоядрова олія, негідрогенізована пальмоядрова олія), крохмаль пшеничний, какао-порошок зі зниженим вмістом жиру 4%, меланж яєчний, молоко сухе знежирене, розпушувачі (гідрокарбонат амонію, дигідропірофосфат натрію, гідрокарбонат натрію), сіль, емульгатори лецитини (містить сою), ароматизатор, регулятор кислотності кислота молочна. Шматочки глазурі какаовмісної - 30%. Може містити арахіс, фундук, мигдаль, кунжут.

*Класичне печиво"""

with open("target.txt", encoding="utf-8") as f:
    target_text = f.read()

matches = find_exact_matches(search_text, target_text)
print(matches)


"""