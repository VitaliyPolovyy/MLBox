import time
from datetime import datetime
from pathlib import Path
from mlbox.settings import ROOT_DIR
import difflib
import re
from rich.console import Console
from rich.text import Text

text1 = "Բաղադրությունը՝ շաքար, ցորենի ալյուր, բուսական ճարպ (չհիդրոգենացված արմավենու յուղ, ամբողջովին հիդրոգենացված արկավենու միջուկի յուղ, չհիդրոգենացված արմավենու միջուկի յուղ), ցորենի օսլա, ցածր յուղայնության կակաոյի փոշի 4%, ձվի մելանժ, յուղազերծված կաթի փոշի,   փխրեցուցիչներ (ամոնիումի հիդրոկարբոնատ, նատրիումի հիդրոկարբոնատ, նատրիումի երկհիդրոպիրոֆոսֆատ), աղ, էմուլգատորներ՝ լեցիտիններ (պարունակում է սոյա), բուրավետիչ, թթվայնության կարգավորիչ՝ կաթնաթթու։ Ջնարակի կտորներ՝30%։ Կարող է պարունակել գետնանուշ, պնդուկ, նուշ, քունջութ։ Սննդային արժեքը 100գ մթերքում՝ սպիտակուցներ-4.3գ, ճարպեր-25.1գ, ածխաջրեր-63.2գ; էներգետիկ արժեքը (կալորիականությունը)՝ 2097կՋ/501կկալ։ Պահել (18±3) °С ջերմաստիճանի և 75%-ից ոչ բարձր օդի հարաբերական խոնավության պայմաններում: "
text2 = "(AM) Թխվածքաբլիթ «Լովիտա կլասիկ քուքիս» ջնարակի կտորներով: Բաղադրությունը՝ շաքար, ցորենի ալյուր, բուսական ճարպ (չհիդրոգենացված արմավենու յուղ, ամբողջովին հիդրոգենացված արկավենու միջուկի յուղ, չհիդրոգենացված արմավենու միջուկի յուղ), ցորենի օսլա, ցածր յուղայնության կակաոյի փոշի 4%, ձվի մելանժ, յուղազերծված կաթի փոշի, փխրեցուցիչներ (ամոնիումի հիդրոկարբոնատ, նատրիումի հիդրոկարբոնատ, նատրիումի երկհիդրոպիրոֆոսֆատ), աղ, էմուլգատորներ՝ լեցիտիններ (պարունակում է սոյա), բուրավետիչ, թթվայնության կարգավորիչ՝ կաթնաթթու։ Ջնարակի կտորներ՝ 30%։ Կարող է պարունակել գետնանուշ, պնդուկ, նուշ, քունջութ։ Պահել (18±5)°С ջերմաստիճանի և 75%-ից ոչ բարձր օդի հարաբերական խոնավության պայմաններում: Հայաստանում Ներմուծող` «Դի-Դի-Թրեյդ» ՍՊԸ, ՀՀ, ք. Երևան, Ար. Միկոյան 37/8։ "


import textdistance

def find_all_common_substrings(text1, text2, min_length=3):
    """Знаходить всі спільні підрядки"""
    
    results = []
    working_text1 = text1
    working_text2 = text2
    
    while True:
        # Знаходимо найдовший спільний підрядок
        common = textdistance.lcsstr(working_text1, working_text2)
        
        # Якщо нічого або занадто короткий - зупиняємося
        if not common or len(common) < min_length:
            break
        
        # Знаходимо позиції в оригінальних текстах
        pos1 = text1.find(common)
        pos2 = text2.find(common)
        
        # Зберігаємо результат
        results.append({
            'text': common,
            'pos1': pos1,
            'pos2': pos2,
            'length': len(common)
        })
        
        # Замінюємо на різні символи
        working_text1 = working_text1.replace(common, '◆' * len(common), 1)
        working_text2 = working_text2.replace(common, '◇' * len(common), 1)
    
    # Сортуємо за довжиною (найдовші першими)
    return sorted(results, key=lambda x: x['length'], reverse=True)

current_time = datetime.now()
results = find_all_common_substrings(text1, text2, min_length=3)
print(datetime.now() - current_time)
    
print("Знайдені спільні фрагменти:")
for i, result in enumerate(results, 1):
    print(f"{i}. '{result['text']}' (довжина: {result['length']})")
    print(f"   Позиції: text1[{result['pos1']}], text2[{result['pos2']}]")

exit()



CURRENT_DIR = Path(__file__).parent

ETALON_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "Lovita_CC_Glazur_150g_UNI_v181224E_etalon.txt"
LABEL_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "source" / "Lovita_CC_Glazur_150g_UNI_v181224E_label.org_"
OUTPUT_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "Lovita_CC_Glazur_150g_UNI_v181224E_label.out"


# Read fragments from ETALON_PATH, split by newlines
with open(ETALON_PATH, encoding="utf-8") as f:
    fragments = [frag.strip() for frag in f.read().split("\n") if frag.strip()]


# Read label text from LABEL_PATH
with open(LABEL_PATH, encoding="utf-8") as f:
    label_text = f.read()

# Далі код працює з уже нормалізованими даними
def normalize_text(text):
    """Нормалізує текст, включаючи проблемні символи"""
    
    #text = text.replace('%', '')
    #text = text.replace('±', '')
    #text = text.replace('°', '')
    #text = text.replace('(', '')
    #text = text.replace(')', '')
    #text = text.replace('С', '')
    
    
    # Стандартна нормалізація
    text = text.replace(' – ', ' - ').replace('–', '-').replace('—', '-')
    #text = re.sub(r'\s+', ' ', text)
    return text.strip()

def IsJunk(x):
    return x == '\n' or x == '*' or x == ' ' 

def find_common_blocks(fragment, label_text):
    fragment = normalize_text(fragment)
    label_text = normalize_text(label_text)

    matcher = difflib.SequenceMatcher(IsJunk, fragment, label_text)
    matching_blocks = matcher.get_matching_blocks()
    
    # Отримуємо фрагменти, що збігаються
    common_parts = []
    for match in matching_blocks:
        if match.size > 10:  # мінімальна довжина фрагменту
            common_text = fragment[match.a:match.a + match.size]
            common_parts.append({
                'text': common_text,
                'fragment_pos': (match.a, match.a + match.size),
                'label_pos': (match.b, match.b + match.size),
                'length': match.size
            })
    
    return common_parts


def visualize_fragments(label_text, fragments, found_indices):
    """
    Підсвічує знайдені фрагменти зеленим у label_text, незнайдені виводить жовтим.
    found_indices: список (start, end) для знайдених фрагментів у label_text
    """
    console = Console()
    rich_text = Text(label_text)
    for start, end in found_indices:
        if start is not None and end is not None:
            rich_text.stylize("bold green", start, end)
    console.print("\n[bold underline]label_text with highlights:[/bold underline]\n")
    console.print(rich_text)


def visualize_common_blocks(label, common_blocks):
    console = Console()
    rich_text = Text(label)
    for block in common_blocks:
        start = block['label_pos'][0]
        end = block['label_pos'][1]
        if start is not None and end is not None and end > start:
            rich_text.stylize("bold green", start, end)
    console.print("\n[bold underline]label (row) with highlights:[/bold underline]\n")
    console.print(rich_text)


if __name__ == "__main__":

    # Нормалізуємо label_text і fragments на самому початку
    label_text = normalize_text(label_text)
    fragments = [normalize_text(frag) for frag in fragments]

    labels = label_text.split('\n')
    found_indices = []
    for i, fragment in enumerate(fragments):
        fragment_stripped = fragment.strip()
        start = label_text.find(fragment_stripped)
        if start != -1:
            end = start + len(fragment_stripped)
            found_indices.append((start, end))
        else:
            found_indices.append((None, None))

    visualize_fragments(label_text, fragments, found_indices)

    # Далі аналіз common_blocks по рядках з підсвічуванням
    for i, fragment in enumerate(fragments):
        print("\n=== Searching text ===")
        print(fragment)
        for k, label in enumerate(labels):
            common_blocks = find_common_blocks(fragment, label)
            if common_blocks:
                visualize_common_blocks(label, common_blocks)
            for j, block in enumerate(common_blocks, 1):
                print(f"Block founded{j}:")
                print(f"  Text: {block['text']}")
                print(f"  Fragment position: {block['fragment_pos']}")
                print(f"  Label position: {block['label_pos']}")
                print(f"  Length: {block['length']}")
                print("----------------------")

# === RapidFuzz LCSseq example ===
text1 = "*Класичне печиво"
text2 = "(UA) ПЕЧИВО ЗДОБНЕ LOVITA CLASSIC COOKIES З КУСОЧКАМИ ГЛАЗУРІ. Склад: цукор, борошно пшеничне, жир рослинний (негідрогенізована пальмова олія, повністю гідрогенізована пальмоядрова олія, негідрогенізована пальмоядрова олія), крохмаль пшеничний, какао-порошок зі зниженим вмістом жиру 4%, меланж яєчний, молоко сухе знежирене, розпушувачі (гідрокарбонат амонію, дигідропірофосфат натрію, гідрокарбонат натрію), сіль, емульгатори лецитини (містить сою), ароматизатор, регулятор кислотності кислота молочна. Шматочки глазурі какаовмісної – 30%. Може містити арахіс, фундук, мигдаль, кунжут. Зберігати за температури (18±5)°С і відносної вологості повітря не вище 75%. (v181224E) Класичне печиво. Какао. "

# Знаходимо LCS та індекси
lcs_len, indices1, indices2 = LCSseq.lcs(text1, text2, return_indices=True)
lcs_str = ''.join([text1[i] for i in indices1])
print(f"LCS length: {lcs_len}")
print(f"LCS indices in text1: {indices1}")
print(f"LCS indices in text2: {indices2}")
print(f"Longest common subsequence: '{lcs_str}'")

