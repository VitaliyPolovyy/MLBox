from fuzzysearch import find_near_matches
from pathlib import Path
from mlbox.settings import ROOT_DIR
from bs4 import BeautifulSoup, NavigableString

CURRENT_DIR = Path(__file__).parent

ETALON_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "Lovita_CC_Glazur_150g_UNI_v181224E_etalon.txt"
LABEL_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "source" / "Lovita_CC_Glazur_150g_UNI_v181224E_label.org"
OUTPUT_PATH = ROOT_DIR / "assets" / "LabelGuard" / "input" / "Lovita_CC_Glazur_150g_UNI_v181224E_label.out"

def find_longest_common_substring(s1, s2):
    """
    Finds the longest common substring between two strings.

    Args:
        s1 (str): The first string.
        s2 (str): The second string.

    Returns:
        str: The longest common substring.
    """
    seq_matcher = SequenceMatcher(None, s1, s2)
    match = seq_matcher.find_longest_match(0, len(s1), 0, len(s2))

    if match.size == 0:
        return ""  # No common substring found
    else:
        return s1[match.a : match.a + match.size]

# Example usage
string1 = "GeeksforGeeks"
string2 = "GeeksQuiz"
lcs_result = find_longest_common_substring(string1, string2)
print(f"The longest common substring is: {lcs_result}")

string3 = "banana"
string4 = "bandana"
lcs_result_2 = find_longest_common_substring(string3, string4)
print(f"The longest common substring is: {lcs_result_2}")

""""""

# Read fragments from ETALON_PATH, split by newlines
with open(ETALON_PATH, encoding="utf-8") as f:
    fragments = [frag.strip() for frag in f.read().split("\n") if frag.strip()]

# Read label text from LABEL_PATH
with open(LABEL_PATH, encoding="utf-8") as f:
    label_text = f.read()


fragments = ["Բաղադրությունը՝ շաքար, ցորենի ալյուր, բուսական ճարպ (չհիդրոգենացված արմավենու յուղ, ամբողջովին հիդրոգենացված արկավենու միջուկի յուղ, չհիդրոգենացված արմավենու միջուկի յուղ), ցորենի օսլա, ցածր յուղայնության կակաոյի փոշի 4%, ձվի մելանժ, յուղազերծված կաթի փոշի,   փխրեցուցիչներ (ամոնիումի հիդրոկարբոնատ, նատրիումի հիդրոկարբոնատ, նատրիումի երկհիդրոպիրոֆոսֆատ), աղ, էմուլգատորներ՝ լեցիտիններ (պարունակում է սոյա), բուրավետիչ, թթվայնության կարգավորիչ՝ կաթնաթթու։ Ջնարակի կտորներ՝30%։ Կարող է պարունակել գետնանուշ, պնդուկ, նուշ, քունջութ։ Սննդային արժեքը 100գ մթերքում՝ սպիտակուցներ-4.3գ, ճարպեր-25.1գ, ածխաջրեր-63.2գ; էներգետիկ արժեքը (կալորիականությունը)՝ 2097կՋ/501կկալ։ Պահել (18±3) °С ջերմաստիճանի և 75%-ից ոչ բարձր օդի հարաբերական խոնավության պայմաններում: "]
label_text = """
(KZ) ГЛАЗУРЬ ТІЛІМДЕРІ БАР "LOVITA" CLASSIC COOKIES" МАЙҚОСПА ПЕЧЕНЬЕСІ. Құрамы: қант, бидай ұны, өсімдік майы (сутектендірілмеген  пальма майы, толықтай сутектендірілген пальма дәнінің майы, сутектендірілмеген пальма дәнінің майы), бидай крахмалы, майдың құрамы төмен какао-ұнтақ 4%, жұмыртқа меланж, майы алынған құрғақ сүт, қопсытқыштар (аммоний гидрокарбонаты, натрий дигидропирофосфаты, натрий гидрокарбонаты), ас тұзы, эмульгаторлар лецитиндер (құрамында соя бар), хошиістендіргіш, қышқылдықты реттеуіш сүт қышқылы. Құрамында какаосы бар глазурь тілімдері – 30%. Құрамында жержаңғақ, орман жаңғағы, бадам, күнжіт болуы мүмкін. (18±5)°С температура мен 75%-дан жоғары емес ауаның салыстырмалы ылғалдылығы жағдайында сақтау керек. Қазақстан Республикасындағы импорттаушы: "Кондитер-Азия" ЖШС, Қазақстан Республикасы, Алматы қ., Жетісу ауданы, Райымбек даңғылы, 169, 050050, Алматы қ., а/ж № 169. 
(AM) Թխվածքաբլիթ «Լովիտա կլասիկ քուքիս» ջնարակի կտորներով: Բաղադրությունը՝ շաքար, ցորենի ալյուր, բուսական ճարպ (չհիդրոգենացված արմավենու յուղ, ամբողջովին հիդրոգենացված արկավենու միջուկի յուղ, չհիդրոգենացված արմավենու միջուկի յուղ), ցորենի օսլա, ցածր յուղայնության կակաոյի փոշի 4%, ձվի մելանժ, յուղազերծված կաթի փոշի, փխրեցուցիչներ (ամոնիումի հիդրոկարբոնատ, նատրիումի հիդրոկարբոնատ, նատրիումի երկհիդրոպիրոֆոսֆատ), աղ, էմուլգատորներ՝ լեցիտիններ (պարունակում է սոյա), բուրավետիչ, թթվայնության կարգավորիչ՝ կաթնաթթու։ Ջնարակի կտորներ՝ 30%։ Կարող է պարունակել գետնանուշ, պնդուկ, նուշ, քունջութ։ Պահել (18±5)°С ջերմաստիճանի և 75%-ից ոչ բարձր օդի հարաբերական խոնավության պայմաններում: Հայաստանում Ներմուծող` «Դի-Դի-Թրեյդ» ՍՊԸ, ՀՀ, ք. Երևան, Ար. Միկոյան 37/8։ 
(GE) ორცხობილა «ლოვიტა» კლასიკ ქუქის» სარკალას ნატეხებით. ინგრედიენტები: შაქარი, ხორბლის ფქვილი, მცენარეული ცხიმი (არაჰიდროგენირებული პალმის ზეთი, სრულად ჰიდროგენირებული პალმის გულის ზეთი, არაჰიდროგენირებული პალმის გულის ზეთი), ხორბლის სახამებელი, კაკაოს ფხვნილი ცხიმის დაბალი შემცველობით 4%, კვერცხის მელანჟი, უცხიმო რძის ფხვნილი, გამაფხვიერებელი კომპონენტები (ამონიუმის ჰიდროკარბონატი, ნატრიუმის დიჰიდროპიროფოსფატი, ნატრიუმის ჰიდროკარბონატი), მარილი, ემულგატორები ლეციტინები (შეიცავს სოიოს), არომატიზატორი, მჟავიანობის რეგულატორი რძემჟავა. კაკაოს შემცველი მინანქრის ნატეხები – 30%. შეიძლება შეიცავდეს არაქისს, თხილს, ნუშს, სეზამის მარცვლებს. ინახება (18±5)°С ტემპერატურისა და არაუმეტეს 75% ჰაერის ფარდობითი ტენიანობის პირობებში. ოფიციალური იმპორტიორი საქართველოში: შპს "როშენ ჯორჯია" საქართველო, ქ. თბილისი, სამგორის რაიონი, თენგიზ ჩანტლაძის ქ., N40/დიდი ლილო, თეთრი ხევის დასახლება, ტელ. +995 322 98 98 28. დამზადებულია უკრაინაში.
"""

for fragment in fragments:
    print(f"Searching for fragment: {fragment}")
    result = find_near_matches(fragment, label_text, max_l_dist=150)
    if result:
        for r in result:
            print("  Matched text:", r.matched)
            print("  Start index:", r.start)
            print("  End index:", r.end)
            print("  Levenshtein distance:", r.dist)
    else:
        print("  No match found.")


"""
results = []

for idx, fragment in enumerate(fragments):
    result = find_near_matches(fragment, text, max_l_dist=2)
    found = bool(result)
    results.append((idx, found))

for idx, found in results:
    status = "found" if found else "not found"
    print(f"{idx} / {status}")
"""