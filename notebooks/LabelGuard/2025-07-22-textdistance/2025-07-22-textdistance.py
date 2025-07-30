import textdistance
import time
from typing import List, Dict, Tuple
import html

# ANSI коди для кольорів в консолі
class Colors:
    COMMON = '\033[42m\033[30m'      # Зелений фон, чорний текст
    UNIQUE_1 = '\033[101m\033[37m'   # Червоний фон, білий текст  
    UNIQUE_2 = '\033[101m\033[37m'   # Синій фон, білий текст
    RESET = '\033[0m'                # Скидання
    BOLD = '\033[1m'
    HEADER = '\033[95m'

# Тестові тексти
text1 = """
ინგრედიენტები: შაქარი, ხორბლის ფქვილი, მცენარეული ცხიმი (არაჰიდროგენირებული პალმის ზეთი, სრულად ჰიდროგენირებული პალმის გულის ზეთი, არაჰიდროგენირებული პალმის გულის ზეთი), ხორბლის სახამებელი, კაკაოს ფხვნილი ცხიმის დაბალი შემცველობით 4%, კვერცხის მელანჟი, უცხიმო რძის ფხვნილი, გამაფხვიერებელი კომპონენტები (ამონიუმის ჰიდროკარბონატი, ნატრიუმის დიჰიდროპიროფოსფატი, ნატრიუმის ჰიდროკარბონატი), მარილი, ემულგატორები ლეციტინები (შეიცავს სოიოს), არომატიზატორი, მჟავიანობის რეგულატორი რძემჟავა. კაკაოს შემცველი მინანქრის ნატეხები - 30%. შეიძლება შეიცავდეს არაქისს, თხილს, ნუშს, სეზამის მარცვლებს.
* კლასიკური ორცხობილა 
კვებითი ღირებულება 100 გრამ პროდუქტზე: ენერგეტიკული ღირებულება – 2097 კილოჯოული (501 კკალ); ცხიმი - 25.1 გრამი; მათ შორის ნაჯერი ცხიმოვანი მჟავები - 14.6 გრამი; ნახშირწყლები - 63.2 გრამი; მათ შორის შაქრები – 36.5 გრამი; ცილა - 4.3 გრამი; მარილი - 0.63 გრამი.
ინახება (18±5)°С ტემპერატურისა და არაუმეტეს 75% ჰაერის ფარდობითი ტენიანობის პირობებში.
"""
text2 = """
(GE) ორცხობილა «ლოვიტა» კლასიკ ქუქის» სარკალას ნატეხებით. ინგრედიენტები: შაქარი, ხორბლის ფქვილი, მცენარეული ცხიმი (არაჰიდროგენირებული პალმის ზეთი, სრულად ჰიდროგენირებული პალმის გულის ზეთი, არაჰიდროგენირებული პალმის გულის ზეთი), ხორბლის სახამებელი, კაკაოს ფხვნილი ცხიმის დაბალი შემცველობით 4%, კვერცხის მელანჟი, უცხიმო რძის ფხვნილი, გამაფხვიერებელი კომპონენტები (ამონიუმის ჰიდროკარბონატი, ნატრიუმის დიჰიდროპიროფოსფატი, ნატრიუმის ჰიდროკარბონატი), მარილი, ემულგატორები ლეციტინები (შეიცავს სოიოს), არომატიზატორი, მჟავიანობის რეგულატორი რძემჟავა. კაკაოს შემცველი მინანქრის ნატეხები – 30%. შეიძლება შეიცავდეს არაქისს, თხილს, ნუშს, სეზამის მარცვლებს. ინახება (18±5)°С ტემპერატურისა და არაუმეტეს 75% ჰაერის ფარდობითი ტენიანობის პირობებში. ოფიციალური იმპორტიორი საქართველოში: შპს "როშენ ჯორჯია" საქართველო, ქ. თბილისი, სამგორის რაიონი, თენგიზ ჩანტლაძის ქ., N40/დიდი ლილო, თეთრი ხევის დასახლება, ტელ. +995 322 98 98 28. დამზადებულია უკრაინაში. 
"""

def find_all_common_fragments(text1: str, text2: str, min_length: int = 10, max_iterations: int = 20) -> List[Dict]:
    """Знаходить всі спільні фрагменти між двома текстами"""
    results = []
    working_text1 = text1
    working_text2 = text2
    
    for iteration in range(max_iterations):
        print(f"🔄 Ітерація {iteration + 1}...", end=" ")
        
        # Знаходимо найдовший спільний фрагмент
        common = textdistance.lcsstr(working_text1, working_text2)
        
        # Якщо нічого або занадто короткий - зупиняємося
        if not common or len(common) < min_length:
            print("❌ Більше нічого не знайдено")
            break
            
        # Показуємо знайдений фрагмент (скорочено)
        preview = common[:50] + ("..." if len(common) > 50 else "")
        print(f"✅ Знайдено: '{preview}' ({len(common)} симв.)")
        
        # Знаходимо позиції в оригінальних текстах
        pos1 = text1.find(common)
        pos2 = text2.find(common)
        
        # Зберігаємо результат
        results.append({
            'text': common,
            'pos1': pos1,
            'pos2': pos2,
            'length': len(common),
            'iteration': iteration + 1
        })
        
        # Замінюємо знайдений фрагмент на різні символи
        placeholder1 = '◆' * len(common)
        placeholder2 = '◇' * len(common)
        
        working_text1 = working_text1.replace(common, placeholder1, 1)
        working_text2 = working_text2.replace(common, placeholder2, 1)
    
    print(f"\n🎯 Загалом знайдено: {len(results)} спільних фрагментів")
    return sorted(results, key=lambda x: x['length'], reverse=True)

def create_markup_data(text1: str, text2: str, common_fragments: List[Dict]) -> Tuple[List, List]:
    """Створює дані для розмітки текстів (спільні/унікальні частини)"""
    
    def mark_text(text: str, fragments: List[Dict], text_num: int) -> List[Dict]:
        """Розмічає один текст на спільні та унікальні частини"""
        marked = []
        last_pos = 0
        
        # Сортуємо фрагменти за позицією
        sorted_fragments = sorted(
            [f for f in fragments if f[f'pos{text_num}'] != -1], 
            key=lambda x: x[f'pos{text_num}']
        )
        
        for fragment in sorted_fragments:
            pos = fragment[f'pos{text_num}']
            length = fragment['length']
            
            # Додаємо унікальну частину перед спільним фрагментом
            if pos > last_pos:
                unique_text = text[last_pos:pos]
                if unique_text.strip():  # Ігноруємо порожні рядки
                    marked.append({
                        'text': unique_text,
                        'type': f'unique_{text_num}',
                        'start': last_pos,
                        'end': pos
                    })
            
            # Додаємо спільний фрагмент
            marked.append({
                'text': fragment['text'],
                'type': 'common',
                'start': pos,
                'end': pos + length,
                'fragment_data': fragment
            })
            
            last_pos = pos + length
        
        # Додаємо останню унікальну частину
        if last_pos < len(text):
            unique_text = text[last_pos:]
            if unique_text.strip():
                marked.append({
                    'text': unique_text,
                    'type': f'unique_{text_num}',
                    'start': last_pos,
                    'end': len(text)
                })
        
        return marked
    
    markup1 = mark_text(text1, common_fragments, 1)
    markup2 = mark_text(text2, common_fragments, 2)
    
    return markup1, markup2

def visualize_console(text1: str, text2: str, common_fragments: List[Dict], max_line_length: int = 100):
    """Візуалізує різниці в консолі з кольорами"""
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "="*120)
    print("🎨 ВІЗУАЛІЗАЦІЯ РІЗНИЦЬ МІЖ ТЕКСТАМИ")
    print("="*120 + Colors.RESET)
    
    # Легенда
    print(f"\n📋 Легенда:")
    print(f"   {Colors.COMMON}СПІЛЬНІ ФРАГМЕНТИ{Colors.RESET}")
    print(f"   {Colors.UNIQUE_1}УНІКАЛЬНІ В ТЕКСТІ 1{Colors.RESET}")
    print(f"   {Colors.UNIQUE_2}УНІКАЛЬНІ В ТЕКСТІ 2{Colors.RESET}")
    
    markup1, markup2 = create_markup_data(text1, text2, common_fragments)
    
    def print_marked_text(markup: List[Dict], title: str):
        print(f"\n{Colors.BOLD}{title}{Colors.RESET}")
        print("-" * len(title))
        
        current_line = ""
        
        for part in markup:
            # Вибираємо колір
            if part['type'] == 'common':
                color = Colors.COMMON
            elif part['type'] == 'unique_1':
                color = Colors.UNIQUE_1
            else:  # unique_2
                color = Colors.UNIQUE_2
            
            text_part = part['text']
            
            # Розбиваємо на рядки, якщо занадто довго
            words = text_part.split()
            for word in words:
                if len(current_line) + len(word) + 1 > max_line_length:
                    print(current_line)
                    current_line = ""
                
                if current_line:
                    current_line += " "
                current_line += f"{color}{word}{Colors.RESET}"
        
        if current_line:
            print(current_line)
    
    print_marked_text(markup1, "📄 ТЕКСТ 1:")
    print_marked_text(markup2, "📄 ТЕКСТ 2:")

def visualize_side_by_side(text1: str, text2: str, common_fragments: List[Dict], width: int = 60):
    """Візуалізує тексти поруч для порівняння"""
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}" + "="*140)
    print("👥 ПОРІВНЯННЯ ТЕКСТІВ SIDE-BY-SIDE")
    print("="*140 + Colors.RESET)
    
    markup1, markup2 = create_markup_data(text1, text2, common_fragments)
    
    def split_into_lines(markup: List[Dict], width: int) -> List[str]:
        """Розбиває розмічений текст на рядки заданої ширини"""
        lines = []
        current_line = ""
        
        for part in markup:
            if part['type'] == 'common':
                color = Colors.COMMON
            elif part['type'] == 'unique_1':
                color = Colors.UNIQUE_1
            else:
                color = Colors.UNIQUE_2
            
            words = part['text'].split()
            for word in words:
                colored_word = f"{color}{word}{Colors.RESET}"
                # Рахуємо довжину без ANSI кодів
                word_length = len(word)
                
                if len(current_line.replace(Colors.COMMON, '').replace(Colors.UNIQUE_1, '').replace(Colors.UNIQUE_2, '').replace(Colors.RESET, '')) + word_length + 1 > width:
                    lines.append(current_line)
                    current_line = colored_word
                else:
                    if current_line:
                        current_line += " "
                    current_line += colored_word
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    lines1 = split_into_lines(markup1, width)
    lines2 = split_into_lines(markup2, width)
    
    # Заголовки
    print(f"{Colors.BOLD}{'ТЕКСТ 1':<{width}} │ {'ТЕКСТ 2':<{width}}{Colors.RESET}")
    print("─" * width + "┼" + "─" * width)
    
    # Виводимо рядки
    max_lines = max(len(lines1), len(lines2))
    for i in range(max_lines):
        line1 = lines1[i] if i < len(lines1) else ""
        line2 = lines2[i] if i < len(lines2) else ""
        
        # Підрахунок реальної довжини без ANSI кодів для правильного вирівнювання
        clean_line1 = line1.replace(Colors.COMMON, '').replace(Colors.UNIQUE_1, '').replace(Colors.UNIQUE_2, '').replace(Colors.RESET, '')
        padding1 = width - len(clean_line1)
        
        print(f"{line1}{' ' * padding1} │ {line2}")

def generate_html_report(text1: str, text2: str, common_fragments: List[Dict], filename: str = "text_comparison.html"):
    """Генерує HTML звіт для візуалізації в браузері"""
    
    markup1, markup2 = create_markup_data(text1, text2, common_fragments)
    
    def markup_to_html(markup: List[Dict]) -> str:
        """Конвертує розмітку в HTML"""
        html_parts = []
        
        for part in markup:
            text = html.escape(part['text'])
            
            if part['type'] == 'common':
                html_parts.append(f'<span class="common" title="Спільний фрагмент">{text}</span>')
            elif part['type'] == 'unique_1':
                html_parts.append(f'<span class="unique1" title="Тільки в тексті 1">{text}</span>')
            else:  # unique_2
                html_parts.append(f'<span class="unique2" title="Тільки в тексті 2">{text}</span>')
        
        return ''.join(html_parts)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="uk">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Порівняння текстів</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5;
            }}
            .container {{ 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            .header {{ 
                text-align: center; 
                color: #333; 
                border-bottom: 2px solid #ddd;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .legend {{ 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 5px; 
                margin-bottom: 30px;
                border-left: 4px solid #007bff;
            }}
            .legend h3 {{ margin-top: 0; color: #007bff; }}
            .text-section {{ 
                margin-bottom: 40px; 
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .text-header {{ 
                background: #007bff; 
                color: white; 
                padding: 15px; 
                font-weight: bold;
                border-radius: 5px 5px 0 0;
            }}
            .text-content {{ 
                padding: 20px; 
                line-height: 1.6; 
                font-size: 16px;
            }}
            .common {{ 
                background-color: #d4edda; 
                padding: 2px 4px; 
                border-radius: 3px;
                border: 1px solid #c3e6cb;
            }}
            .unique1 {{ 
                background-color: #f8d7da; 
                padding: 2px 4px; 
                border-radius: 3px;
                border: 1px solid #f5c6cb;
            }}
            .unique2 {{ 
                background-color: #cce7ff; 
                padding: 2px 4px; 
                border-radius: 3px;
                border: 1px solid #99d3ff;
            }}
            .stats {{ 
                background: #e9ecef; 
                padding: 15px; 
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .side-by-side {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin-top: 30px;
            }}
            @media (max-width: 768px) {{
                .side-by-side {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔍 Порівняння текстів</h1>
                <p>Аналіз спільних та унікальних фрагментів</p>
            </div>
            
            <div class="legend">
                <h3>📋 Легенда</h3>
                <p>
                    <span class="common">Спільні фрагменти</span> - 
                    <span class="unique1">Унікальні в тексті 1</span> - 
                    <span class="unique2">Унікальні в тексті 2</span>
                </p>
            </div>
            
            <div class="stats">
                <strong>📊 Статистика:</strong><br>
                Знайдено спільних фрагментів: {len(common_fragments)}<br>
                Загальна довжина спільних фрагментів: {sum(f['length'] for f in common_fragments)} символів<br>
                Покриття тексту 1: ~{(sum(f['length'] for f in common_fragments)/len(text1)*100):.1f}%<br>
                Покриття тексту 2: ~{(sum(f['length'] for f in common_fragments)/len(text2)*100):.1f}%
            </div>
            
            <div class="side-by-side">
                <div class="text-section">
                    <div class="text-header">📄 Текст 1</div>
                    <div class="text-content">{markup_to_html(markup1)}</div>
                </div>
                
                <div class="text-section">
                    <div class="text-header">📄 Текст 2</div>
                    <div class="text-content">{markup_to_html(markup2)}</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ HTML звіт збережено в файл: {filename}")
    return filename

def print_results(results: List[Dict], text1: str, text2: str):
    """Виводить результати в зручному форматі"""
    
    print("\n" + "="*80)
    print("📊 РЕЗУЛЬТАТИ ПОШУКУ СПІЛЬНИХ ФРАГМЕНТІВ")
    print("="*80)
    
    total_length = sum(r['length'] for r in results)
    print(f"🔢 Кількість фрагментів: {len(results)}")
    print(f"📏 Загальна довжина: {total_length} символів")
    print(f"📈 Покриття text1: ~{(total_length/len(text1)*100):.1f}%")
    print(f"📈 Покриття text2: ~{(total_length/len(text2)*100):.1f}%")
    print()
    
    for i, result in enumerate(results, 1):
        print(f"{i}. ФРАГМЕНТ №{i} (довжина: {result['length']})")
        
        # Показуємо текст фрагмента
        if len(result['text']) > 100:
            print(f"   📝 Текст: '{result['text'][:50]}...{result['text'][-50:]}'")
        else:
            print(f"   📝 Текст: '{result['text']}'")
        
        print(f"   📍 Позиції: text1[{result['pos1']}], text2[{result['pos2']}]")
        
        # Перевіряємо правільність
        if result['pos1'] != -1 and result['pos2'] != -1:
            check1 = text1[result['pos1']:result['pos1'] + result['length']]
            check2 = text2[result['pos2']:result['pos2'] + result['length']]
            is_correct = check1 == check2 == result['text']
            print(f"   ✅ Перевірка: {'✓ Правильно' if is_correct else '✗ Помилка!'}")
        else:
            print(f"   ⚠️  Попередження: фрагмент не знайдено в одному з текстів")
        
        print()

def find_and_visualize_differences(text1: str, text2: str, min_length: int = 10, max_iterations: int = 20, 
                                 console_vis: bool = True, side_by_side: bool = True, 
                                 html_report: bool = True, html_filename: str = "comparison.html"):
    """ГОЛОВНА ФУНКЦІЯ - знаходить фрагменти та створює всі види візуалізації"""
    
    print(f"{Colors.BOLD}🚀 ЗАПУСК ПОВНОГО АНАЛІЗУ ТЕКСТІВ{Colors.RESET}")
    
    # 1. Знаходимо спільні фрагменти
    start_time = time.perf_counter()
    results = find_all_common_fragments(text1, text2, min_length, max_iterations)
    end_time = time.perf_counter()
    
    print(f"\n⏱️  Час виконання пошуку: {end_time - start_time:.4f} секунд")
    
    if html_report and results:
        generate_html_report(text1, text2, results, html_filename)
    
    return results

# Тестування
if __name__ == "__main__":
    print("🔍 Аналізатор текстів з візуалізацією v2.0\n")
    
    # Запускаємо повний аналіз
    results = find_and_visualize_differences(
        text1, text2, 
        min_length=15,
        console_vis=True,
        side_by_side=True, 
        html_report=True,
        html_filename="armenian_text_comparison.html"
    )
    
    print(f"\n{Colors.BOLD}✨ Аналіз завершено! Знайдено {len(results)} спільних фрагментів{Colors.RESET}")