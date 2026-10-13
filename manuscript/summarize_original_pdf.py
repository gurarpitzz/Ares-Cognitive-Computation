import fitz
import json

doc = fitz.open('d:/ares/Ares__Cognitive_computation.pdf')

page_summary = []
for pno in range(len(doc)):
    page = doc[pno]
    text = page.get_text('text')
    images = page.get_images()
    tables_found = [line for line in text.split('\n') if line.strip().startswith('TABLE ')]
    figures_found = [line for line in text.split('\n') if line.strip().startswith('Fig. ') or line.strip().startswith('Figure ')]
    sections_found = [line for line in text.split('\n') if any(line.strip().startswith(f'{r}.') for r in ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI', 'XXXII', 'XXXIII'])]
    
    page_summary.append({
        'page': pno + 1,
        'char_count': len(text),
        'image_count': len(images),
        'sections': sections_found,
        'tables': tables_found,
        'figures': figures_found,
        'first_100_chars': text[:100].replace('\n', ' ')
    })

with open('d:/ares/manuscript/page_structure_summary.json', 'w', encoding='utf-8') as f:
    json.dump(page_summary, f, indent=2)

print(f'Summarized {len(page_summary)} pages to page_structure_summary.json')
