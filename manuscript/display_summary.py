import json

with open('d:/ares/manuscript/page_structure_summary.json', 'r', encoding='utf-8') as f:
    pages = json.load(f)

for p in pages:
    sec = '; '.join(p['sections']) if p['sections'] else '-'
    tab = '; '.join(p['tables']) if p['tables'] else '-'
    fig = '; '.join(p['figures']) if p['figures'] else '-'
    print(f"P{p['page']:02d}: Img:{p['image_count']} | Sec: {sec[:45]} | Tab: {tab[:30]} | Fig: {fig[:30]}")
