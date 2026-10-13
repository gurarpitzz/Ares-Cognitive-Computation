import json

with open('d:/ares/manuscript/page_structure_summary.json', 'r', encoding='utf-8') as f:
    pages = json.load(f)

for p in pages[30:]:
    sec = '; '.join(p['sections']) if p['sections'] else '-'
    tab = '; '.join(p['tables']) if p['tables'] else '-'
    fig = '; '.join(p['figures']) if p['figures'] else '-'
    # replace non-ascii
    clean_sec = sec.encode('ascii', 'replace').decode('ascii')
    clean_tab = tab.encode('ascii', 'replace').decode('ascii')
    clean_fig = fig.encode('ascii', 'replace').decode('ascii')
    print(f"P{p['page']:02d}: Sec: {clean_sec[:45]} | Tab: {clean_tab[:30]} | Fig: {clean_fig[:30]}")
