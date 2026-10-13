import json
import fitz

doc = fitz.open('d:/ares/Ares__Cognitive_computation.pdf')
with open('d:/ares/manuscript/claims_actions_summary.json', 'r', encoding='utf-8') as f:
    claims = json.load(f)

matched = 0
results = []
for c in claims:
    rep = c['reported'].strip()
    if not rep or len(rep) < 2 or rep.lower() in ['n/a', 'none', 'varies']:
        continue
    found_pages = []
    for pno in range(len(doc)):
        r = doc[pno].search_for(rep)
        if r:
            found_pages.append(pno + 1)
    if found_pages:
        matched += 1
        results.append({
            'id': c['id'],
            'section': c['section'],
            'reported': rep,
            'measured': c['measured'],
            'ci95': c['ci95'],
            'pages': found_pages,
            'action': c['action']
        })

print(f"Matched {matched} claims by reported string in PDF")
with open('d:/ares/manuscript/matched_claims_in_pdf.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

for r in results[:20]:
    print(f"{r['id']} ({r['section']}): '{r['reported']}' -> '{r['measured']}' on pages {r['pages']}")
