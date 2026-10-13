import json

data = json.load(open('d:/ares/manuscript/matched_claims_in_pdf.json', encoding='utf-8'))
print(f'Total matched claims: {len(data)}')
for item in data:
    clean_meas = item['measured'][:60].encode('ascii', 'replace').decode('ascii')
    print(f"Claim {item['id']} ({item['section']}): '{item['reported']}' -> '{clean_meas}' on pages {item['pages']}")
