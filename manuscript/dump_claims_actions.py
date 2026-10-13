import csv
import json

with open('d:/ares/plan/claims_ledger.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    claims = list(reader)

summary = []
for c in claims:
    summary.append({
        'id': c['claim_id'],
        'section': c['section'],
        'status': c['status'],
        'reported': c['reported_value'],
        'measured': c['measured_value'],
        'ci95': c['ci95'],
        'action': c['manuscript_action'],
        'text': c['claim_text'][:80]
    })

with open('d:/ares/manuscript/claims_actions_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2)

print(f'Wrote {len(summary)} claim actions to claims_actions_summary.json')
