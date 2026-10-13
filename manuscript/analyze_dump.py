import json

with open('d:/ares/manuscript/original_manuscript_dump.json', 'r', encoding='utf-8') as f:
    dump = json.load(f)

print('Total pages:', len(dump))
total_chars = sum(len(p['text']) for p in dump)
print('Total chars:', total_chars, 'Approx words:', total_chars // 6)

for i, p in enumerate(dump):
    lines = [l.strip() for l in p['text'].split('\n') if l.strip()]
    first3 = ' | '.join(lines[:2])
    print(f'P{i+1:02d}: chars={len(p["text"])}: {first3[:75]}')
