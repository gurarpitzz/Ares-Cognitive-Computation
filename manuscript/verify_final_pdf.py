import fitz

doc = fitz.open('d:/ares/ARES_Cognitive_Computation_Revised.pdf')
print('Total pages:', len(doc))

for idx, label in [(0, 'PAGE 1'), (15, 'PAGE 16 (INSERT 1)'), (27, 'PAGE 28 (INSERT 2)'), (34, 'PAGE 35 (INSERT 3)'), (42, 'PAGE 43 (LAST PAGE)')]:
    print(f"\n--- {label} TEXT SAMPLE ---")
    txt = doc[idx].get_text('text')[:600]
    print(txt.encode('ascii', 'replace').decode('ascii'))
