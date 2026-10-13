import fitz

doc = fitz.open('d:/ares/ARES_Cognitive_Computation_Revised.pdf')
print('Total pages:', len(doc))

# Inspect page 1
p1 = doc[0]
print('\n--- PAGE 1 TEXT SAMPLE ---')
print(p1.get_text('text')[:800])

# Inspect insert page 16 (Section XIV-D Ethical Matrix)
p16 = doc[15]
print('\n--- PAGE 16 (INSERT 1) TEXT SAMPLE ---')
print(p16.get_text('text')[:800])

# Inspect insert page 28 (Section XXVI-E ARES-Synth v1)
p28 = doc[27]
print('\n--- PAGE 28 (INSERT 2) TEXT SAMPLE ---')
print(p28.get_text('text')[:800])

# Inspect insert page 35 (Section XXXI-B M15 Benchmark Tracks)
p35 = doc[34]
print('\n--- PAGE 35 (INSERT 3) TEXT SAMPLE ---')
print(p35.get_text('text')[:800])

# Inspect last page 43
p43 = doc[42]
print('\n--- PAGE 43 TEXT SAMPLE ---')
print(p43.get_text('text')[:800])
