import fitz

doc = fitz.open('d:/ares/Ares__Cognitive_computation.pdf')

figs = [
    (3, 'Fig. 3:'),
    (4, 'Fig. 6:'),
    (7, 'Fig. 10:'),
    (7, 'Fig. 11:'),
    (7, 'Fig. 12:'),
    (8, 'Fig. 14:'),
    (8, 'Fig. 15:'),
    (8, 'Fig. 16:'),
    (11, 'Fig. 19:'),
    (13, 'Fig. 22:'),
    (19, 'Fig. 28:'),
    (20, 'Fig. 29:'),
    (32, 'Fig. 38:'),
    (32, 'Fig. 39:'),
    (32, 'Fig. 40:'),
    (35, 'Fig. 41:'),
    (35, 'Fig. 42:')
]

for pno, cap in figs:
    p = doc[pno]
    rects = p.search_for(cap)
    if rects:
        r = rects[0]
        # check preceding text block or table above it to find top
        blocks = p.get_text('blocks')
        # find blocks in same column that are above r.y0
        col_x = 300 if r.x0 > 300 else 0
        above_blocks = [b for b in blocks if (b[0] > 300 if col_x > 0 else b[0] < 300) and b[3] < r.y0]
        prev_bottom = max((b[3] for b in above_blocks), default=50.0)
        gap = r.y0 - prev_bottom
        print(f"P{pno+1} {cap:8s}: Caption y0={r.y0:.1f}, Prev bottom={prev_bottom:.1f}, Available height={gap:.1f}")
