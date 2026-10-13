import fitz

doc = fitz.open('d:/ares/Ares__Cognitive_computation.pdf')

figs_to_find = [
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

fig_boxes = {}
for pno, cap in figs_to_find:
    p = doc[pno]
    rects = p.search_for(cap)
    if rects:
        r_cap = rects[0]
        col_x0 = 311.6 if r_cap.x0 > 300 else 48.5
        col_x1 = 564.8 if r_cap.x0 > 300 else 302.0
        
        # Look at blocks above r_cap.y0 in same column
        blocks = [b for b in p.get_text('blocks') if (b[0] > 300 if col_x0 > 300 else b[2] < 310) and b[3] < r_cap.y0]
        keywords = ['TABLE', 'Section', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX', 'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX', 'XXXI', 'XXXII', 'XXXIII', 'A.', 'B.', 'C.', 'D.', 'E.', '1)', '2)', '3)']
        text_blocks = [b for b in blocks if len(b[4].strip()) > 30 or any(k in b[4] for k in keywords) or 'Fig.' in b[4]]
        
        top_y = max((b[3] for b in text_blocks), default=52.0)
        bot_y = r_cap.y0 - 4
        print(f"P{pno+1:02d} {cap:8s}: col=({col_x0:.1f}, {col_x1:.1f}), top_y={top_y:.1f}, bot_y={bot_y:.1f}, height={bot_y - top_y:.1f}")
        fig_boxes[cap] = (pno, col_x0, top_y, col_x1, bot_y)
