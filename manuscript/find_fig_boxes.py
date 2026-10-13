import fitz

doc = fitz.open('d:/ares/Ares__Cognitive_computation.pdf')

def find_fig_boxes(fig_num_str):
    results = []
    for pno in range(len(doc)):
        p = doc[pno]
        rects = p.search_for(fig_num_str)
        if rects:
            for r in rects:
                results.append((pno + 1, r))
    return results

figures_to_check = [
    'Fig. 3:', 'Fig. 6:', 'Fig. 10:', 'Fig. 11:', 'Fig. 12:', 'Fig. 14:',
    'Fig. 15:', 'Fig. 16:', 'Fig. 19:', 'Fig. 22:', 'Fig. 28:', 'Fig. 29:',
    'Fig. 38:', 'Fig. 39:', 'Fig. 40:', 'Fig. 41:', 'Fig. 42:'
]

for f in figures_to_check:
    res = find_fig_boxes(f)
    print(f'{f} found at: {res}')
