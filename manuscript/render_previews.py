import fitz

doc = fitz.open('d:/ares/ARES_Cognitive_Computation_Revised.pdf')

pages_to_render = [1, 4, 8, 11, 16, 20, 28, 33, 35, 36]
for pno in pages_to_render:
    page = doc[pno - 1]
    pix = page.get_pixmap(dpi=150)
    out_img = f'd:/ares/manuscript/preview_page_{pno:02d}.png'
    pix.save(out_img)
    print(f'Rendered Page {pno:02d} to {out_img} ({pix.width}x{pix.height})')
