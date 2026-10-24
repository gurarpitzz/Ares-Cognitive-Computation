import fitz
import json

doc = fitz.open(r"d:\ares\Ares__Cognitive_computation.pdf")
p = doc[0]
d = json.loads(p.get_text("json"))

for b_idx in range(min(5, len(d["blocks"]))):
    b = d["blocks"][b_idx]
    if "lines" in b:
        bbox = [round(x, 1) for x in b["bbox"]]
        print(f"=== Block {b_idx} (bbox: {bbox}) ===")
        for line in b["lines"]:
            line_str = "".join([s["text"] for s in line["spans"]])
            font = line["spans"][0]["font"]
            size = line["spans"][0]["size"]
            y_top = round(line["bbox"][1], 1)
            y_bot = round(line["bbox"][3], 1)
            x_left = round(line["bbox"][0], 1)
            x_right = round(line["bbox"][2], 1)
            print(f"  Line (font={font}, size={size:.2f}, x=[{x_left}, {x_right}], y=[{y_top}, {y_bot}]): {repr(line_str)}")
