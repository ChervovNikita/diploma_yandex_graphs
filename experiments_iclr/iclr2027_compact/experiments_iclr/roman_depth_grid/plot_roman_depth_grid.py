"""Draw the audited, post hoc Roman depth grid without inferential error bars."""
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
audit = json.loads((ROOT / "ROMAN_DEPTH_GRID_40_CELL_AUDIT.json").read_text())
assert audit["status"] == "COMPLETE_40_CELL_READ_ONLY_AUDIT_PASS"
summary = audit["depth_summary"]
depths = [row["depth"] for row in summary]
values = [[100 * v for v in row["paired_differences"]] for row in summary]
means = [100 * row["mean_difference"] for row in summary]
S = 2
im = Image.new("RGB", (1800*S, 1000*S), "white")
d = ImageDraw.Draw(im)
font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 29*S)
small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 24*S)
bold = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 31*S)
left, top, right, bottom = 190*S, 100*S, 1680*S, 790*S
ymin, ymax = -1.8, 1.3

def xpix(x):
    return left + (x-2)/3*(right-left)

def ypix(y):
    return bottom - (y-ymin)/(ymax-ymin)*(bottom-top)

for y in (-1.5, -1.0, -0.5, 0, 0.5, 1.0):
    py = ypix(y)
    d.line((left, py, right, py), fill="#dadfe5" if y else "#343b46", width=2*S)
    label = f"{y:+.1f}" if y else "0"
    d.text((left-25*S, py), label, fill="#374151", font=small, anchor="rm")
for x in depths:
    px = xpix(x)
    d.line((px, bottom, px, bottom+10*S), fill="#374151", width=2*S)
    d.text((px, bottom+21*S), str(x), fill="#374151", font=font, anchor="mt")
for seed in range(5):
    pts = [(xpix(depths[i]), ypix(values[i][seed])) for i in range(4)]
    d.line(pts, fill="#aeb7c3", width=3*S, joint="curve")
    for x, y in pts:
        d.ellipse((x-6*S, y-6*S, x+6*S, y+6*S), fill="#8d98a7")
pts = [(xpix(depths[i]), ypix(means[i])) for i in range(4)]
d.line(pts, fill="#163b65", width=8*S, joint="curve")
for i, (x, y) in enumerate(pts):
    d.ellipse((x-10*S, y-10*S, x+10*S, y+10*S), fill="#163b65")
    offset = -24*S if means[i] >= 0 else 25*S
    d.text((x, y+offset), f"{means[i]:+.2f}", fill="#163b65", font=bold,
           anchor="mb" if means[i] >= 0 else "mt")
d.text(((left+right)/2, 880*S), "Number of residual SAGE blocks", fill="#111827", font=font, anchor="mm")
d.text(((left+right)/2, 955*S), "Roman Empire, mask 0 · five paired seeds · 300-epoch cap", fill="#4b5563", font=small, anchor="mm")
d.text((left, 32*S), "Tied minus untied test accuracy (percentage points)", fill="#111827", font=bold)
im.save(ROOT.parent / "overleaf" / "roman_depth_grid.png", optimize=True)
