"""Build Hermes-SAE-Writeup.pdf from WRITEUP.md.

Usage: python3 make_pdf.py   (needs python-markdown + google-chrome)

Choices that matter for output quality:
- The architecture figure is embedded as SVG, so the whole PDF is vector
  (text stays crisp at any zoom; regenerate figures with
  figures/make_f2.py first if they changed).
- Repo-relative links are rewritten to absolute GitHub URLs so every
  link works for a reader who receives only the PDF.
"""
import re
import subprocess
from pathlib import Path

import markdown

HERE = Path(__file__).parent
OUT = HERE / "Hermes-SAE-Writeup.pdf"
BLOB = "https://github.com/SolshineCode/hermes-sae/blob/main/"
TREE = "https://github.com/SolshineCode/hermes-sae/tree/main/"

src = (HERE / "WRITEUP.md").read_text(encoding="utf-8")


def fix(m):
    path = m.group(2)
    if path.startswith("../"):
        clean = path[3:]
        base = TREE if clean.endswith("/") else BLOB
        return f"[{m.group(1)}]({base}{clean})"
    return m.group(0)


src = re.sub(r"\[([^\]]+)\]\((\.\./[^)]+)\)", fix, src)
src = src.replace("figures/f2_architecture.png", "figures/f2_architecture.svg")

body = markdown.markdown(src, extensions=["extra", "sane_lists"])

html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
@page {{ margin: 22mm 19mm; }}
body {{ font: 11pt/1.55 'Liberation Serif', Georgia, serif; color:#1a1c1e;
       max-width: 46em; margin: 0 auto; }}
h1 {{ font: 700 19pt/1.25 'Liberation Sans', Helvetica, sans-serif; margin: 0 0 4px; }}
h2 {{ font: 700 13.5pt/1.3 'Liberation Sans', Helvetica, sans-serif;
     margin: 22px 0 6px; border-bottom: 1px solid #d8d8d4; padding-bottom: 3px;
     break-after: avoid; }}
p {{ margin: 8px 0; text-align: justify; }}
em {{ color:#444; }}
a {{ color:#1a56a0; text-decoration: none; }}
code {{ font: 9.5pt 'DejaVu Sans Mono', monospace; background:#f2f2ee;
       padding: 0 3px; border-radius: 3px; }}
img {{ max-width: 100%; margin: 10px 0; }}
ol, ul {{ margin: 8px 0; padding-left: 1.4em; }}
li {{ margin: 4px 0; }}
strong {{ color:#111; }}
</style></head><body>{body}</body></html>"""

render = HERE / "_render.html"
render.write_text(html, encoding="utf-8")
try:
    subprocess.run(
        ["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox",
         "--no-pdf-header-footer", f"--print-to-pdf={OUT}",
         f"file://{render}"],
        check=True, capture_output=True, timeout=120)
finally:
    render.unlink(missing_ok=True)
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
