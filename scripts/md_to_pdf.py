"""Markdown (+ mermaid) → PDF via headless Chrome.

  python scripts/md_to_pdf.py result_reporting/architecture_v4.md        → result_reporting/architecture_v4.pdf
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

CHROME = [r"C:\Program Files\Google\Chrome\Application\chrome.exe", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]

CSS = """
@page { size: A4; margin: 14mm 12mm; }
body { font-family: "Leelawadee UI", Tahoma, "Segoe UI", sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1d1d1f; }
h1 { font-size: 20pt; border-bottom: 3px solid #2b5cab; padding-bottom: 4px; }
h2 { font-size: 15pt; color: #2b5cab; border-bottom: 1px solid #ccd; padding-bottom: 2px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12pt; margin-top: 16px; page-break-after: avoid; }
code { font-family: Consolas, "Leelawadee UI", monospace; font-size: 9pt; background: #f2f3f7; padding: 0 3px; border-radius: 3px; }
pre { background: #f6f7fb; border: 1px solid #dde; border-radius: 6px; padding: 8px 10px; white-space: pre-wrap; word-break: break-word;
      font-size: 7.6pt; line-height: 1.35; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 8px 0; page-break-inside: avoid; }
th, td { border: 1px solid #ccd; padding: 4px 6px; vertical-align: top; }
th { background: #e9eef8; }
blockquote { border-left: 4px solid #e0a800; background: #fff8e1; margin: 8px 0; padding: 6px 10px; }
pre.mermaid { background: white; border: 1px solid #dde; text-align: center; white-space: pre; font-size: 10pt; }
pre.mermaid svg { max-width: 100% !important; height: auto; }
"""


def render(md_text: str, title: str) -> str:
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render(md_text)
    # ```mermaid fences → <pre class="mermaid"> (mermaid needs the raw text)
    body = re.sub(r'<pre><code class="language-mermaid">(.*?)</code></pre>',
                  lambda m: f'<pre class="mermaid">{m.group(1)}</pre>', body, flags=re.S)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>{CSS}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script></head>
<body>{body}
<script>
  mermaid.initialize({{startOnLoad: false, theme: "default", securityLevel: "loose",
                      themeVariables: {{fontFamily: '"Leelawadee UI", Tahoma, sans-serif'}}, flowchart: {{useMaxWidth: true, htmlLabels: true}}}});
  mermaid.run({{querySelector: "pre.mermaid"}}).then(() => document.title = "RENDERED");
</script></body></html>"""


def main(src):
    src = Path(src).resolve()
    out = src.with_suffix(".pdf")
    page = render(src.read_text(encoding="utf-8"), src.stem)
    with tempfile.TemporaryDirectory() as td:
        h = Path(td) / f"{src.stem}.html"
        h.write_text(page, encoding="utf-8")
        chrome = next(c for c in CHROME if Path(c).exists())
        subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000",
                        "--run-all-compositor-stages-before-draw", f"--print-to-pdf={out}", h.as_uri()], check=True, timeout=180)
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
