"""Bundle the submission for one sweep into submission/<sweep id>/.

- packet/: claim_packet.json, report.html, report.pdf and every frame and crop they reference (zipped as well)
- results.md and results.pdf: the ground-truth sheet next to the system's numbers and every pass bar
- architecture.pdf and failure_log.pdf: the two one-page notes from docs/

Run: uv run python scripts/package_submission.py --sweep <id> [--truth ground_truth/mine]
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from typing import Any

from claimagent.capture.store import SweepStore
from claimagent.config import get_settings
from claimagent.eval.evaluate import evaluate, render_markdown

ROOT = Path(__file__).resolve().parents[1]
CSS = """
body { font: 10pt/1.38 -apple-system, Helvetica, Arial, sans-serif; color: #111; margin: 0; }
h1 { font-size: 15pt; margin: 0 0 5pt; } h2 { font-size: 12pt; margin: 11pt 0 3pt; }
h3 { font-size: 10.5pt; margin: 9pt 0 3pt; } p, li { margin: 2.5pt 0; } ul { margin: 2pt 0 2pt 15pt; padding: 0; }
code { font: 8.5pt ui-monospace, Menlo, monospace; background: #f2f2f2; padding: 0 2px; }
pre { font: 7.6pt/1.22 ui-monospace, Menlo, monospace; background: #f6f6f6; padding: 5pt; white-space: pre; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 3pt 0; }
th, td { border: 1px solid #ccc; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { background: #f0f0f0; } a { color: #0645ad; text-decoration: none; word-break: break-all; }
"""


# ----------------------------------------------------------------------------- Markdown subset -> HTML
def _inline(text: str) -> str:
    out = []
    for part in re.split(r"(`[^`]*`)", text):
        if len(part) >= 2 and part[0] == part[-1] == "`":
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        s = html.escape(part, quote=False)
        s = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![*\w])\*([^*\s][^*]*)\*(?![*\w])", r"<em>\1</em>", s)
        out.append(s)
    return "".join(out)


def _cells(line: str) -> list[str]:
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def md_to_html(md: str) -> str:
    """The Markdown our docs use: headings, paragraphs, bullet lists, pipe tables, code fences, bold, code, links."""
    lines = md.splitlines()
    out: list[str] = []
    para: list[str] = []
    i = 0
    bullet = re.compile(r"^\s*[-*]\s+")

    def flush() -> None:
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre>" + html.escape("\n".join(lines[i + 1 : j])) + "</pre>")
            i = j + 1
        elif m := re.match(r"^(#{1,3})\s+(.*)$", line):
            flush()
            n = len(m.group(1))
            out.append(f"<h{n}>{_inline(m.group(2))}</h{n}>")
            i += 1
        elif line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+$", lines[i + 1].strip()):
            flush()
            head, rows, i = _cells(line), [], i + 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(_cells(lines[i]))
                i += 1
            out.append(
                "<table><thead><tr>"
                + "".join(f"<th>{_inline(c)}</th>" for c in head)
                + "</tr></thead><tbody>"
                + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows)
                + "</tbody></table>"
            )
        elif bullet.match(line):
            flush()
            items: list[str] = []
            while i < len(lines) and (bullet.match(lines[i]) or (items and lines[i].startswith("  "))):
                if bullet.match(lines[i]):
                    items.append(bullet.sub("", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append("<ul>" + "".join(f"<li>{_inline(t)}</li>" for t in items) + "</ul>")
        elif not line.strip():
            flush()
            i += 1
        else:
            para.append(line.strip())
            i += 1
    flush()
    return "\n".join(out)


def print_pdf(md: str, title: str, pdf: Path) -> int:
    """Portrait A4 PDF of a Markdown document; returns its page count."""
    from playwright.sync_api import sync_playwright

    page_html = f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style>"
    page_html += md_to_html(md)
    tmp = pdf.with_suffix(".html")
    tmp.write_text(page_html, encoding="utf-8")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(tmp.resolve().as_uri(), wait_until="load")
            page.pdf(
                path=str(pdf),
                format="A4",
                print_background=True,
                margin={"top": "13mm", "bottom": "13mm", "left": "14mm", "right": "14mm"},
            )
        finally:
            browser.close()
    tmp.unlink()
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes()))


# ----------------------------------------------------------------------------- packet with its evidence
def _refs(o: Any, found: set[str]) -> None:
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("frame_ref", "frame_refs") and v:
                found.update(v if isinstance(v, list) else [v])
            else:
                _refs(v, found)
    elif isinstance(o, list):
        for x in o:
            _refs(x, found)


def bundle_packet(sweep_dir: Path, out: Path) -> Path:
    dest = out / "packet"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    refs: set[str] = set()
    _refs(json.loads((sweep_dir / "claim_packet.json").read_text()), refs)
    report = (sweep_dir / "report.html").read_text(encoding="utf-8")
    refs.update(r for r in re.findall(r'(?:src|href)="([^"#:?]+)"', report) if not r.startswith("/"))
    for name in ("claim_packet.json", "report.html", "report.pdf", *sorted(refs)):
        src = sweep_dir / name
        if src.is_file():
            (dest / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / name)
        else:
            print(f"  missing evidence file: {name}")
    return Path(shutil.make_archive(str(out / f"claim_packet_{sweep_dir.name}"), "zip", dest))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--truth", default="ground_truth/mine")
    ap.add_argument("--out", default="submission")
    args = ap.parse_args()
    store = SweepStore(get_settings().sweeps_dir, args.sweep)
    out = ROOT / args.out / args.sweep
    out.mkdir(parents=True, exist_ok=True)

    res = evaluate(store, ROOT / args.truth)
    results_md = render_markdown(res)
    (store.dir / "eval.md").write_text(results_md, encoding="utf-8")
    (out / "results.md").write_text(results_md, encoding="utf-8")
    zip_path = bundle_packet(store.dir, out)

    docs = {
        "results": (results_md, "Ground truth and results"),
        "architecture": ((ROOT / "docs/architecture.md").read_text(encoding="utf-8"), "Architecture"),
        "failure_log": ((ROOT / "docs/failure_log.md").read_text(encoding="utf-8"), "Failure log"),
    }
    for name, (md, title) in docs.items():
        pages = print_pdf(md, title, out / f"{name}.pdf")
        note = "" if name == "results" or pages == 1 else f"  <- {pages} pages; the brief asks for one"
        print(f"{name}.pdf: {pages} page(s){note}")
    if "TODO(final run)" in docs["failure_log"][0]:
        print("WARNING: docs/failure_log.md still has TODO(final run) placeholders")
    print(f"packet: {zip_path.relative_to(ROOT)}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    print(f"bundle: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
