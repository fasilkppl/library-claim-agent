"""report.pdf: the HTML report printed by headless Chromium, so the PDF always matches the HTML."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    """Writes the PDF; False (and a log line) when Chromium is not available. The packet never depends on it."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.warning("playwright not installed: no PDF report")
        return False
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(html_path.resolve().as_uri(), wait_until="load")
                page.pdf(
                    path=str(pdf_path),
                    format="A4",
                    landscape=True,
                    print_background=True,
                    scale=0.8,
                    margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
                )
            finally:
                browser.close()
    except Exception as exc:  # any Chromium failure: the packet and HTML report stand without the PDF
        log.warning("PDF report failed (run `uv run playwright install chromium`): %s", exc)
        return False
    return True
