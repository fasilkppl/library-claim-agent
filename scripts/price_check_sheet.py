"""A blank hand-check sheet for the book prices of a sweep, to fill in right after the sweep.

Listing prices move by up to 2x within hours (one listing went from Rs 640 to Rs 291 in 45 minutes), so a price
checked hours before or after the sweep is a different measurement. Run this right after the final sweep:

    uv run python scripts/price_check_sheet.py <sweep_id>

It writes sweeps/<id>/price_check.csv with one row per identified book: title, author and the Google Shopping
India link for the same search the app runs. It shows no app price, so the check is not anchored on it. Open
each link, note the prices of new English single copies from established booksellers (Amazon, Flipkart,
Crossword, ...; no sets, translations, used or imported copies), write them in `counted_prices` and their middle
value in `replacement_price`, then copy the rows into ground_truth/mine/prices.csv.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(sweep_id: str) -> Path:
    from claimagent.config import get_settings
    from claimagent.pipeline.price.engine import main_title, query_author

    d = get_settings().sweeps_dir / sweep_id
    packet = json.loads((d / "claim_packet.json").read_text())
    out = d / "price_check.csv"
    cols = ["title", "author", "search_url", "counted_prices", "replacement_price", "checked_on"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for b in packet["books"]:
            if not b.get("title"):
                continue
            q = f"{main_title(b['title'], b.get('author', ''))} {query_author(b.get('author', ''))} book".strip()
            url = "https://www.google.com/search?tbm=shop&gl=in&q=" + urllib.parse.quote_plus(q)
            w.writerow(
                {
                    "title": b["title"],
                    "author": b.get("author", ""),
                    "search_url": url,
                    "counted_prices": "",
                    "replacement_price": "",
                    "checked_on": time.strftime("%Y-%m-%d"),
                }
            )
    print(f"wrote {out}: open each link, fill counted_prices and replacement_price")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
