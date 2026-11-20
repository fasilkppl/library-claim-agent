"""CLI: python -m claimagent.pipeline run|reprice|eval ..."""

from __future__ import annotations

import argparse
import json
import logging

from claimagent.capture.store import SweepStore
from claimagent.config import get_settings


def main() -> None:
    p = argparse.ArgumentParser(prog="claimagent.pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the post-sweep pipeline on a saved sweep")
    r.add_argument("--sweep", required=True)
    r.add_argument("--offline", action="store_true", help="use cached HTTP responses only")
    r.add_argument("--fresh", action="store_true", help="recompute frame analysis instead of reusing it")
    rp = sub.add_parser("reprice", help="price the same books for a second country")
    rp.add_argument("--sweep", required=True)
    rp.add_argument("--country", default=None)
    rp.add_argument("-n", type=int, default=10)
    e = sub.add_parser("eval", help="compare a packet with ground truth")
    e.add_argument("--sweep", required=True)
    e.add_argument("--truth", required=True, help="folder with books.csv, items.csv, room.csv, prices.csv")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # its request lines carry API keys
    s = get_settings()
    store = SweepStore(s.sweeps_dir, args.sweep)
    if args.cmd == "run":
        from claimagent.pipeline.runner import run_pipeline

        if args.fresh:
            for f in (store.stages_dir / "detect").glob("*.json"):
                f.unlink()
        packet = run_pipeline(store, s, offline=args.offline)
        print(json.dumps(packet.totals.model_dump(), indent=2))
        print(f"packet: {store.dir / 'claim_packet.json'}\nreport: {store.dir / 'report.html'}")
    elif args.cmd == "reprice":
        from claimagent.pipeline.runner import reprice

        out = reprice(store, args.country or s.second_country, args.n, s)
        print(json.dumps(out, indent=2))
    elif args.cmd == "eval":
        from pathlib import Path

        from claimagent.eval.evaluate import evaluate, render_markdown

        res = evaluate(store, Path(args.truth))
        md = render_markdown(res)
        (store.dir / "eval.md").write_text(md)
        (store.dir / "eval.json").write_text(json.dumps(res, indent=2))
        print(md)


if __name__ == "__main__":
    main()
