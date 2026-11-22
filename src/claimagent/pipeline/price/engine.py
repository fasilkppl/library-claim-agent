"""Turn retrieved offers into the two book figures and the item range.

Rules (from the brief):
- every figure carries source, URL, retrieval time and the condition assumed;
- a local price is preferred; otherwise another market's price is converted and labelled converted;
- no price at all -> amount None, excluded from totals;
- art, signed, rare and above-threshold lines are never auto-priced (handled by the caller).
"""

from __future__ import annotations

import logging
import re
import statistics
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz

from claimagent.pipeline.price.sources import (
    COUNTRY_CURRENCY,
    USED_BOOK_SELLERS,
    EbayBrowse,
    Frankfurter,
    GoogleBooksSale,
    Offer,
    SerpApiShopping,
    SerperShopping,
)
from claimagent.pipeline.resolve import content_tokens, english_words

log = logging.getLogger(__name__)

FORMAT_WORDS = {
    "paperback",
    "hardcover",
    "hardback",
    "softcover",
    "edition",
    "reprint",
    "english",
    "german",
    "hindi",
    "international",
    "indian",
    "binding",
    "unabridged",
    "illustrated",
}
NOISE = (
    "kindle",
    "ebook",
    "e-book",
    "audiobook",
    "audio cd",
    "summary of",
    "study guide",
    "poster",
    "bookmark",
    "t-shirt",
    "mug",
    "sticker",
    "notebook",
    "box set",
    "set of",
    "bundle",
    "lot of",
    "combo",
    "collection",
    "books set",
    "book set",
    "series set",
    " books!",
    "large print",
    "signed",
    "signiert",
    "first edition",
    "1st edition",
    "abridged",
    "antiquarian",
    "rare book",
    "unauthorized",
    "deciphering",
    "secrets of",
    "analysis of",
    "companion",
    "audiobook",
    "audio cd",
    "cd-audio",
    "[cd",
    " cd ",
    "mp3",
    "brief read",
    "analysis",
    "teacher's guide",
    "lesson",
    "reviewed",
    "study of",
    "approach to",
    "critical",
    "notes by",
    "guide to",
    "summary",
    " + ",
)
# Replacement = a new copy from an established bookseller. General marketplaces also carry cheap
# unlicensed prints, so their offers count only when no established seller lists the book.
ESTABLISHED_SELLERS = (
    "amazon",
    "flipkart",
    "crossword",
    "bookswagon",
    "sapna",
    "atlantic",
    "odyssey",
    "kitabay",
    "barnes",
    "books-a-million",
    "booksamillion",
    "bookshop.org",
    "powell",
    "target",
    "walmart",
    "waterstones",
    "blackwell",
    "whsmith",
    "thalia",
    "hugendubel",
)


# Budget marketplaces run under an established name: not a bookseller's own listing.
NOT_ESTABLISHED = ("shopsy",)


def is_established(seller: str) -> bool:
    s = seller.lower()
    return any(x in s for x in ESTABLISHED_SELLERS) and not any(x in s for x in NOT_ESTABLISHED)


def established(offers: list[Offer]) -> tuple[list[Offer], bool]:
    """Offers from established booksellers when any exists, else all (flag False). One established listing
    beats a median of marketplace offers (seen: one Amazon listing at Rs 851 outvoted by import offers)."""
    est = [o for o in offers if is_established(o.seller)]
    return (est, True) if est else (offers, False)


# Translations sold under the original title name their language in the listing.
OTHER_LANGUAGES = (
    "hindi",
    "tamil",
    "telugu",
    "marathi",
    "bengali",
    "kannada",
    "malayalam",
    "gujarati",
    "urdu",
    "farsi",
    "persian",
    "arabic",
)  # European translations carry their own titles ("El Símbolo Perdido") and fail the title match anyway


@dataclass
class PriceFigure:
    amount: float | None = None
    low: float | None = None
    high: float | None = None
    currency: str = ""
    source: str = ""
    url: str = ""
    retrieved_at: str = ""
    converted: bool = False
    converted_from: str = ""
    fx_rate: float | None = None
    fx_date: str = ""
    condition_assumed: str = ""
    offers_used: int = 0
    query: str = ""
    note: str = ""
    offers: list[dict[str, Any]] = field(default_factory=list)


def main_title(title: str, author: str = "") -> str:
    """The title a shop lists a book under: without "by <author>" and without a subtitle. A catalogue record
    can read "Go Set a Watchman by Harper Lee" or "Mahatma Gandhi: The Story of My Experiments with Truth"
    (author first); a listing reads "Go Set a Watchman" or "The Story of My Experiments with Truth"."""
    surnames = {toks[-1] for a in author.split(",") if (toks := content_tokens(a))}
    t = title
    m = re.search(r"\s+by\s+(.*)$", title, flags=re.I)
    if m and surnames & set(content_tokens(m.group(1))):  # "... by Harper Lee", not "Step by Step"
        t = title[: m.start()]
    head, sep, rest = t.partition(":")
    if sep and rest.strip():
        head_toks = content_tokens(head)
        # "Author: Title" -> the title; "Title: subtitle" -> the title
        t = (
            rest
            if head_toks and set(head_toks) <= (surnames | {w for a in author.split(",") for w in content_tokens(a)})
            else head
        )
    return t.strip()


def query_author(author: str) -> str:
    """The author as shops list the name: the first author, and of a long full name only its last two words
    ("Avul Pakir Jainulabdeen Abdul Kalam" -> "Abdul Kalam"; listings read "A.P.J. Abdul Kalam")."""
    words = author.split(",")[0].split()
    return " ".join(words[-2:] if len(words) > 3 else words)


def _listing_parts(listing: str) -> list[str]:
    """A listing title cut at its separators: "ReWork: Change the Way You Work Forever (Paperback)" ->
    ["ReWork", "Change the Way You Work Forever", "Paperback"], each also without a trailing "by ..."."""
    parts = re.split(r"[:|()\[\]\u2013\u2014]| - ", listing)
    return [re.sub(r"\s+by\s+.*$", "", p, flags=re.I).strip() for p in parts if p.strip()]


MAX_EXTRA_WORDS = 1  # descriptive words in the matching listing part beyond the main title (see DESCRIPTOR_WORDS)


ITEM_NOISE = re.compile(
    r"\b(combo|set of|sets|pack of|packs?|bundle|kit|pcs|pieces|pair of|refill|spare|replacement|gasket|"
    r"whistle|accessor(y|ies)|cover|stand|holder|organi[sz]er|mini|toy|miniature|keychain|sticker|decal|"
    r"poster|print|charm|earrings?|pendant|necklace|handbag|bag|t-?shirt)\b|\+|\bx\s?\d\b|\(\s*\d+\s*pcs?\s*\)",
    re.I,
)


def item_offer_matches(offer: Offer, query: str) -> bool:
    """The listing sells the product itself: it names the product noun (the last word of the query: "cooker",
    "iron", "bottle"), not only after "for" (an accessory for it), and it is not a set, pack or accessory."""
    t = offer.title.lower()
    if re.search(r"[^\x00-\u024f\u2000-\u206f\u20b9]", offer.title):
        return False
    q = content_tokens(query)
    if not q:
        return False
    noun = q[-1]
    if ITEM_NOISE.search(t) and not ITEM_NOISE.search(query.lower()):
        return False
    words = content_tokens(t)
    hits = [i for i, w in enumerate(words) if w == noun or (len(noun) >= 4 and w.startswith(noun))]
    if not hits:
        return False
    for_at = [i for i, w in enumerate(re.findall(r"[a-z0-9]+", t)) if w == "for"]
    if for_at:
        # "Whistle for Pressure Cooker": the product noun only after "for" means an accessory
        plain = re.findall(r"[a-z0-9]+", t)
        noun_at = [i for i, w in enumerate(plain) if w == noun or (len(noun) >= 4 and w.startswith(noun))]
        if noun_at and min(noun_at) > min(for_at):
            return False
    return True


def _term(term: str) -> re.Pattern[str]:
    """A list term matched as whole words ("mug" must not hit "smuggler", "lot of" not "pilot of")."""
    t = term.strip()
    left = r"\b" if t[:1].isalnum() else ""
    right = r"\b" if t[-1:].isalnum() else ""
    return re.compile(left + re.escape(t) + right)


_NOISE_RE = [(n, _term(n)) for n in NOISE]
# Copy condition in a marketplace listing ("Godel Escher Bach used - very good"): says nothing about the work.
CONDITION_WORDS = {"used", "good", "very", "fair", "acceptable", "condition", "pre", "owned", "preowned", "like", "new"}
# Extra words a shop adds to a title without it being another work ("Wings of Fire An Autobiography").
DESCRIPTOR_WORDS = {
    "novel",
    "autobiography",
    "memoir",
    "memoirs",
    "biography",
    "classic",
    "classics",
    "edition",
    "revised",
    "updated",
    "anniversary",
    "complete",
    "original",
    "book",
    "story",
    "expanded",
}


@lru_cache(maxsize=1)
def _english() -> frozenset[str]:
    return english_words()


def _another_language(listing: str, title: str, author: str, parts: list[str], match: int) -> bool:
    """A translation sold under the original title: a word with accents the book's title does not have
    ("Rework : réussir autrement"), or a subtitle mostly in words that are not English ("Rework : Apni
    Karyashaili Ko Hamesha Ke Liye Badal Dein", the Hindi edition)."""
    accented = set(re.findall(r"[a-zß-ÿ]*[à-öø-ÿ][a-zà-ÿ]*", listing.lower()))
    if accented - set(re.findall(r"[a-zß-ÿ]+", title.lower())):
        return True
    words = _english()
    if not words:
        return False
    names = set(re.findall(r"[a-z]+", f"{title} {author}".lower()))  # names are not English words
    for j, part in enumerate(parts):
        toks = [w for w in re.findall(r"[a-z]{3,}", part.lower()) if w not in names]
        if j != match and len(toks) >= 3 and sum(w in words for w in toks) * 2 < len(toks):
            return True
    return False


def book_offer_matches(offer: Offer, title: str, author: str, isbn_query: bool = False) -> bool:
    """The listing is this book on its own: its main title equals the book's main title (one descriptive word
    such as "novel" extra at most), it is not a study, summary, audio edition, translation or a set of several
    books. For an ISBN search the shop already matched the edition, so extra title words are allowed, but the
    listing must still contain the whole main title and pass the same exclusions."""
    t = offer.title.lower()
    title_l = title.lower()
    # A term that is part of the book's own title ("The Companion") does not exclude its listings.
    if any(rx.search(t) and not rx.search(title_l) for _, rx in _NOISE_RE):
        return False
    if any(lang in t and lang not in title_l for lang in OTHER_LANGUAGES):
        return False
    if re.search(r"[^\x00-\u024f\u2000-\u206f\u20b9]", offer.title):
        return False  # a listing title in another script (Bengali, Persian...) is another language edition
    if len(re.findall(r"\bby\b", t)) >= 2:
        return False  # "A by X and B by Y": two books sold together
    if re.search(r"\(\d{4}-\d{2}-\d{2}\)", offer.title):
        return False  # "Title by Author(2003-05-27)": a reseller listing that names no edition (old or imported stock)
    author_toks = {w for a in author.split(",") for w in content_tokens(a)}
    # Initials as shops print them glued: "A.P.J. Abdul Kalam" -> "apj"
    author_toks |= {
        "".join(ini)
        for a in author.split(",")
        if len(ini := re.findall(r"\b([a-z])\b", a.lower().replace(".", " "))) >= 2
    }
    want = [w for w in content_tokens(main_title(title, author)) if w not in author_toks]
    if not want:
        return False
    parts = _listing_parts(offer.title)
    in_brackets = {p.strip().lower() for p in re.findall(r"[(\[]([^)\]]*)", offer.title)}
    outside = [w for p in parts if p.strip().lower() not in in_brackets for w in content_tokens(p)]
    for i, part in enumerate(parts):
        # A match only inside brackets, after another title, is a series name: "The Dragonet Prophecy (Wings
        # Of Fire #1)" is not Wings of Fire.
        if part.strip().lower() in in_brackets:
            others = [w for w in outside if w not in DESCRIPTOR_WORDS | FORMAT_WORDS | author_toks | set(want)]
            if len(others) >= 2:
                continue
        skip = (FORMAT_WORDS | CONDITION_WORDS | author_toks) - set(want)  # "Brave New World" keeps "new"
        toks = [w for w in content_tokens(part) if w not in skip]
        if not toks:
            continue
        hits = sum(1 for w in want if any(w == x or fuzz.ratio(w, x) >= 90 for x in toks))
        extra = [x for x in toks if not any(x == w or fuzz.ratio(w, x) >= 90 for w in want)]
        if hits < len(want):
            continue
        if isbn_query or (len(extra) <= MAX_EXTRA_WORDS and all(x in DESCRIPTOR_WORDS for x in extra)):
            if _another_language(offer.title, title, author, parts, i):
                return False
            # "Title: Subtitle & Another Title" or "Title: Subtitle and Another Title": two books sold together (an
            # "&" in the book's own title is fine).
            subtitle = title_l.partition(":")[2].strip()
            for j, q in enumerate(parts):
                rest = q.lower().strip()
                if j == i:
                    continue
                if subtitle and rest.startswith(subtitle):
                    rest = rest[len(subtitle) :].strip()
                    if re.match(r"(&|and\b|\+)\s*\w", rest):
                        return False
                if "&" in rest and "&" not in title_l:
                    return False
            return True
    return False


CLUSTER_SPAN = 2.5  # the densest price band spans at most this factor (e.g. 200-500)


def main_cluster(values: list[float]) -> list[float]:
    """The densest band of prices: ordinary retail copies. Collector, import and mispriced listings scatter
    far above it and must not move the replacement figure. Ties go to the cheaper band."""
    vs = sorted(values)
    best: list[float] = []
    j = 0
    for i, lo in enumerate(vs):
        j = max(j, i)
        while j + 1 < len(vs) and vs[j + 1] <= lo * CLUSTER_SPAN:
            j += 1
        if j - i + 1 > len(best):
            best = vs[i : j + 1]
    return best


def local_band(values: list[float]) -> list[float]:
    """The prices of the edition sold in this market. A jump of more than CLUSTER_SPAN between neighbouring
    prices separates editions for different markets: imported copies cost several times the local edition (seen:
    Rs 851 here, Rs 3,089-3,489 imported) and collector copies more. Within a continuous price range every
    listing counts (its median is the figure). The cheapest range holding at least a third of the listings is the
    local edition; when none does, all listings. Chosen over the densest-band rule on five listing snapshots
    taken over six hours (31 of 48 book prices within 25% of the reference sheet, against 26)."""
    vs = sorted(values)
    ranges: list[list[float]] = [[vs[0]]] if vs else []
    for v in vs[1:]:
        if v > ranges[-1][-1] * CLUSTER_SPAN:
            ranges.append([v])
        else:
            ranges[-1].append(v)
    return next((r for r in ranges if len(r) * 3 >= len(vs)), vs)


def robust_median(values: list[float]) -> tuple[float, list[float]]:
    """Median of the main price band (see main_cluster)."""
    if len(values) <= 2:
        return statistics.median(values), values
    kept = main_cluster(values)
    return statistics.median(kept), kept


def quantiles(values: list[float]) -> tuple[float, float]:
    vs = sorted(values)
    if len(vs) < 4:
        return vs[0], vs[-1]
    q = statistics.quantiles(vs, n=4)
    return q[0], q[2]


class PriceEngine:
    def __init__(
        self, serp: SerpApiShopping | SerperShopping, ebay: EbayBrowse, fx: Frankfurter, gbooks: GoogleBooksSale
    ) -> None:
        self.serp, self.ebay, self.fx, self.gbooks = serp, ebay, fx, gbooks

    # ---------------------------------------------------------------- helpers
    def _to_local(self, fig: PriceFigure, currency: str) -> PriceFigure:
        if not fig.currency or fig.currency == currency or (fig.amount is None and fig.low is None):
            return fig
        r = self.fx.rate(fig.currency, currency)
        if r is None:
            fig.note = (fig.note + f"; no FX rate {fig.currency}->{currency}, excluded").strip("; ")
            fig.amount = fig.low = fig.high = None
            return fig
        rate, date, _ = r
        fig.converted, fig.converted_from, fig.fx_rate, fig.fx_date = True, fig.currency, rate, date
        if fig.amount is not None:
            fig.amount = round(fig.amount * rate, 2)
        if fig.low is not None:
            fig.low = round(fig.low * rate, 2)
        if fig.high is not None:
            fig.high = round(fig.high * rate, 2)
        fig.currency = currency
        return fig

    @staticmethod
    def _figure(offers: list[Offer], condition: str, query: str, local: bool = False) -> PriceFigure:
        amounts = [o.amount for o in offers]
        # Two listings far apart (Rs 449 and an imported Rs 3,489) are two editions: their mean is no listing's price.
        if local and len(amounts) >= 2:
            kept = local_band(amounts)
            med = statistics.median(kept)
        else:
            med, kept = robust_median(amounts)
        rep = min(offers, key=lambda o: abs(o.amount - med))  # the cited offer is the one closest to the median
        lo, hi = quantiles(kept)
        return PriceFigure(
            amount=round(med, 2),
            low=round(lo, 2),
            high=round(hi, 2),
            currency=rep.currency,
            source=rep.source,
            url=rep.url,
            retrieved_at=rep.retrieved_at,
            condition_assumed=condition,
            offers_used=len(kept),
            query=query,
            offers=[
                {"title": o.title[:90], "amount": o.amount, "currency": o.currency, "seller": o.seller, "url": o.url}
                for o in offers[:8]
            ],
        )

    def _used_shopping(
        self, query: str, title: str, author: str, isbn: str, country: str, currency: str, cap: float | None
    ) -> list[Offer]:
        """Second-hand marketplace offers under the "used" filter, at or below the new price (collectible copies
        priced above a new copy are not a used value). Needs at least two."""
        offers = [
            o
            for o in self.serp.search(query, country, used=True)
            if o.condition == "used"
            and book_offer_matches(o, title, author, bool(isbn))
            and o.currency in (currency, "")
            and (cap is None or o.amount <= cap)
        ]
        return offers if len(offers) >= 2 else []

    def _cap_in(self, cap_local: float | None, local: str, other: str) -> float | None:
        if cap_local is None or local == other:
            return cap_local
        r = self.fx.rate(other, local)
        return cap_local / r[0] if r else None

    def _within_cap(self, o: Offer, cap_local: float | None, local: str) -> bool:
        """The offer is at or below the local cap, compared in the offer's own currency. Unknown rate: dropped."""
        if cap_local is None:
            return True
        c = self._cap_in(cap_local, local, o.currency or local)
        return c is not None and o.amount <= c

    # ---------------------------------------------------------------- books
    def book(self, title: str, author: str, isbn: str, volume_id: str, country: str) -> tuple[PriceFigure, PriceFigure]:
        # The catalogue match names the work, not the copy on the shelf: its ISBN is one edition among many (seen:
        # the US hardcover of Go Set a Watchman at Rs 1,818; the paperback sells for about Rs 600). Prices are
        # searched by title and author; the ISBN is not used.
        new_fig, shop = self._book_new(title, author, "", volume_id, country)
        used_fig = self._book_used(title, author, "", country, shop, new_fig)
        return new_fig, used_fig

    def _book_new(
        self, title: str, author: str, isbn: str, volume_id: str, country: str
    ) -> tuple[PriceFigure, list[Offer]]:
        currency = COUNTRY_CURRENCY.get(country.upper(), "")
        title_query = f"{main_title(title, author)} {query_author(author)} book".strip()
        # When no established bookseller shows up, one more search as a paperback: Google Shopping then lists
        # booksellers' print copies (seen: Rework, whose first search held only translations and study notes).
        queries = [title_query, f"{main_title(title, author)} {query_author(author)} paperback".strip()]
        new_fig = PriceFigure(currency=currency, query=queries[0])
        shop: list[Offer] = []
        fallback: PriceFigure | None = None
        for query in queries:
            found = [o for o in self.serp.search(query, country) if book_offer_matches(o, title, author)]
            found = [o for o in found if o.currency == currency or not o.currency]
            shop += found
            # Rare-book marketplaces (Biblio, AbeBooks...) list collector and second-hand copies: not a new copy.
            new_offers, from_established = established(
                [o for o in found if o.condition == "new" and not USED_BOOK_SELLERS.search(o.seller)]
            )
            if not new_offers:
                continue
            fig = self._figure(new_offers, "new", query, local=True)
            fig.currency = currency
            if from_established:
                new_fig = fig
                break
            if fallback is None:
                fig.note = "no established bookseller lists it; median of all new offers"
                fallback = fig
        else:
            if fallback is not None:
                new_fig = fallback
        # Other market: US Google Shopping, converted.
        if new_fig.amount is None and country.upper() != "US" and self.serp.enabled:
            for query in queries[:1]:
                us = [
                    o
                    for o in self.serp.search(query, "US")
                    if book_offer_matches(o, title, author) and o.condition == "new" and o.currency in ("USD", "")
                ]
                if us:
                    fig = self._figure(us, "new", query)
                    fig.currency = "USD"
                    new_fig = self._to_local(fig, currency)
                    new_fig.note = f"no local new offer in {country}; US price converted"
                    break
        if new_fig.amount is None:
            eb = self.gbooks.price(volume_id, country)
            if eb is not None:
                fig = self._figure([eb], "new (ebook edition, closest equivalent found)", new_fig.query)
                new_fig = self._to_local(fig, currency)
                new_fig.note = "only an ebook list price was found; review whether a print copy is needed"
        if new_fig.amount is None and not new_fig.note:
            new_fig.note = "no price found from any configured source"
        return new_fig, shop

    def _book_used(
        self, title: str, author: str, isbn: str, country: str, shop: list[Offer], new_fig: PriceFigure
    ) -> PriceFigure:
        currency = COUNTRY_CURRENCY.get(country.upper(), "")
        query = new_fig.query
        # Cap only against a print price; an ebook list price says nothing about a used print copy.
        cap = new_fig.amount if new_fig.condition_assumed == "new" else None
        used_fig = PriceFigure(currency=currency, query=query)
        local = [o for o in shop if o.condition == "used" and (cap is None or o.amount <= cap)]
        if len(local) < 2:
            local = self._used_shopping(query, title, author, isbn, country, currency, cap)
        if local:
            used_fig = self._figure(local, "used (median of second-hand offers on Google Shopping)", query)
            used_fig.currency = currency
            return used_fig
        if country.upper() != "US":
            us_used = self._used_shopping(query, title, author, isbn, "US", "USD", self._cap_in(cap, currency, "USD"))
            if us_used:
                used_fig = self._to_local(
                    self._figure(us_used, "used (median of second-hand marketplace offers, US)", query), currency
                )
                used_fig.note = f"too few second-hand offers in {country}; US second-hand offers converted"
                return used_fig
        ebay_offers, market = self.ebay.search_used(query, country)
        ebay_offers = [
            o
            for o in ebay_offers
            if book_offer_matches(o, title, author, bool(isbn)) and self._within_cap(o, cap, currency)
        ]
        if len(ebay_offers) >= 2:
            used_fig = self._to_local(
                self._figure(ebay_offers, "used, good (median of eBay fixed-price listings)", query), currency
            )
            if market and not market.endswith(country.upper()):
                used_fig.note = f"no local eBay marketplace for {country}; {market} listings converted"
            return used_fig
        used_fig.note = "no used price found from any configured source"
        return used_fig

    # ---------------------------------------------------------------- items
    def item(self, query: str, brand_model: str, country: str) -> tuple[str, PriceFigure]:
        """Returns (status, figure): 'priced' when brand/model offers match, else a sourced 'range'."""
        currency = COUNTRY_CURRENCY.get(country.upper(), "")
        # "orient clothes iron", never just "orient": a brand alone finds every product the brand makes.
        q = f"{brand_model} {query}".strip() if brand_model else query
        found = [
            o
            for o in self.serp.search(q, country)
            if o.condition == "new" and (o.currency == currency or not o.currency)
        ]
        # Only listings of this one product: not a combo or set, not an accessory "for" it, not another product
        # that shares a word ("Pressure Pan" for a pressure cooker, a "Bucket" handbag for a bucket). Then the
        # main price band: imports and premium one-offs far above it are not the usual replacement.
        offers = [o for o in found if item_offer_matches(o, query)]
        if len(offers) >= 3:
            band = main_cluster([o.amount for o in offers])
            offers = [o for o in offers if band and band[0] <= o.amount <= band[-1]]
        log.info("item %r: %s offers, %s kept as this product in the main price band", q, len(found), len(offers))
        if brand_model:
            toks = content_tokens(brand_model)
            matched = [o for o in offers if toks and all(t in content_tokens(o.title) for t in toks[:2])]
            if len(matched) >= 1:
                fig = self._figure(matched, "new", q)
                fig.currency = currency
                fig.low = fig.high = fig.amount
                return "priced", fig
        if len(offers) >= 3:
            fig = self._figure(offers, "new, comparable items (brand/model not legible)", q)
            fig.currency = currency
            fig.note = f"range = 25th-75th percentile of {fig.offers_used} comparable offers"
            return "range", fig
        return "unpriced", PriceFigure(
            currency=currency, query=q, note="not enough comparable offers from configured sources"
        )
