"""Price sources. Each returns Offers with the URL and retrieval time they came from. No model involved."""

from __future__ import annotations

import base64
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from claimagent.pipeline.http import CachedHttp

log = logging.getLogger(__name__)

CURRENCY_SYMBOLS = {
    "₹": "INR",
    "$": "USD",
    "US$": "USD",
    "£": "GBP",
    "€": "EUR",
    "Rs": "INR",
    "Rs.": "INR",
    "C$": "CAD",
    "CA$": "CAD",
    "A$": "AUD",
    "AU$": "AUD",
    "NZ$": "NZD",
    "HK$": "HKD",
    "S$": "SGD",
    "SG$": "SGD",
    "MX$": "MXN",
}
GOOGLE_DOMAIN = {
    "IN": "google.co.in",
    "US": "google.com",
    "GB": "google.co.uk",
    "DE": "google.de",
    "FR": "google.fr",
    "CA": "google.ca",
    "AU": "google.com.au",
}
EBAY_MARKET = {
    "US": "EBAY_US",
    "GB": "EBAY_GB",
    "DE": "EBAY_DE",
    "FR": "EBAY_FR",
    "CA": "EBAY_CA",
    "AU": "EBAY_AU",
    "IT": "EBAY_IT",
    "ES": "EBAY_ES",
}
COUNTRY_CURRENCY = {
    "IN": "INR",
    "US": "USD",
    "GB": "GBP",
    "DE": "EUR",
    "FR": "EUR",
    "CA": "CAD",
    "AU": "AUD",
    "IT": "EUR",
    "ES": "EUR",
}


@dataclass
class Offer:
    title: str
    amount: float
    currency: str
    url: str
    seller: str
    condition: str  # new | used | unknown
    source: str
    retrieved_at: str
    query_url: str
    raw: dict[str, Any] = field(default_factory=dict)


def currency_of(price_text: str) -> str:
    text = price_text.strip()
    for sym in sorted(CURRENCY_SYMBOLS, key=len, reverse=True):
        if text.startswith(sym) or text.endswith(sym):
            return CURRENCY_SYMBOLS[sym]
    m = re.search(r"\b([A-Z]{3})\b", text)
    if m:
        return m.group(1)
    # A dollar sign with an unknown country prefix is not the searched currency: never guess it.
    return "?" if "$" in text else ""


USED_FILTER_TBS = "mr:1,new:3"  # Google Shopping "condition: used" filter
# Google ignores the used filter when it has no used listings (seen for India: the same new amazon.in offers
# come back), so an offer filtered as used counts only when the seller is a second-hand book marketplace.
USED_BOOK_SELLERS = re.compile(
    r"abebooks|biblio|ebay|bookbot|thriftbooks|better ?world|world ?of ?books|wob|alibris|half ?price|"
    r"powell|bookchor|mypustak|zoobooks|booksrun|etsy",
    re.I,
)


USED_WORDS = re.compile(r"\b(used|pre-?owned|second ?hand|refurbished)\b", re.I)


def amount_of(price_text: str) -> float | None:
    """Number in a displayed price such as '₹1,299.50' or 'US$12'. None when there is no number."""
    m = re.search(r"\d[\d,]*(?:\.\d+)?", price_text)
    if not m:
        return None
    try:
        value = float(m.group(0).replace(",", ""))
    except ValueError:
        return None
    return value if value > 0 else None


class SerpApiShopping:
    name = "Google Shopping (via SerpAPI)"

    def __init__(self, http: CachedHttp, key: str) -> None:
        self.http, self.key = http, key

    @property
    def enabled(self) -> bool:
        return bool(self.key) and "serpapi" not in self.http.disabled

    def search(self, query: str, country: str, used: bool = False) -> list[Offer]:
        """Shopping offers; used=True applies Google's "used" condition filter."""
        if not self.enabled:
            return []
        params: dict[str, Any] = {
            "engine": "google_shopping",
            "q": query,
            "gl": country.lower(),
            "hl": "en",
            "google_domain": GOOGLE_DOMAIN.get(country.upper(), "google.com"),
            "num": 40,
            "api_key": self.key,
        }
        if used:
            params["tbs"] = USED_FILTER_TBS
        try:
            r = self.http.get("https://serpapi.com/search.json", params, source="serpapi", paid=True)
        except (RuntimeError, LookupError) as exc:  # network failure or offline cache miss: no offers
            log.warning("serpapi search failed: %s", exc)
            return []
        if r.status >= 400:
            return []
        out: list[Offer] = []
        expected = COUNTRY_CURRENCY.get(country.upper(), "")
        for it in (r.data or {}).get("shopping_results", []) or []:
            amount = it.get("extracted_price")
            if not isinstance(amount, (int, float)) or amount <= 0:
                continue
            cur = currency_of(str(it.get("price", ""))) or expected
            second = str(it.get("second_hand_condition", "")).lower()
            title = str(it.get("title", ""))
            seller = str(it.get("source", ""))
            is_used = (
                bool(second) or bool(USED_WORDS.search(title)) or (used and bool(USED_BOOK_SELLERS.search(seller)))
            )
            out.append(
                Offer(
                    title=title,
                    amount=float(amount),
                    currency=cur,
                    url=str(it.get("product_link") or it.get("link") or r.url),
                    seller=str(it.get("source", "")),
                    condition="used" if is_used else "new",
                    source=self.name,
                    retrieved_at=r.retrieved_at,
                    query_url=r.url,
                    raw={k: it.get(k) for k in ("title", "price", "source", "second_hand_condition")},
                )
            )
        return out


class SerperShopping:
    """Google Shopping results via Serper.dev. Same contract as SerpApiShopping; the key goes in a header."""

    name = "Google Shopping (via Serper)"

    def __init__(self, http: CachedHttp, key: str) -> None:
        self.http, self.key = http, key

    @property
    def enabled(self) -> bool:
        return bool(self.key) and "serper" not in self.http.disabled

    def search(self, query: str, country: str, used: bool = False) -> list[Offer]:
        """Shopping offers; used=True applies Google's "used" condition filter."""
        if not self.enabled:
            return []
        params = {"q": query, "gl": country.lower(), "hl": "en"}
        if used:
            params["tbs"] = USED_FILTER_TBS
        try:
            r = self.http.get(
                "https://google.serper.dev/shopping",
                params,
                headers={"X-API-KEY": self.key},
                source="serper",
                paid=True,
            )
        except (RuntimeError, LookupError) as exc:  # network failure or offline cache miss: no offers
            log.warning("serper search failed: %s", exc)
            return []
        if r.status >= 400:
            return []
        out: list[Offer] = []
        expected = COUNTRY_CURRENCY.get(country.upper(), "")
        for it in (r.data or {}).get("shopping", []) or []:
            price_text = str(it.get("price", ""))
            amount = amount_of(price_text)
            if amount is None:
                continue
            title = str(it.get("title", ""))
            out.append(
                Offer(
                    title=title,
                    amount=amount,
                    currency=currency_of(price_text) or expected,
                    url=str(it.get("link") or r.url),
                    seller=str(it.get("source", "")),
                    condition="used"
                    if USED_WORDS.search(title) or (used and USED_BOOK_SELLERS.search(str(it.get("source", ""))))
                    else "new",
                    source=self.name,
                    retrieved_at=r.retrieved_at,
                    query_url=r.url,
                    raw={k: it.get(k) for k in ("title", "price", "source", "productId")},
                )
            )
        return out


class EbayBrowse:
    name = "eBay Browse API"

    def __init__(self, http: CachedHttp, client_id: str, client_secret: str) -> None:
        self.http, self.client_id, self.client_secret = http, client_id, client_secret
        self._token: tuple[str, float] | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.client_id and self.client_secret) and "ebay" not in self.http.disabled

    def _bearer(self) -> str:
        if self._token and self._token[1] > time.time() + 60:
            return self._token[0]
        basic = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
        r = httpx.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            timeout=20,
            headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "client_credentials", "scope": "https://api.ebay.com/oauth/api_scope"},
        )
        r.raise_for_status()
        data = r.json()
        self._token = (data["access_token"], time.time() + float(data.get("expires_in", 7200)))
        return self._token[0]

    def search_used(self, query: str, country: str) -> tuple[list[Offer], str]:
        """Used listings on the country's marketplace, or the US marketplace when the country has none."""
        if not self.enabled:
            return [], ""
        market = EBAY_MARKET.get(country.upper(), "EBAY_US")
        try:
            token = self._bearer()
        except httpx.HTTPError as exc:
            self.http.disabled["ebay"] = f"token request failed: {exc}"
            return [], market
        params = {"q": query, "limit": 50, "filter": "conditionIds:{3000|4000|5000|6000},buyingOptions:{FIXED_PRICE}"}
        try:
            r = self.http.get(
                "https://api.ebay.com/buy/browse/v1/item_summary/search",
                params,
                headers={"Authorization": f"Bearer {token}", "X-EBAY-C-MARKETPLACE-ID": market},
                source="ebay",
                vary=market,
            )
        except (RuntimeError, LookupError) as exc:  # network failure or offline cache miss: no offers
            log.warning("ebay search failed: %s", exc)
            return [], market
        if r.status >= 400:
            return [], market
        out: list[Offer] = []
        for it in (r.data or {}).get("itemSummaries", []) or []:
            p = it.get("price") or {}
            try:
                amount = float(p.get("value", 0))
            except (TypeError, ValueError):
                continue
            if amount <= 0:
                continue
            out.append(
                Offer(
                    title=str(it.get("title", "")),
                    amount=amount,
                    currency=str(p.get("currency", "")),
                    url=str(it.get("itemWebUrl", r.url)),
                    seller="eBay",
                    condition="used",
                    source=f"{self.name} ({market})",
                    retrieved_at=r.retrieved_at,
                    query_url=r.url,
                    raw={"condition": it.get("condition"), "title": it.get("title")},
                )
            )
        return out, market


class Frankfurter:
    """European Central Bank reference rates via frankfurter.app (no key)."""

    name = "ECB reference rate (frankfurter.app)"

    def __init__(self, http: CachedHttp) -> None:
        self.http = http

    def rate(self, base: str, quote: str) -> tuple[float, str, str] | None:
        if base == quote:
            return 1.0, "", ""
        try:
            r = self.http.get("https://api.frankfurter.app/latest", {"from": base, "to": quote}, source="fx")
        except (RuntimeError, LookupError):
            return None
        rate = (r.data or {}).get("rates", {}).get(quote)
        if not rate:
            return None
        return float(rate), str(r.data.get("date", "")), r.url


class GoogleBooksSale:
    """Google Play Books list price in a country. Ebook price: used only as a labelled last resort."""

    name = "Google Play Books (ebook list price)"

    def __init__(self, http: CachedHttp, key: str) -> None:
        self.http, self.key = http, key

    def price(self, volume_id: str, country: str) -> Offer | None:
        if not volume_id or "googlebooks" in self.http.disabled:
            return None
        params: dict[str, Any] = {"country": country.upper()}
        if self.key:
            params["key"] = self.key
        try:
            r = self.http.get(f"https://www.googleapis.com/books/v1/volumes/{volume_id}", params, source="googlebooks")
        except (RuntimeError, LookupError):
            return None
        sale = (r.data or {}).get("saleInfo", {})
        lp = sale.get("retailPrice") or sale.get("listPrice")
        if not lp or not lp.get("amount"):
            return None
        return Offer(
            title=(r.data.get("volumeInfo") or {}).get("title", ""),
            amount=float(lp["amount"]),
            currency=str(lp.get("currencyCode", "")),
            url=str(sale.get("buyLink") or r.url),
            seller="Google Play",
            condition="new (ebook)",
            source=self.name,
            retrieved_at=r.retrieved_at,
            query_url=r.url,
        )
