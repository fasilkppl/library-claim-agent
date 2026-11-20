import json

import pytest

from claimagent.capture.store import SweepStore
from claimagent.live.tools import run_tool
from claimagent.live.voice import sentences
from claimagent.pipeline.http import CachedHttp


def test_cache_strips_secrets_and_serves_offline(tmp_path, monkeypatch):
    http = CachedHttp(tmp_path)

    class R:
        status_code = 200
        text = "{}"

        def json(self):
            return {"ok": 1}

    monkeypatch.setattr(http.client, "request", lambda *a, **k: R())
    r = http.get("https://serpapi.com/search.json", {"q": "x", "api_key": "SECRET"}, source="serpapi", paid=True)
    assert "SECRET" not in r.url and http.paid_calls["serpapi"] == 1
    saved = next(tmp_path.glob("serpapi-*.json")).read_text()
    assert "SECRET" not in saved and json.loads(saved)["data"] == {"ok": 1}
    off = CachedHttp(tmp_path, offline=True)
    assert off.get("https://serpapi.com/search.json", {"q": "x", "api_key": "OTHER"}, source="serpapi").from_cache
    with pytest.raises(LookupError):
        off.get("https://serpapi.com/search.json", {"q": "y"}, source="serpapi")


def test_quota_error_disables_source(tmp_path, monkeypatch):
    http = CachedHttp(tmp_path)

    class R:
        status_code = 429
        text = "Quota exceeded for quota metric"

    monkeypatch.setattr(http.client, "request", lambda *a, **k: R())
    with pytest.raises(RuntimeError):
        http.get("https://www.googleapis.com/books/v1/volumes", {"q": "x"}, source="googlebooks")
    assert "googlebooks" in http.disabled


def test_tools_record_evidence(tmp_path):
    st = SweepStore(tmp_path, "s1")
    assert run_tool(st, "set_locale", {"country_code": "us", "currency": ""}, 10)["currency"] == "USD"
    run_tool(st, "skip_region", {"statement": "not mine"}, 20)
    run_tool(st, "note_book", {"statement": "signed first edition", "first_edition": True, "signed": True}, 30)
    assert not run_tool(st, "shelf_width", {"statement": "x", "width_cm": 5000}, 40)["ok"]
    kinds = [e["kind"] for e in st.events()]
    assert kinds == ["locale", "skip_region", "book_note"]
    assert st.read_session()["country"] == "US"


def test_sentences_stream_split():
    toks = iter(["Logged twelve", " books. Please slow", " down! Thanks"])
    assert list(sentences(toks)) == ["Logged twelve books.", "Please slow down!", "Thanks"]


def test_serper_offers_parsed_and_key_not_saved(tmp_path, monkeypatch):
    from claimagent.pipeline.price.sources import SerperShopping, amount_of

    assert amount_of("₹1,299.50") == 1299.5 and amount_of("US$12") == 12.0 and amount_of("free") is None
    http = CachedHttp(tmp_path)
    body = {
        "shopping": [
            {"title": "Siddhartha (Paperback)", "source": "amazon.in", "link": "https://g/1", "price": "₹135.50"},
            {"title": "Siddhartha used copy", "source": "Meesho", "link": "https://g/2", "price": "₹1,176"},
            {"title": "No price", "source": "x", "link": "https://g/3", "price": ""},
        ]
    }
    seen = {}

    class R:
        status_code = 200
        text = json.dumps(body)

        def json(self):
            return body

    def fake(method, url, params=None, headers=None, json=None, **kwargs):
        seen["headers"] = headers
        return R()

    monkeypatch.setattr(http.client, "request", fake)
    offers = SerperShopping(http, "SERPERKEY").search("Siddhartha Hesse", "IN")
    assert [(o.amount, o.currency, o.condition) for o in offers] == [(135.5, "INR", "new"), (1176.0, "INR", "used")]
    assert seen["headers"]["X-API-KEY"] == "SERPERKEY" and http.paid_calls["serper"] == 1
    assert "SERPERKEY" not in next(tmp_path.glob("serper-*.json")).read_text()


def test_price_search_network_failure_returns_no_offers(tmp_path, monkeypatch):
    import httpx

    from claimagent.pipeline.price.sources import SerperShopping

    http = CachedHttp(tmp_path)

    def boom(*a, **k):
        raise httpx.ReadTimeout("The read operation timed out")

    monkeypatch.setattr(http.client, "request", boom)
    monkeypatch.setattr("time.sleep", lambda s: None)
    assert SerperShopping(http, "K").search("anything", "IN") == []


def test_per_minute_quota_waits_instead_of_disabling(tmp_path, monkeypatch):
    http = CachedHttp(tmp_path)
    replies = iter(
        [(429, "Quota exceeded for quota metric 'Queries' and limit 'Queries per minute per user'"), (200, "{}")]
    )

    class R:
        def __init__(self, code, text):
            self.status_code, self.text = code, text

        def json(self):
            return {"ok": 1}

    monkeypatch.setattr(http.client, "request", lambda *a, **k: R(*next(replies)))
    monkeypatch.setattr("time.sleep", lambda s: None)
    r = http.get("https://www.googleapis.com/books/v1/volumes", {"q": "x"}, source="googlebooks")
    assert r.status == 200 and "googlebooks" not in http.disabled


def test_slow_catalogue_site_is_skipped_after_three_timeouts(tmp_path, monkeypatch):
    import httpx

    http = CachedHttp(tmp_path)

    def slow(*a, **k):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(http.client, "request", slow)
    monkeypatch.setattr("time.sleep", lambda s: None)
    for q in ("a", "b"):
        with pytest.raises(RuntimeError):
            http.get("https://openlibrary.org/search.json", {"q": q}, source="openlibrary")
    assert "openlibrary" in http.disabled  # 2 + 2 attempts: disabled at the third timeout in a row
