from claimagent.pipeline.price.engine import PriceEngine, book_offer_matches, robust_median
from claimagent.pipeline.price.sources import Offer, currency_of


def offer(title, amount, cur="INR", cond="new", src="Google Shopping (via SerpAPI)"):
    return Offer(
        title=title,
        amount=amount,
        currency=cur,
        url=f"https://shop/{amount}",
        seller="s",
        condition=cond,
        source=src,
        retrieved_at="2026-11-19T00:00:00+00:00",
        query_url="https://serpapi/q",
    )


class FakeSerp:
    enabled = True

    def __init__(self, by_country, used=None):
        self.by_country, self.used = by_country, used or {}

    def search(self, q, country, used=False):
        return list((self.used if used else self.by_country).get(country, []))


class FakeEbay:
    def __init__(self, offers=()):
        self.offers = list(offers)

    def search_used(self, q, country):
        return list(self.offers), "EBAY_US"


class FakeFx:
    def rate(self, base, quote):
        return (83.0, "2026-11-18", "https://fx") if (base, quote) == ("USD", "INR") else (1.0, "", "")


class FakeGBooks:
    def price(self, vid, country):
        return None


def engine(serp, ebay=(), used=None):
    return PriceEngine(FakeSerp(serp, used), FakeEbay(ebay), FakeFx(), FakeGBooks())


def test_currency_parsing():
    assert currency_of("₹499.00") == "INR" and currency_of("$12.99") == "USD" and currency_of("12,00 €") == "EUR"


def test_offer_title_filter_rejects_noise():
    assert book_offer_matches(offer("Shame and Necessity (Paperback)", 1), "Shame and Necessity", "Bernard Williams")
    assert not book_offer_matches(offer("Shame and Necessity Kindle Edition", 1), "Shame and Necessity", "")
    assert not book_offer_matches(offer("Necessity of Shame", 1), "Shame and Necessity in Ancient Greece", "")


def test_robust_median_drops_multipacks():
    med, kept = robust_median([400, 420, 450, 3900])
    assert med == 420 and 3900 not in kept


def test_local_new_and_converted_used_price():
    e = engine(
        {"IN": [offer("Gödel, Escher, Bach paperback", 1499), offer("Gödel Escher Bach", 1550)]},
        ebay=[
            offer("Godel Escher Bach used", 12.0, "USD", "used", "eBay Browse API (EBAY_US)"),
            offer("Gödel, Escher, Bach - good", 14.0, "USD", "used", "eBay Browse API (EBAY_US)"),
        ],
    )
    new, used = e.book("Gödel, Escher, Bach", "Douglas R. Hofstadter", "", "", "IN")
    assert new.amount == 1524.5 and not new.converted and new.url.startswith("https://shop/")
    assert used.converted and used.converted_from == "USD" and used.amount == round(13.0 * 83, 2)
    assert used.note.startswith("no local eBay marketplace")


def test_no_local_offer_converts_from_us_and_labels_it():
    e = engine({"US": [offer("Shame and Necessity", 30.0, "USD")]})
    new, used = e.book("Shame and Necessity", "Bernard Williams", "", "", "IN")
    assert new.converted and new.amount == 2490.0 and "US price converted" in new.note
    assert used.amount is None and used.note


def test_nothing_found_is_none_not_zero():
    new, used = engine({}).book("Obscure Title", "Nobody", "", "", "IN")
    assert new.amount is None and used.amount is None and new.note


def test_item_range_and_brand():
    offers = [offer(f"Floor lamp {i}", p) for i, p in enumerate([1200, 1500, 1800, 2400, 3000, 9000])]
    status, fig = engine({"IN": offers}).item("floor lamp", "", "IN")
    assert status == "range" and fig.low < fig.high and fig.offers_used == 5
    status, fig = engine({"IN": [offer("Philips Hue Go lamp", 7999)]}).item("table lamp", "Philips Hue", "IN")
    assert status == "priced" and fig.low == fig.high == 7999


def test_used_price_from_google_shopping_used_filter_before_ebay():
    used = {"IN": [offer("Siddhartha Hermann Hesse", 150, cond="used"), offer("Siddhartha (used)", 170, cond="used")]}
    ebay = [offer("Siddhartha", 5, cur="USD", cond="used"), offer("Siddhartha", 6, cur="USD", cond="used")]
    new, usedfig = engine({"IN": [offer("Siddhartha Paperback", 199)]}, ebay, used).book(
        "Siddhartha", "Hermann Hesse", "", "", "IN"
    )
    assert new.amount == 199 and usedfig.amount == 160 and usedfig.currency == "INR" and not usedfig.converted


def test_single_used_offer_falls_back_to_ebay():
    used = {"IN": [offer("Siddhartha Hermann Hesse", 150, cond="used")]}
    ebay = [offer("Siddhartha", 5, cur="USD", cond="used"), offer("Siddhartha", 7, cur="USD", cond="used")]
    _, usedfig = engine({"IN": []}, ebay, used).book("Siddhartha", "Hermann Hesse", "", "", "IN")
    assert usedfig.converted and usedfig.amount == 498.0


def test_us_second_hand_offers_converted_when_india_has_too_few():
    used = {
        "US": [offer("Siddhartha Hesse", 4, cur="USD", cond="used"), offer("Siddhartha", 6, cur="USD", cond="used")]
    }
    _, usedfig = engine({"IN": [offer("Siddhartha Paperback", 599)]}, (), used).book(
        "Siddhartha", "Hermann Hesse", "", "", "IN"
    )
    assert usedfig.converted and usedfig.amount == 415.0 and "US second-hand" in usedfig.note


def test_serper_used_filter_counts_only_second_hand_marketplaces(tmp_path, monkeypatch):
    from claimagent.pipeline.http import CachedHttp
    from claimagent.pipeline.price.sources import SerperShopping

    body = {
        "shopping": [
            {"title": "Siddhartha", "source": "amazon.in", "link": "https://g/1", "price": "₹135"},
            {"title": "Siddhartha", "source": "Bookchor", "link": "https://g/2", "price": "₹90"},
            {"title": "Siddhartha", "source": "AbeBooks.com", "link": "https://g/3", "price": "₹300"},
        ]
    }

    class R:
        status_code = 200
        text = "{}"

        def json(self):
            return body

    http = CachedHttp(tmp_path)
    monkeypatch.setattr(http.client, "request", lambda *a, **k: R())
    conds = [o.condition for o in SerperShopping(http, "K").search("Siddhartha", "IN", used=True)]
    assert conds == ["new", "used", "used"]


def test_used_offers_above_new_price_are_not_a_used_value():
    used = {
        "IN": [
            offer("Gödel Escher Bach", 900, cond="used"),
            offer("Gödel Escher Bach", 1100, cond="used"),
            offer("Gödel Escher Bach first edition", 33000, cond="used"),
        ]
    }
    new, usedfig = engine({"IN": [offer("Gödel, Escher, Bach paperback", 1500)]}, (), used).book(
        "Gödel, Escher, Bach", "Douglas R. Hofstadter", "", "", "IN"
    )
    assert new.amount == 1500 and usedfig.amount == 1000 and usedfig.offers_used == 2


def test_sets_translations_and_collector_copies_do_not_set_the_price():
    from claimagent.pipeline.price.engine import main_cluster

    noise = [
        "Harbour Series Collection 7 Books Set By Ann Writer",
        "The Silent Harbour - Hindi",
        "The Silent harbour (দ্য সাইলেন্ট হারবার)",
        "The silent harbour - signiert - signed Ann Writer",
        "The Silent Harbour LARGE PRINT by Ann Writer",
    ]
    assert not any(book_offer_matches(offer(t, 1), "The Silent Harbour", "Ann Writer") for t in noise)
    assert book_offer_matches(offer("The Silent Harbour: (Harbour Book 3)", 1), "The Silent Harbour", "Ann Writer")
    prices = [100, 149, 200, 280, 300, 445, 539, 599, 1479, 2719, 7733, 14008, 18584]
    assert main_cluster(prices) == [280, 300, 445, 539, 599]


def test_listing_must_be_the_same_work():
    ok = lambda t, title, author="", isbn=False: book_offer_matches(offer(t, 300), title, author, isbn)  # noqa: E731
    assert ok("Wings of Fire An Autobiography", "Wings of Fire", "A P J Abdul Kalam")
    assert not ok("Foundation and Empire", "Foundation", "Isaac Asimov")
    assert ok("Brave New World (Paperback)", "Brave New World", "Aldous Huxley")
    assert ok("Step by Step", "Step by Step", "Jane Doe")  # "by" here is not the author
    assert ok("The Smuggler", "The Smuggler", "")  # "mug" must not match inside a word
    assert not ok("Summary of Dune", "Dune", "Frank Herbert", isbn=True)  # an ISBN query is still filtered


def test_unknown_dollar_is_not_local_currency():
    assert currency_of("CA$24.99") == "CAD"
    assert currency_of("R$ 20,00") == "?"
    assert currency_of("$12.00") == "USD"


def test_item_listings_must_be_the_product_itself():
    from claimagent.pipeline.price.engine import item_offer_matches

    ok = lambda t, q="pressure cooker": item_offer_matches(offer(t, 1500), q)  # noqa: E731
    assert ok("Prestige Popular Svachh Pressure Cooker")
    assert not ok("The Indus Valley 3L+5L Tri-ply Pressure Cooker Combo")
    assert not ok("Whistle for Pressure Cooker")
    assert not ok("Prestige Svachh Aluminium Deep Pressure Pan")
    assert ok("Princeware Super Deluxe Blue Plastic Bucket 16 L", "bucket")
    assert not ok("Kuber Industries Pack Of 2 Multi-purpose Small Bucket", "bucket")


def test_printed_text_is_a_brand_only_when_it_reads_as_words():
    from claimagent.pipeline.assemble import _reads_as_words

    assert _reads_as_words("CRACKER SHAPE CHOCO BITES")
    assert not _reads_as_words("12 8bun ong")
    assert not _reads_as_words("C perch serch youeг")
