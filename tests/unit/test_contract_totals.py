from claimagent.contract import Book, BookPrice, Item, ItemPrice
from claimagent.pipeline.assemble import compute_totals


def book(i, status="identified", r=None, u=None):
    return Book(
        id=f"B{i}",
        shelf="s",
        position=i,
        frame_ref="frames/000001.jpg",
        status=status,
        replacement_cost=BookPrice(amount=r),
        used_value=BookPrice(amount=u),
    )


def test_totals_add_up_and_exclusions_counted():
    books = [
        book(1, r=500, u=200),
        book(2, r=300.5, u=100),
        book(3, "unidentified"),
        book(4, "needs_appraisal"),
        book(5, r=100, u=None),
    ]
    items = [
        Item(
            id="I1", category="seating", frame_ref="f", status="range", replacement_cost=ItemPrice(low=1000, high=3000)
        ),
        Item(id="I2", category="art", frame_ref="f", status="needs_appraisal"),
    ]
    t = compute_totals(books, items, 1.25, "INR")
    assert t.book_count == 5 and t.books_identified == 3 and t.books_unidentified == 1 and t.books_needs_appraisal == 1
    assert t.books_replacement_cost == 900.5 and t.books_used_value == 300
    assert t.items_replacement_cost_low == 1000 and t.items_replacement_cost_high == 3000
    assert t.excluded_from_totals == 4  # unidentified, appraisal, missing used value, art
