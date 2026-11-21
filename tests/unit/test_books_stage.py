from claimagent.pipeline.stages.books import holds_spine


def test_a_frame_that_also_read_a_neighbours_whole_spine():
    almanack = "THE ALMANACK OF NAVAL RAVIKANT ERIC JORGENSON"
    assert holds_spine(f"{almanack} | ATADKOR | TVO CINPS | HARPER", [almanack])
    assert holds_spine("TIO CTNPS | A TALON | THE ALMANACK OF NAVAL RAVIKANT ERIC JORGENSOM", [almanack])
    assert not holds_spine("IWO CTMPS | A TADEON | A TAIL OR", [almanack])
    assert not holds_spine("ALMANACK", [almanack])  # one word of it is not the spine
    assert not holds_spine("REWORK JASON FRIED", ["REWORK"])  # a one-word spine proves nothing


def test_photo_and_page_count_thickness_are_combined_unless_the_outline_is_wrong():
    from claimagent.pipeline.stages.books import fuse_thickness

    t, why = fuse_thickness(1.87, (1.2, "173 pages"))  # photo too wide, page count too thin: 1.5 on the shelf
    assert round(t, 2) == 1.5 and "combined" in why
    t, why = fuse_thickness(3.12, (0.8, "100 pages"))  # four times the page count: an outline over two spines
    assert t == 0.8 and "photo gave 3.1 cm" in why
    assert fuse_thickness(None, (1.8, "279 pages"))[0] == 1.8
