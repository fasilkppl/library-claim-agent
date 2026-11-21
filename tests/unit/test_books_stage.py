from claimagent.pipeline.stages.books import holds_spine


def test_a_frame_that_also_read_a_neighbours_whole_spine():
    almanack = "THE ALMANACK OF NAVAL RAVIKANT ERIC JORGENSON"
    assert holds_spine(f"{almanack} | ATADKOR | TVO CINPS | HARPER", [almanack])
    assert holds_spine("TIO CTNPS | A TALON | THE ALMANACK OF NAVAL RAVIKANT ERIC JORGENSOM", [almanack])
    assert not holds_spine("IWO CTMPS | A TADEON | A TAIL OR", [almanack])
    assert not holds_spine("ALMANACK", [almanack])  # one word of it is not the spine
    assert not holds_spine("REWORK JASON FRIED", ["REWORK"])  # a one-word spine proves nothing
