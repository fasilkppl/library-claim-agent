from claimagent.pipeline.stages import consensus_text


def line(t, conf=1.0):
    return {"text": t, "conf": conf}


def test_leaked_neighbour_line_dropped():
    obs = [[line("Brisson"), line("Einführung in die Philosophie des Mythos")]] * 4 + [
        [line("Brisson"), line("Einführung in die Philosophie des Mythos"), line("Veyne Glaubten die Griechen")]
    ]
    text = consensus_text(obs)
    assert "Veyne" not in text and "Mythos" in text


def test_best_reading_of_a_line_kept():
    obs = [[line("GRIECHISCHIES", 0.5)], [line("GRIECHISCHES", 1.0)], [line("GRIECHISCHES", 1.0)]]
    assert consensus_text(obs) == "GRIECHISCHES"


def test_width_outlier_observation_does_not_vote():
    obs = [
        [line("Tom Reiss"), line("DER ORIENTALIST")],
        [line("Tom Reiss"), line("Hofstadter Gödel")],
        [line("Tom Reiss")],
    ]
    text = consensus_text(obs, widths=[120, 290, 125])
    assert "ORIENTALIST" in text and "Hofstadter" not in text
