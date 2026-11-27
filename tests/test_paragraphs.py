import pytest

from policy_assistant.paragraphs import LabelTracker, format_citation, leading_markers


def labels(markers):
    """Run a list of marker runs through a tracker, as parse_part does."""
    tracker = LabelTracker()
    out = []
    flat = [m for run in markers for m in run]
    i = 0
    for run in markers:
        for m in run:
            nxt = flat[i + 1] if i + 1 < len(flat) else None
            out_label = tracker.push(m, nxt)
            i += 1
        out.append(out_label)
    return out


def test_leading_markers_splits_runs():
    assert leading_markers("(b)(1) Except as provided") == (["b", "1"], "Except as provided")
    assert leading_markers("No marker here") == ([], "No marker here")
    assert leading_markers("(i)  Text") == (["i"], "Text")


def test_nested_labels_follow_the_tree():
    got = labels([["a"], ["1"], ["2"], ["b"], ["1"], ["i"], ["ii"], ["A"], ["c"]])
    assert got == [
        ("a",),
        ("a", "1"),
        ("a", "2"),
        ("b",),
        ("b", "1"),
        ("b", "1", "i"),
        ("b", "1", "ii"),
        ("b", "1", "ii", "A"),
        ("c",),
    ]


def test_i_after_h_is_a_letter_when_j_follows():
    got = labels([["h"], ["1"], ["i"], ["j"]])
    assert got[2] == ("i",)
    assert got[3] == ("j",)


def test_i_after_h1_is_a_numeral_when_ii_follows():
    got = labels([["h"], ["1"], ["i"], ["ii"]])
    assert got[2] == ("h", "1", "i")
    assert got[3] == ("h", "1", "ii")


def test_v_and_x_as_numerals_and_letters():
    got = labels([["a"], ["1"], ["i"], ["ii"], ["iii"], ["iv"], ["v"], ["vi"]])
    assert got[6] == ("a", "1", "v")
    run = [[c] for c in "abcdefghijklmnopqrstuvwx"]
    assert labels(run)[-2] == ("w",)
    assert labels(run)[-1] == ("x",)


def test_doubled_letters_past_z():
    run = [[c] for c in "abcdefghijklmnopqrstuvwxyz"] + [["aa"], ["bb"]]
    assert labels(run)[-1] == ("bb",)


def test_rejects_non_markers():
    with pytest.raises(ValueError):
        LabelTracker().push("ab")


def test_format_citation():
    assert format_citation(12, "1005.6", ("b", "1", "ii")) == "12 CFR 1005.6(b)(1)(ii)"
    assert format_citation(12, "1005.6") == "12 CFR 1005.6"
