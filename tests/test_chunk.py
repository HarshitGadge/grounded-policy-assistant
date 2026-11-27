from policy_assistant.chunk import chunk_section
from policy_assistant.ecfr import parse_part


def chunks_by_section(part_xml, max_chars=1500):
    out = {}
    for s in parse_part(part_xml, 99, "9999"):
        out[s.number] = chunk_section(s, "2026-01-01", max_chars)
    return out


def test_chunks_never_mix_top_level_paragraphs(part_xml):
    for chunks in chunks_by_section(part_xml).values():
        for c in chunks:
            tops = {label[1] for label in c.labels}  # "(b)(1)" -> "b"
            assert len(tops) <= 1


def test_citation_is_the_deepest_shared_paragraph(part_xml):
    cites = [c.citation for c in chunks_by_section(part_xml)["9999.2"]]
    assert cites[0] == "99 CFR 9999.2"  # the intro sentence
    assert "99 CFR 9999.2(a)" in cites
    assert "99 CFR 9999.2(b)" in cites  # (b)(1) and (b)(2) share only (b)


def test_small_chunks_cite_deeper(part_xml):
    cites = [c.citation for c in chunks_by_section(part_xml, max_chars=160)["9999.3"]]
    assert "99 CFR 9999.3(b)(1)" in cites
    assert "99 CFR 9999.3(b)(2)" in cites


def test_chunk_text_starts_with_citation_and_heading(part_xml):
    c = chunks_by_section(part_xml)["9999.3"][0]
    assert c.text.startswith("99 CFR 9999.3(a) Liability of consumer.\n(a) Conditions.")


def test_max_chars_is_respected_except_single_long_paragraphs(part_xml):
    for chunks in chunks_by_section(part_xml, max_chars=200).values():
        for c in chunks:
            body = c.text.split("\n", 1)[1]
            assert len(body) <= 200 or "\n" not in body


def test_appendix_chunks_follow_headings(part_xml):
    appendix = chunks_by_section(part_xml)["Appendix A to Part 9999"]
    assert [c.citation for c in appendix] == [
        "99 CFR part 9999, Appendix A to Part 9999, A-1 Consumer's Liability",
        "99 CFR part 9999, Appendix A to Part 9999, A-2 Business Days",
        "99 CFR part 9999, Appendix A to Part 9999, A-3 Model Form",
    ]


def test_interpretations_are_cited_as_comments(part_xml):
    supp = chunks_by_section(part_xml)["Supplement I to Part 9999"]
    base = "99 CFR part 9999, Supplement I to Part 9999"
    assert [c.citation for c in supp] == [
        f"{base}, comments 2(a)-1 through 2(a)-2",
        f"{base}, comment 2(b)(1)-1",
        f"{base}, comment 2(b)(2)-1",
        f"{base}, comments 3(b)(1)-1 through 3(b)(1)-2",
        f"{base}, comment 3(c)-1",
    ]
    assert (
        supp[3].text.split("\n")[1] == "3(b)(1) Timely notice."
    )  # the heading stays with the text
    small = chunks_by_section(part_xml, max_chars=60)["Supplement I to Part 9999"]
    assert small[0].citation.endswith("comment 2(a)-1")


def test_sub_points_keep_their_comment_number(part_xml):
    # "i." and "ii." belong to comment 1 even when a size split starts a chunk with them.
    small = chunks_by_section(part_xml, max_chars=70)["Supplement I to Part 9999"]
    sub_points = [c for c in small if c.text.split("\n")[2].startswith(("i.", "ii."))]
    assert sub_points
    assert all(c.citation.endswith("comment 3(b)(1)-1") for c in sub_points)


def test_repeated_wording_gets_distinct_ids(part_xml):
    # Model forms repeat the same blank lines; each chunk still needs its own id.
    appendix = chunks_by_section(part_xml, max_chars=20)["Appendix A to Part 9999"]
    repeated = [c for c in appendix if "Name of institution" in c.text]
    assert len(repeated) == 2
    assert repeated[0].text == repeated[1].text
    assert repeated[0].id != repeated[1].id


def test_ids_are_stable_and_depend_on_date(part_xml):
    s = parse_part(part_xml, 99, "9999")[1]
    a = [c.id for c in chunk_section(s, "2026-01-01")]
    b = [c.id for c in chunk_section(s, "2026-01-01")]
    c = [c.id for c in chunk_section(s, "2026-02-01")]
    assert a == b
    assert set(a).isdisjoint(c)
