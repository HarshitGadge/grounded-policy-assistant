import re
from pathlib import Path

import pytest

from policy_assistant import demo
from policy_assistant.agent import PolicyAgent
from policy_assistant.answerers import ExtractiveAnswerer
from policy_assistant.index import Passage

ROOT = Path(__file__).parent.parent
XML = (Path(__file__).parent / "fixtures" / "part-9999.xml").read_bytes()


@pytest.fixture(scope="module")
def corpus_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("demo-corpus")


@pytest.fixture(scope="module")
def index(corpus_dir):
    return demo.build_index(
        corpus_dir, "2026-01-01", "hashing-512", [(99, "9999")], fetch=lambda t, p, d: XML
    )


def test_build_index_builds_the_corpus_once_and_reuses_it(corpus_dir, index):
    assert (corpus_dir / "corpus.jsonl").exists()
    n = index.count()
    assert n > 0
    assert index.info()["corpus_date"] == "2026-01-01"

    def no_download(*args):
        raise AssertionError("the corpus was already built")

    again = demo.build_index(corpus_dir, "2026-01-01", "hashing-512", fetch=no_download)
    assert again.count() == n


def test_the_model_is_offered_only_with_its_key():
    assert list(demo.answerer_choices({}).values()) == ["extractive"]
    with_key = demo.answerer_choices({"OPENROUTER_API_KEY": "x"})
    assert list(with_key.values()) == ["extractive", "openrouter"]


def passage(section, part="1005", citation="12 CFR 1005.6(b)(1)", date="2026-11-01"):
    return Passage("id", 0.5, citation, "", "", part, section, "", date)


def test_ecfr_links_go_to_the_section_or_the_part():
    assert (
        demo.ecfr_link(passage("1005.6"))
        == "https://www.ecfr.gov/on/2026-11-01/title-12/section-1005.6"
    )
    supplement = passage("Supplement I to Part 1005", citation="12 CFR part 1005, Supp. I")
    assert demo.ecfr_link(supplement) == "https://www.ecfr.gov/on/2026-11-01/title-12/part-1005"
    assert demo.ecfr_link(passage("1005.6", date="")).startswith(
        f"https://www.ecfr.gov/on/{demo.CORPUS_DATE}/"
    )


def test_view_numbers_claims_and_sources(index):
    answer = PolicyAgent(index, ExtractiveAnswerer(), k=5).ask("What is a preauthorized transfer?")
    shown = demo.view(answer)
    assert shown["answered"]
    assert shown["claims"][0]["text"].startswith("(i) Preauthorized transfer means")
    assert shown["claims"][0]["marks"] == [1]
    first = shown["sources"][0]
    assert first["n"] == 1
    assert first["citation"] == "99 CFR 9999.2(i)"
    assert first["link"].endswith("/title-99/section-9999.2")


def test_view_says_why_it_refused(index):
    answer = PolicyAgent(index, ExtractiveAnswerer(), k=5).ask("What is the capital of France?")
    shown = demo.view(answer)
    assert not shown["answered"]
    assert shown["why"] == demo.REFUSAL_LABEL[answer.reason]
    assert len(shown["closest"]) <= 3


def test_every_figure_on_the_page_is_in_the_readme_with_its_run():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = readme[readme.index("### Results") :]
    for fig in demo.RESULTS:
        assert fig.value in results, fig.label
        assert demo.run_url(fig.run) in results, fig.label
        for number in re.findall(r"\d+(?:\.\d+)?", fig.detail):
            assert number in results, f"{fig.label}: {number}"


def test_the_landing_page_renders_and_answers(monkeypatch, corpus_dir, index):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GPA_DEMO_CORPUS", str(corpus_dir))
    monkeypatch.setenv("GPA_DEMO_ENCODER", "hashing-512")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Grounded Policy Assistant"
    assert [m.value for m in at.metric] == [f.value for f in demo.RESULTS]
    assert at.radio[0].options == ["Extractive baseline (no model)"]

    at.text_area(key="question").set_value("What is a preauthorized transfer?")
    at.button[-1].click().run()
    assert not at.exception
    page = "\n".join(m.value for m in at.markdown)
    assert "Preauthorized transfer means" in page
    assert "[1]" in page
    assert at.expander[0].label == "[1] 99 CFR 9999.2(i)"

    at.text_area(key="question").set_value("What is the capital of France?")
    at.button[-1].click().run()
    assert not at.exception
    assert at.warning[0].value.startswith("Refused.")
