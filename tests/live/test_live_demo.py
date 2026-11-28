"""Runs the landing page on the real corpus and asks it every example question.

It uses the corpus the live job built (``GPA_CORPUS``) and the default encoder, with the
extractive answerer only, so it sends no model requests. The answers are printed so the
job log shows what a visitor sees for each example.
"""

import os

import pytest

from policy_assistant import demo

pytestmark = pytest.mark.live

CORPUS = os.environ.get("GPA_CORPUS", "data/corpus")


def test_the_landing_page_answers_the_examples(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GPA_DEMO_CORPUS", CORPUS)
    monkeypatch.delenv("GPA_DEMO_ENCODER", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    at = AppTest.from_file("streamlit_app.py", default_timeout=600).run()
    assert not at.exception

    outcomes = {}
    for example in demo.EXAMPLES:
        at.text_area(key="question").set_value(example)
        at.button[-1].click().run()
        assert not at.exception, example
        if at.warning:
            outcomes[example] = at.warning[0].value
        else:
            outcomes[example] = " | ".join(e.label for e in at.expander)
        print(f"\n{example}\n  -> {outcomes[example]}")

    assert outcomes[demo.EXAMPLES[-1]].startswith("Refused."), "the out-of-scope example"
    answered = [q for q, o in outcomes.items() if not o.startswith("Refused.")]
    assert answered, "no example was answered"
