"""Landing page for the Grounded Policy Assistant: what it is, how it scores, and a box to
ask it a question. Run it with ``streamlit run streamlit_app.py``.

On first start it builds the index from eCFR (see ``policy_assistant.demo``), which takes
about a minute. ``GPA_DEMO_CORPUS`` sets where the corpus is kept (a folder that already
holds a built corpus is reused) and ``GPA_DEMO_ENCODER`` the embedding model.
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import streamlit as st

from policy_assistant import demo
from policy_assistant.agent import PolicyAgent
from policy_assistant.answerers import make_answerer
from policy_assistant.embed import DEFAULT_MODEL
from policy_assistant.openrouter import OpenRouterError

st.set_page_config(page_title="Grounded Policy Assistant", page_icon="⚖️", layout="wide")

CORPUS_DIR = os.environ.get("GPA_DEMO_CORPUS") or str(
    Path(tempfile.gettempdir()) / f"gpa-corpus-{demo.CORPUS_DATE}"
)
ENCODER = os.environ.get("GPA_DEMO_ENCODER") or DEFAULT_MODEL


@st.cache_resource(show_spinner="Building the index from eCFR (first visit only, about a minute)")
def get_index(corpus_dir: str, encoder: str):
    return demo.build_index(corpus_dir, encoder=encoder), threading.Lock()


st.title("Grounded Policy Assistant")
st.markdown(
    "Ask a question about four U.S. consumer-finance regulations. It answers **only from "
    "the regulation text it retrieves**, cites the exact paragraph each claim rests on, "
    "and **refuses** when the text doesn't support an answer."
)
links = st.columns([1, 1, 1, 3])
links[0].link_button("Code on GitHub", demo.REPO)
links[1].link_button("Evaluated with RAGBench", demo.RAGBENCH)
links[2].link_button("Back to the booth", demo.BOOTH)

st.subheader("How it scores")
for col, fig in zip(st.columns(len(demo.RESULTS)), demo.RESULTS, strict=True):
    col.metric(fig.label, fig.value)
    col.caption(f"{fig.detail}. [CI run {fig.run}]({demo.run_url(fig.run)})")

st.subheader("Ask it")
choices = demo.answerer_choices()
if "question" not in st.session_state:
    st.session_state.question = demo.EXAMPLES[0]
st.caption("Try an example, or write your own:")
for col, example in zip(st.columns(len(demo.EXAMPLES)), demo.EXAMPLES, strict=True):
    if col.button(example, key=f"example-{example}", use_container_width=True):
        st.session_state.question = example

with st.form("ask"):
    question = st.text_area("Question", key="question", height=80)
    left, right = st.columns(2)
    label = left.radio("Answerer", list(choices), horizontal=True)
    parts = right.multiselect(
        "Only search these parts (optional)",
        [cite.split()[-1] for cite, _, _ in demo.REGULATIONS],
        format_func=lambda p: next(f"{c} ({r})" for c, r, _ in demo.REGULATIONS if p in c),
    )
    asked = st.form_submit_button("Ask", type="primary")

if len(choices) == 1:
    st.caption(
        "This page runs the extractive baseline, which quotes the best-matching sentence "
        "and needs no model. The model scores above come from the free OpenRouter model."
    )

if asked and question.strip():
    index, lock = get_index(CORPUS_DIR, ENCODER)
    try:
        with lock, st.spinner("Retrieving, drafting and checking citations"):
            agent = PolicyAgent(index, make_answerer(choices[label]), k=5)
            answer = agent.ask(question.strip(), parts=parts or None)
    except (OpenRouterError, ValueError, RuntimeError) as err:
        st.error(f"The model couldn't answer just now: {err}")
    else:
        shown = demo.view(answer)
        if shown["answered"]:
            for claim in shown["claims"]:
                marks = "".join(f"[{n}]" for n in claim["marks"])
                st.markdown(f"- {claim['text']} **{marks}**")
            st.markdown("**Sources**")
            for src in shown["sources"]:
                with st.expander(f"[{src['n']}] {src['citation']}"):
                    st.write(src["text"])
                    st.markdown(f"[Read it on eCFR]({src['link']}) · similarity {src['score']}")
        else:
            st.warning(f"Refused. {shown['why']}")
            if shown["detail"]:
                st.caption(shown["detail"])
            if shown["closest"]:
                st.markdown("**Closest passages**")
                for src in shown["closest"]:
                    st.markdown(f"- [{src['citation']}]({src['link']}) · similarity {src['score']}")

st.subheader("How it works")
st.graphviz_chart(demo.GRAPH)
st.markdown(
    "- **retrieve** takes the 5 closest chunks from Qdrant (bge-small-en-v1.5 embeddings).\n"
    "- **grade** drops weak matches and refuses without calling a model if none are left.\n"
    "- **generate** drafts claims in a fixed JSON schema that can only cite the passages "
    "it was shown, or says the text doesn't answer.\n"
    "- **verify** keeps a claim only if it cites a retrieved chunk, gives the model one "
    "more try if it drops one, and refuses if none survive. So every citation points at "
    "text the agent actually read."
)

st.subheader("What it reads")
st.table(
    [{"Part": c, "Regulation": r, "Covers": what} for c, r, what in demo.REGULATIONS],
)
st.caption(
    f"Regulation text from eCFR as of {demo.CORPUS_DATE}, a U.S. government work in the "
    "public domain. This is a portfolio project, not legal advice."
)
