"""The question-answering agent: retrieve, check, answer with citations, verify, or refuse.

It is a small LangGraph graph over :class:`~policy_assistant.index.PolicyIndex`::

    retrieve -> grade --(nothing relevant)--------------------------> refuse
                  \\--> generate -> verify --(every claim cited)-----> answer
                           ^          |----(a claim lost its source)-> generate (once more)
                           |__________|----(the text doesn't answer)-> refuse

* **retrieve** searches the index for the question's ``k`` closest chunks.
* **grade** drops chunks scoring below ``min_score``. If none are left, it refuses
  without asking the model anything.
* **generate** asks an :class:`Answerer` for claims, each citing the chunks it rests on,
  or for a refusal when the chunks don't answer the question.
* **verify** keeps a claim only if it cites at least one chunk that was retrieved. If it
  had to drop a claim, it sends the answerer back once with a note saying which ones.
  If no claim survives, the agent refuses.

So every claim in an answer points at a chunk the agent actually read, and a question
the regulation text doesn't cover gets a refusal instead of a guess.

Two answerers ship with it: :class:`~policy_assistant.answerers.ClaudeAnswerer` (the
real one) and :class:`~policy_assistant.answerers.ExtractiveAnswerer`, which quotes
the best-matching sentence and needs no model, so the tests and CI run without a key.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from .index import Passage, PolicyIndex

Status = Literal["answered", "refused"]

# Why the agent refused. The answerer's own words, if any, are in Answer.detail.
NO_PASSAGES = "no_passages"  # nothing retrieved scored at least min_score
UNSUPPORTED = "unsupported"  # the answerer said the passages don't answer the question
UNCITED = "uncited"  # no claim cited a retrieved passage
MODEL_REFUSED = "model_refused"  # the model declined the request


@dataclass(frozen=True)
class Claim:
    """One statement of the answer and the chunk ids it rests on."""

    text: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Draft:
    """What an answerer returns before verification."""

    answerable: bool
    claims: tuple[Claim, ...] = ()
    reason: str = ""
    refused_by_model: bool = False


class Answerer(Protocol):
    name: str

    def draft(self, question: str, passages: Sequence[Passage], feedback: str = "") -> Draft: ...


@dataclass(frozen=True)
class Answer:
    question: str
    status: Status
    claims: tuple[Claim, ...]
    passages: tuple[Passage, ...]
    reason: str = ""
    detail: str = ""
    attempts: int = 0
    answerer: str = ""

    @property
    def answered(self) -> bool:
        return self.status == "answered"

    def sources(self) -> list[Passage]:
        """The cited passages, in the order the claims first cite them."""
        by_id = {p.chunk_id: p for p in self.passages}
        seen: dict[str, Passage] = {}
        for claim in self.claims:
            for chunk_id in claim.sources:
                if chunk_id in by_id and chunk_id not in seen:
                    seen[chunk_id] = by_id[chunk_id]
        return list(seen.values())

    def render(self) -> str:
        """The answer as text, with numbered citations and a source list."""
        if not self.answered:
            lines = [f"Refused ({self.reason}): {self.detail or _REFUSAL_TEXT[self.reason]}"]
            if self.passages:
                lines += ["", "Closest passages:"]
                lines += [f"- {p.citation} (score {p.score:.3f})" for p in self.passages[:3]]
            return "\n".join(lines)
        number = {p.chunk_id: i for i, p in enumerate(self.sources(), 1)}
        lines = []
        for claim in self.claims:
            marks = "".join(f"[{number[c]}]" for c in claim.sources if c in number)
            lines.append(f"{claim.text} {marks}")
        lines += ["", "Sources:"]
        lines += [f"[{n}] {p.citation}" for p in self.sources() for n in [number[p.chunk_id]]]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "status": self.status,
            "reason": self.reason,
            "detail": self.detail,
            "attempts": self.attempts,
            "answerer": self.answerer,
            "claims": [
                {
                    "text": c.text,
                    "sources": list(c.sources),
                    "citations": [p.citation for p in self.passages if p.chunk_id in c.sources],
                }
                for c in self.claims
            ],
            "passages": [asdict(p) for p in self.passages],
        }


_REFUSAL_TEXT = {
    NO_PASSAGES: "nothing in the index is close enough to this question.",
    UNSUPPORTED: "the retrieved regulation text doesn't answer this question.",
    UNCITED: "no statement in the draft answer was backed by a retrieved passage.",
    MODEL_REFUSED: "the model declined this request.",
}


class State(TypedDict, total=False):
    question: str
    parts: list[str] | None
    passages: list[Passage]
    draft: Draft
    claims: list[Claim]
    feedback: str
    attempts: int
    status: Status
    reason: str
    detail: str


@dataclass
class PolicyAgent:
    index: PolicyIndex
    answerer: Answerer
    k: int = 5
    min_score: float = 0.0
    max_attempts: int = 2
    graph: object = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.graph = self._build()

    def ask(self, question: str, parts: Sequence[str] | None = None) -> Answer:
        state = self.graph.invoke(
            {"question": question, "parts": list(parts) if parts else None, "attempts": 0}
        )
        status = state.get("status", "refused")
        return Answer(
            question=question,
            status=status,
            claims=tuple(state.get("claims", ())) if status == "answered" else (),
            passages=tuple(state.get("passages", ())),
            reason=state.get("reason", ""),
            detail=state.get("detail", ""),
            attempts=state.get("attempts", 0),
            answerer=self.answerer.name,
        )

    # -- the graph -------------------------------------------------------------------

    def _build(self):
        g = StateGraph(State)
        g.add_node("retrieve", self._retrieve)
        g.add_node("grade", self._grade)
        g.add_node("generate", self._generate)
        g.add_node("verify", self._verify)
        g.add_edge(START, "retrieve")
        g.add_edge("retrieve", "grade")
        g.add_conditional_edges("grade", self._after_grade, {"generate": "generate", END: END})
        g.add_edge("generate", "verify")
        g.add_conditional_edges("verify", self._after_verify, {"generate": "generate", END: END})
        return g.compile()

    def _retrieve(self, state: State) -> State:
        return {"passages": self.index.search(state["question"], k=self.k, parts=state["parts"])}

    def _grade(self, state: State) -> State:
        kept = [p for p in state["passages"] if p.score >= self.min_score]
        if not kept:
            return {"passages": kept, "status": "refused", "reason": NO_PASSAGES}
        return {"passages": kept}

    def _after_grade(self, state: State) -> str:
        return END if state.get("status") == "refused" else "generate"

    def _generate(self, state: State) -> State:
        draft = self.answerer.draft(state["question"], state["passages"], state.get("feedback", ""))
        return {"draft": draft, "attempts": state["attempts"] + 1}

    def _verify(self, state: State) -> State:
        draft = state["draft"]
        if draft.refused_by_model:
            return {"status": "refused", "reason": MODEL_REFUSED, "detail": draft.reason}
        if not draft.answerable:
            return {"status": "refused", "reason": UNSUPPORTED, "detail": draft.reason}
        retrieved = {p.chunk_id for p in state["passages"]}
        kept, dropped = [], []
        for claim in draft.claims:
            sources = tuple(dict.fromkeys(s for s in claim.sources if s in retrieved))
            if claim.text.strip() and sources:
                kept.append(Claim(claim.text.strip(), sources))
            else:
                dropped.append(claim)
        if dropped and state["attempts"] < self.max_attempts:
            return {"claims": kept, "feedback": _feedback(dropped), "status": None}
        if not kept:
            return {"claims": [], "status": "refused", "reason": UNCITED, "detail": ""}
        return {"claims": kept, "status": "answered", "reason": "", "detail": ""}

    def _after_verify(self, state: State) -> str:
        return END if state.get("status") else "generate"


def _feedback(dropped: Sequence[Claim]) -> str:
    listed = "\n".join(f"- {c.text!r}" for c in dropped)
    return (
        "In your last answer these statements did not cite any of the passages given, "
        "so they were removed:\n"
        f"{listed}\n"
        "Answer again. Cite the passages each statement rests on, and leave out any "
        "statement the passages don't support."
    )
