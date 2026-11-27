"""Paragraph markers in the Code of Federal Regulations.

CFR paragraphs are numbered (a), (1), (i), (A), then italic (1) and (i) again.
eCFR publishes each paragraph as its own <P> element that starts with its
marker, for example "(b)(1) Except as provided in ..." for a paragraph that
opens a new (b) and its first (1) at once. This module turns that run of
markers into a full label such as ("b", "1", "ii"), so a chunk can be cited
as 12 CFR 1005.6(b)(1)(ii).
"""

from __future__ import annotations

import re

_LEADING = re.compile(r"^\s*\(([0-9]+|[a-z]+|[A-Z]+)\)")

_ROMAN = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9,
    "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17,
    "xviii": 18, "xix": 19, "xx": 20, "xxi": 21, "xxii": 22, "xxiii": 23, "xxiv": 24,
    "xxv": 25,
}  # fmt: skip

# Level 0 is (a), 1 is (1), 2 is (i), 3 is (A). Deeper levels reuse (1) and (i);
# they are rare and are treated as levels 4 and 5 by position.
LOWER, DIGIT, ROMAN, UPPER = 0, 1, 2, 3


def leading_markers(text: str) -> tuple[list[str], str]:
    """Split "(b)(1) Text" into (["b", "1"], "Text")."""
    markers: list[str] = []
    rest = text
    while True:
        m = _LEADING.match(rest)
        if not m:
            break
        markers.append(m.group(1))
        rest = rest[m.end() :]
    return markers, rest.strip()


def _letter_value(s: str) -> int:
    # (a)=1 ... (z)=26, (aa)=27, (bb)=28: eCFR doubles letters past z.
    if len(set(s)) != 1:
        return -1
    return (ord(s[0].lower()) - ord("a") + 1) + 26 * (len(s) - 1)


def _candidates(marker: str) -> list[int]:
    if marker.isdigit():
        return [DIGIT]
    if marker.islower():
        kinds = []
        if marker in _ROMAN:
            kinds.append(ROMAN)
        if _letter_value(marker) > 0:
            kinds.append(LOWER)
        return kinds
    if marker.isupper() and _letter_value(marker) > 0:
        return [UPPER]
    return []


def _value(marker: str, level: int) -> int:
    if level == DIGIT:
        return int(marker)
    if level == ROMAN:
        return _ROMAN[marker]
    return _letter_value(marker)


class LabelTracker:
    """Follows the paragraph tree of one section as its paragraphs go by."""

    def __init__(self) -> None:
        self.stack: list[tuple[int, str]] = []  # (level, marker) from the top down

    def _fits(self, marker: str, level: int) -> bool:
        """Is `marker` the next paragraph at `level` given the current stack?"""
        value = _value(marker, level)
        for i in range(len(self.stack) - 1, -1, -1):
            lvl, prev = self.stack[i]
            if lvl == level:
                return value == _value(prev, level) + 1
        # A level not on the stack must start at its first value, exactly one
        # level below the current paragraph: (a) is followed by (1), never by (i).
        expected = self.stack[-1][0] + 1 if self.stack else LOWER
        return value == 1 and level == expected

    def push(self, marker: str, next_marker: str | None = None) -> tuple[str, ...]:
        """Place `marker` in the tree and return its full label.

        (i), (v) and (x) can be a letter or a roman numeral. After (h)(1),
        "(i)" is either the next letter or the first numeral under (1). The
        marker that follows settles it: (ii) means a numeral, (j) or (1) a
        letter. Without a following marker the numeral reading wins.
        """
        kinds = _candidates(marker)
        if not kinds:
            raise ValueError(f"not a paragraph marker: ({marker})")
        fitting = [k for k in kinds if self._fits(marker, k)] or kinds[:1]
        if len(fitting) > 1 and next_marker is not None:
            fitting = [k for k in fitting if self._next_fits(k, marker, next_marker)] or fitting
        level = max(fitting)
        self._place(level, marker)
        return self.label

    def copy(self) -> LabelTracker:
        other = LabelTracker()
        other.stack = list(self.stack)
        return other

    def _place(self, level: int, marker: str) -> None:
        while self.stack and self.stack[-1][0] >= level:
            self.stack.pop()
        self.stack.append((level, marker))

    def _next_fits(self, level: int, marker: str, next_marker: str) -> bool:
        trial = LabelTracker()
        trial.stack = list(self.stack)
        trial._place(level, marker)
        return any(trial._fits(next_marker, k) for k in _candidates(next_marker))

    @property
    def label(self) -> tuple[str, ...]:
        return tuple(m for _, m in self.stack)


def format_citation(title: int, section: str, label: tuple[str, ...] = ()) -> str:
    """12, "1005.6", ("b", "1") -> "12 CFR 1005.6(b)(1)"."""
    return f"{title} CFR {section}" + "".join(f"({m})" for m in label)
