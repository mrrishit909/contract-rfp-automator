"""Word-level diff between a clause and its proposed redraft, for the side-by-side view."""
from __future__ import annotations

import re
from difflib import SequenceMatcher


def word_diff(original: str, proposed: str) -> list[dict[str, str]]:
    """[{"op": "equal" | "delete" | "insert" | "replace", "a": original words, "b": proposed words}, ...]"""
    a, b = re.findall(r"\S+|\n", original), re.findall(r"\S+|\n", proposed)

    def join(words: list[str]) -> str:
        return " ".join(words).replace(" \n ", "\n").replace("\n ", "\n").replace(" \n", "\n")

    return [{"op": op, "a": join(a[i1:i2]), "b": join(b[j1:j2])}
            for op, i1, i2, j1, j2 in SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()]
