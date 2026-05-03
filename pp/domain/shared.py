from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TextDelta:
    text: str

    def __str__(self) -> str:
        return self.text


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
        )


@dataclass
class FileDiff:
    path: Path
    old: str
    new: str
    is_new_file: bool = False
    is_deletion: bool = False

    def diff(self) -> str:
        a = self.old.splitlines(keepends=True)
        b = self.new.splitlines(keepends=True)
        fromfile = "/dev/null" if self.is_new_file else str(self.path)
        tofile = "/dev/null" if self.is_deletion else str(self.path)

        def format_range_unified(start: int, stop: int) -> str:
            beginning = start + 1
            length = stop - start
            if length == 1:
                return str(beginning)
            if not length:
                beginning -= 1
            return f"{beginning},{length}"

        def _unified_diff_no_junk():
            started = False
            for group in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(3):
                if not started:
                    started = True
                    yield f"--- {fromfile}\n"
                    yield f"+++ {tofile}\n"

                first, last = group[0], group[-1]
                file1_range = format_range_unified(first[1], last[2])
                file2_range = format_range_unified(first[3], last[4])
                yield f"@@ -{file1_range} +{file2_range} @@\n"

                for tag, i1, i2, j1, j2 in group:
                    if tag == "equal":
                        for line in a[i1:i2]:
                            yield " " + line
                            if not line.endswith("\n"):
                                yield "\n\\ No newline at end of file\n"
                    elif tag in {"replace", "delete"}:
                        for line in a[i1:i2]:
                            yield "-" + line
                            if not line.endswith("\n"):
                                yield "\n\\ No newline at end of file\n"
                    if tag in {"replace", "insert"}:
                        for line in b[j1:j2]:
                            yield "+" + line
                            if not line.endswith("\n"):
                                yield "\n\\ No newline at end of file\n"

        return "".join(_unified_diff_no_junk())
