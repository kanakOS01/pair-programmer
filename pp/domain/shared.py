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
        return "".join(
            difflib.unified_diff(
                self.old.splitlines(keepends=True),
                self.new.splitlines(keepends=True),
                fromfile="/dev/null" if self.is_new_file else str(self.path),
                tofile="/dev/null" if self.is_deletion else str(self.path),
            )
        )
