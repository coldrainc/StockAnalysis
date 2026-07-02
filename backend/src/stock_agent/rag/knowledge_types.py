from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeChunk:
    source: Path
    heading: str
    content: str

    def render(self) -> str:
        return f"Source: {self.source}\nHeading: {self.heading}\n{self.content}".strip()
