"""Structured question types.

Mirrors the three shapes a System One style classifier exposes, so a caller can
ask several questions about one state in a single request:

    noul    - yes/no, returns P(true)
    choice  - pick one of a set, returns P(each) plus a confidence
    score   - rate against ordered levels, returns a continuous value

Deliberately *not* a generative interface. Every question is a typed request
with a typed answer; nothing here produces prose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Sequence


class QuestionType(str, Enum):
    NOUL = "noul"
    CHOICE = "choice"
    SCORE = "score"


@dataclass(frozen=True)
class Noul:
    """Yes/no question about a state.

    `instructions` describe the property being judged; they are embedded
    alongside the state so the question itself is part of what is evaluated.
    """

    instructions: str
    type: Literal[QuestionType.NOUL] = QuestionType.NOUL


@dataclass(frozen=True)
class Choice:
    """Pick one option from `options`, scored independently.

    Options are (label, description) pairs. The description is what gets
    embedded, because "delete" alone is ambiguous while "permanently delete the
    user's saved work" is not.
    """

    options: Sequence[tuple[str, str]]
    instructions: str = ""
    type: Literal[QuestionType.CHOICE] = QuestionType.CHOICE

    @property
    def labels(self) -> list[str]:
        return [label for label, _ in self.options]


@dataclass(frozen=True)
class Score:
    """Rate a state against ordered levels.

    Levels run low to high. `labels` is what the caller passes; the ordering is
    what makes a continuous value meaningful, so an out-of-order declaration is
    a programming error rather than a silently wrong score.
    """

    labels: Sequence[str]
    instructions: str = ""
    min_label: str | None = None
    type: Literal[QuestionType.SCORE] = QuestionType.SCORE

    def __post_init__(self) -> None:
        if len(self.labels) < 2:
            raise ValueError("Score needs at least two ordered levels")
        if self.min_label is not None and self.min_label not in self.labels:
            raise ValueError(
                f"min_label {self.min_label!r} is not one of {list(self.labels)!r}"
            )

    @property
    def levels(self) -> list[tuple[str, str]]:
        """(label, description) pairs; description defaults to the label."""
        return [(label, label) for label in self.labels]


Question = Noul | Choice | Score


@dataclass
class Answer:
    """One question's answer.

    `probability` is only meaningful once the classifier has been calibrated on
    labelled data. Before that it is a similarity score wearing a probability's
    shape, and `calibrated` says so.
    """

    type: QuestionType
    value: object
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    calibrated: bool = False

    @property
    def is_true(self) -> float:
        if self.type is not QuestionType.NOUL:
            raise TypeError("is_true is only valid for a noul answer")
        return float(self.value)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.type is QuestionType.NOUL:
            return f"Noul({self.value:.3f})"
        top = max(self.probabilities.items(), key=lambda kv: kv[1], default=("-", 0))
        return f"{self.type.value}({top[0]}={top[1]:.3f}, conf={self.confidence:.3f})"
