"""Sentinel - local structured classification for agent action gating.

Zero network calls for decisions, no LLM in the path, and an explicit
`calibrated` flag on every answer so nothing downstream has to guess whether a
probability means anything yet.

Quick start:

    from sentinel import Classifier, Policy, HashingEmbedder

    policy = Policy(Classifier(HashingEmbedder()))
    d = policy.evaluate("browse_click", {"selector": "search button"})
    if d.needs_human:
        ...  # ask the user

    d = policy.evaluate("browse_click", {"selector": "place your order"})
    assert d.needs_human   # always true, regardless of classifier confidence
"""

from __future__ import annotations

from .classify import Calibration, Classifier, softmax
from .embed import (
    DEFAULT_MODEL,
    Embedder,
    HashingEmbedder,
    OllamaEmbedder,
    UnavailableEmbedder,
    cosine,
    default_embedder,
    describe,
)
from .policy import (
    RISK_CONCEPTS,
    RISKY_TEXT,
    RISKY_URL_PREFIXES,
    TOOL_BASELINE,
    ClassifierMode,
    Decision,
    Policy,
    RefResolver,
    Risk,
    RoleResolver,
    describe_item,
    make_item_resolver,
    matches_concept,
    matches_weak,
)
from .types import Answer, Choice, Noul, Question, QuestionType, Score

__version__ = "0.2.0"

__all__ = [
    "Answer",
    "Calibration",
    "Choice",
    "Classifier",
    "ClassifierMode",
    "DEFAULT_MODEL",
    "Decision",
    "Embedder",
    "HashingEmbedder",
    "Noul",
    "OllamaEmbedder",
    "Policy",
    "Question",
    "QuestionType",
    "RISK_CONCEPTS",
    "RISKY_TEXT",
    "RISKY_URL_PREFIXES",
    "RefResolver",
    "Risk",
    "RoleResolver",
    "Score",
    "TOOL_BASELINE",
    "UnavailableEmbedder",
    "cosine",
    "default_embedder",
    "describe",
    "describe_item",
    "make_item_resolver",
    "matches_concept",
    "matches_weak",
    "softmax",
    "__version__",
]
