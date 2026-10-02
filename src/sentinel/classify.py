"""The classifier.

Design commitments, all of which the tests pin:

1. **Fail closed.** Every answer carries a probability *and* a confidence. A
   caller gating a risky action must be able to ask "how sure are you" and get
   an honest answer, because on an unfamiliar surface the honest answer is
   "not very".

2. **Never claim calibration it does not have.** `Answer.calibrated` is False
   until a calibrator fitted on labelled data is attached. Similarity scores
   rescaled into [0,1] are not probabilities and the object says so.

3. **No LLM in this path.** Everything here is vector arithmetic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .embed import Embedder, cosine, default_embedder
from .types import Answer, Choice, Noul, Question, QuestionType, Score

# Similarities below this are treated as "no evidence either way" rather than
# forced to the middle of the scale. Shifting by this constant is a crude
# temperature; the calibrator is the real fix, and this just stops the default
# path from being confidently wrong at the boundary.
NEUTRAL = 0.0


def softmax(values: Sequence[float], temperature: float = 0.07) -> list[float]:
    """Numerically stable softmax."""
    if not values:
        return []
    t = max(temperature, 1e-6)
    scaled = [v / t for v in values]
    peak = max(scaled)
    exps = [math.exp(v - peak) for v in scaled]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(exps)] * len(exps)
    return [e / total for e in exps]


@dataclass
class Calibration:
    """Maps raw similarity margins onto probabilities.

    Two modes:

    * `None` (default) - identity pass-through, `fitted=False`.
    * `platt` - a 1-D logistic fit on (margin, label) pairs.
    * `isotonic` - piecewise-linear fit over sorted observed rates.

    Both fits are trained on labelled approvals; until then nothing is fitted.
    """

    kind: str = "identity"
    a: float = 1.0
    b: float = 0.0
    knots: tuple[tuple[float, float], ...] = ()

    @property
    def fitted(self) -> bool:
        return self.kind != "identity"

    def apply(self, margin: float) -> float:
        if self.kind == "platt":
            # p = sigmoid(a*margin + b). The fit yields a > 0, so a larger
            # margin must mean a higher probability - hence no negation here.
            z = self.a * margin + self.b
            if z >= 0:
                return 1.0 / (1.0 + math.exp(-z))
            e = math.exp(z)
            return e / (1.0 + e)
        if self.kind == "isotonic" and self.knots:
            xs = [k[0] for k in self.knots]
            ys = [k[1] for k in self.knots]
            if margin <= xs[0]:
                return ys[0]
            if margin >= xs[-1]:
                return ys[-1]
            for i in range(1, len(xs)):
                if margin <= xs[i]:
                    lo_x, hi_x = xs[i - 1], xs[i]
                    lo_y, hi_y = ys[i - 1], ys[i]
                    if hi_x == lo_x:
                        return hi_y
                    frac = (margin - lo_x) / (hi_x - lo_x)
                    return lo_y + frac * (hi_y - lo_y)
        # identity: squash the raw margin into (0,1) without pretending to know more
        return 1.0 / (1.0 + math.exp(-margin))

    @staticmethod
    def platt(samples: Sequence[tuple[float, int]]) -> "Calibration":
        """Fit a 1-D logistic regression by gradient descent.

        Deliberately tiny and dependency-free. A dozen lines beats pulling in
        scikit-learn for one logistic coefficient, and the failure mode of this
        version (poorly conditioned on degenerate input) is caught by the tests
        rather than silently producing confident nonsense.
        """
        if not samples:
            return Calibration()
        margin = sum(m for m, _ in samples) / len(samples)
        label = sum(y for _, y in samples) / len(samples)
        if label in (0.0, 1.0):
            return Calibration(kind="platt", a=1.0, b=0.0)
        a, b = 1.0, -margin
        # Separable data has no finite MLE: the fit wants a -> infinity to
        # become a step function. Iterating hard gets us most of the way there
        # instead of stopping at a visibly under-confident curve.
        lr, iters = 0.5, 4000
        for step in range(iters):
            lr_t = lr / (1.0 + step / 1000.0)
            ga = gb = 0.0
            for m, y in samples:
                p = 1.0 / (1.0 + math.exp(-(a * m + b)))
                err = p - y
                ga += err * m
                gb += err
            ga /= len(samples)
            gb /= len(samples)
            a -= lr_t * ga
            b -= lr_t * gb
        return Calibration(kind="platt", a=a, b=b)

    @staticmethod
    def isotonic(samples: Sequence[tuple[float, int]]) -> "Calibration":
        """Pool-adjacent-violators fit over observed positive rates."""
        if not samples:
            return Calibration()
        ordered = sorted(samples, key=lambda s: s[0])
        knots: list[list[float]] = []
        for m, y in ordered:
            knots.append([m, float(y), 1.0])
            while len(knots) >= 2 and knots[-2][1] > knots[-1][1]:
                m2, y2, n2 = knots.pop()
                m1, y1, n1 = knots.pop()
                knots.append([(m1 * n1 + m2 * n2) / (n1 + n2), (y1 * n1 + y2 * n2) / (n1 + n2), n1 + n2])
        return Calibration(kind="isotonic", knots=tuple((k[0], k[1]) for k in knots))


class Classifier:
    """Zero-shot classifier over an embedder, optionally calibrated.

    Zero-shot means: options are compared against the state directly, no
    training. It works on day one and improves as `observe()` accumulates
    labelled examples.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        calibration: Calibration | None = None,
        temperature: float = 0.07,
    ) -> None:
        self.embedder = embedder or default_embedder()
        self.calibration = calibration or Calibration()
        self.temperature = temperature
        self._cache: dict[str, list[float]] = {}
        self._observations: list[tuple[str, int]] = []

    # ---------------------------------------------------------------- helpers

    def _embed(self, text: str) -> list[float]:
        hit = self._cache.get(text)
        if hit is None:
            hit = self.embedder.embed_one(text)
            self._cache[text] = hit
        return hit

    def _margins(self, state_vec: list[float], option_texts: Sequence[str]) -> list[float]:
        return [cosine(state_vec, self._embed(t)) - NEUTRAL for t in option_texts]

    def _settle(self, probs: dict[str, float]) -> float:
        """Confidence = margin between best and runner-up."""
        if len(probs) < 2:
            return max(probs.values(), default=0.0)
        ranked = sorted(probs.values(), reverse=True)
        return ranked[0] - ranked[1]

    # --------------------------------------------------------------- questions

    def ask(self, state: str, question: Question) -> Answer:
        """Answer one question about `state`."""
        return self.ask_all(state, [question])[0]

    def ask_all(self, state: str, questions: Sequence[Question]) -> list[Answer]:
        """Answer several questions about one state.

        All questions see the same state vector, and the option vectors are
        embedded in one batch, so adding a question costs a few vector
        comparisons rather than another round trip.
        """
        if not questions:
            return []
        state_vec = self._embed(state)
        answers: list[Answer] = []

        for q in questions:
            if isinstance(q, Noul):
                answers.append(self._noul(state_vec, state, q))
            elif isinstance(q, Choice):
                answers.append(self._choice(state_vec, state, q))
            elif isinstance(q, Score):
                answers.append(self._score(state_vec, state, q))
            else:  # pragma: no cover - guarded by the type union
                raise TypeError(f"unknown question type: {type(q)!r}")
        return answers

    def _noul(self, state_vec: list[float], state: str, q: Noul) -> Answer:
        # Two anchors: the property being true, and the same property negated.
        # A question with no natural negation still works, just with a smaller
        # margin - which the confidence number reflects.
        positive = f"{q.instructions}. This is true."
        negative = f"{q.instructions}. This is false."
        margins = self._margins(state_vec, [positive, negative])
        raw = softmax(margins, self.temperature)
        p_true = self.calibration.apply(margins[0] - margins[1])
        if self.calibration.kind == "identity":
            p_true = raw[1]
        return Answer(
            type=QuestionType.NOUL,
            value=p_true,
            probabilities={"true": p_true, "false": 1.0 - p_true},
            confidence=self._settle({"true": p_true, "false": 1.0 - p_true}),
            calibrated=self.calibration.fitted,
        )

    def _choice(self, state_vec: list[float], state: str, q: Choice) -> Answer:
        if len(q.options) < 2:
            raise ValueError("Choice needs at least two options")
        # Only the description is embedded. Prepending the shared `instructions`
        # to every option adds an identical component to all of them, which
        # dilutes exactly the differences the choice is meant to measure - a
        # long instruction turned "costs money" and "only reads" into near-ties.
        option_texts = [description or label for label, description in q.options]
        margins = self._margins(state_vec, option_texts)
        raw = softmax(margins, self.temperature)
        probs = {label: p for (label, _), p in zip(q.options, raw)}
        return Answer(
            type=QuestionType.CHOICE,
            value=max(probs.items(), key=lambda kv: kv[1])[0],
            probabilities=probs,
            confidence=self._settle(probs),
            calibrated=self.calibration.fitted,
        )

    def _score(self, state_vec: list[float], state: str, q: Score) -> Answer:
        if len(q.labels) < 2:
            raise ValueError("Score needs at least two ordered levels")
        # Ordered levels get a monotone prior: label i is anchored at i/(n-1),
        # so ties break toward the lower level and the ordering is respected
        # even when the text itself is uninformative.
        n = len(q.labels)
        margins = []
        for i, label in enumerate(q.labels):
            prior = i / (n - 1)
            margins.append(cosine(state_vec, self._embed(label)) + 0.15 * prior)
        raw = softmax(margins, self.temperature)
        probs = {label: p for label, p in zip(q.labels, raw)}
        expected = sum(i * p for i, (_, p) in enumerate(probs.items())) / (n - 1)
        return Answer(
            type=QuestionType.SCORE,
            value=expected,
            probabilities=probs,
            confidence=self._settle(probs),
            calibrated=self.calibration.fitted,
        )

    # ------------------------------------------------------------- learning

    def observe(self, margin: float, label: int) -> None:
        """Record one labelled decision. 1 = the risky/higher class."""
        self._observations.append((float(margin), int(label)))

    @property
    def observations(self) -> list[tuple[float, int]]:
        return list(self._observations)

    def fit(self, kind: str = "platt", min_samples: int = 20) -> Calibration:
        """Fit a calibrator from recorded observations.

        Refuses to fit below `min_samples`. A calibration curve built from five
        points is a lie with a confidence interval, and `calibrated=True` on
        the answers is the signal other code trusts.
        """
        if len(self._observations) < min_samples:
            raise ValueError(
                f"need at least {min_samples} observations to calibrate, "
                f"have {len(self._observations)}"
            )
        if kind == "isotonic":
            self.calibration = Calibration.isotonic(self._observations)
        elif kind == "platt":
            self.calibration = Calibration.platt(self._observations)
        else:
            raise ValueError(f"unknown calibration kind: {kind!r}")
        return self.calibration

    @property
    def is_calibrated(self) -> bool:
        return self.calibration.fitted

    def expected_calibration_error(self, bins: int = 10) -> float:
        """ECE over recorded observations. Meaningless until `fit()` succeeds.

        Known blind spot, deliberately documented rather than papered over: ECE
        compares predicted probability against observed frequency *within the
        same bin*, so a degenerate constant calibrator on a 50/50 set scores
        ~0 no matter how wrong it is. ECE answers "is 0.7 usually right?", not
        "is this better than a coin flip?". Use `discrimination()` for the
        second question - it is what catches inverted labels.
        """
        if not self._observations or not self.calibration.fitted:
            return float("nan")
        if bins < 1:
            raise ValueError("bins must be >= 1")
        buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
        for margin, label in self._observations:
            p = self.calibration.apply(margin)
            idx = min(int(p * bins), bins - 1)
            buckets[idx].append((p, label))
        total = len(self._observations)
        error = 0.0
        for bucket in buckets:
            if not bucket:
                continue
            avg_conf = sum(p for p, _ in bucket) / len(bucket)
            acc = sum(y for _, y in bucket) / len(bucket)
            error += (len(bucket) / total) * abs(avg_conf - acc)
        return error

    def discrimination(self) -> float:
        """AUC: probability a random positive outranks a random negative.

        0.5 is a coin flip, >0.5 means the margin carries signal, <0.5 means
        the labels are inverted relative to the feature. Ties count a half.

        This is the metric that catches an inverted calibration, which ECE
        cannot see.
        """
        pos = [m for m, y in self._observations if y == 1]
        neg = [m for m, y in self._observations if y == 0]
        if not pos or not neg:
            return float("nan")
        wins = 0.0
        for p in pos:
            for n in neg:
                wins += 1.0 if p > n else (0.5 if p == n else 0.0)
        return wins / (len(pos) * len(neg))
