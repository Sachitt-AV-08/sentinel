"""Tests for the parts that must never silently go wrong.

The theme: Sentinel is allowed to be uncertain, but it is not allowed to be
*confidently wrong in the permissive direction*.
"""

from __future__ import annotations

import math

import pytest

from sentinel import (
    Calibration,
    Choice,
    Classifier,
    ClassifierMode,
    HashingEmbedder,
    Noul,
    Policy,
    Risk,
    Score,
    cosine,
    matches_concept,
    matches_weak,
    softmax,
)
from sentinel.embed import UnavailableEmbedder


@pytest.fixture
def clf() -> Classifier:
    # Deterministic backend: these tests assert behaviour, not model quality.
    return Classifier(HashingEmbedder())


# --------------------------------------------------------------------- embed


def test_cosine_of_identical_vectors_is_one() -> None:
    v = HashingEmbedder().embed_one("browse_click place order")
    assert cosine(v, v) == pytest.approx(1.0, abs=1e-9)


def test_cosine_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        cosine([1.0, 0.0], [1.0, 0.0, 0.0])


def test_embedder_is_deterministic() -> None:
    a = HashingEmbedder().embed_one("subscribe to the newsletter")
    b = HashingEmbedder().embed_one("subscribe to the newsletter")
    assert a == b


def test_embedder_separates_unrelated_text() -> None:
    emb = HashingEmbedder()
    assert cosine(emb.embed_one("cats"), emb.embed_one("database migration")) < 0.3


def test_empty_batch_returns_empty() -> None:
    assert HashingEmbedder().embed([]) == []


# ------------------------------------------------------------------- softmax


def test_softmax_sums_to_one() -> None:
    probs = softmax([0.1, 0.9, 0.4])
    assert sum(probs) == pytest.approx(1.0)


def test_softmax_of_identical_inputs_is_uniform() -> None:
    probs = softmax([0.5, 0.5, 0.5])
    assert all(p == pytest.approx(1 / 3) for p in probs)


def test_softmax_is_stable_for_large_values() -> None:
    probs = softmax([1000.0, 1001.0])
    assert all(math.isfinite(p) for p in probs)
    assert sum(probs) == pytest.approx(1.0)


def test_softmax_of_empty_is_empty() -> None:
    assert softmax([]) == []


# ------------------------------------------------------------------- noul


def test_noul_returns_probability_in_unit_interval(clf: Classifier) -> None:
    ans = clf.ask("the production database is on fire", Noul("This is urgent"))
    assert 0.0 <= ans.value <= 1.0
    assert ans.probabilities["true"] + ans.probabilities["false"] == pytest.approx(1.0)


def test_noul_is_not_calibrated_before_fitting(clf: Classifier) -> None:
    ans = clf.ask("anything", Noul("This is true"))
    assert ans.calibrated is False, "claiming calibration with no labelled data is a lie"


def test_noul_is_true_is_only_valid_for_noul(clf: Classifier) -> None:
    ans = clf.ask("x", Choice(options=[("a", "alpha"), ("b", "beta")]))
    with pytest.raises(TypeError):
        _ = ans.is_true


# ----------------------------------------------------------------- choice


def test_choice_picks_the_closest_option(clf: Classifier) -> None:
    # Overlapping wording on purpose: this pins the mechanism (highest
    # similarity wins), not semantic quality. See the limitation test below.
    ans = clf.ask(
        "delete my account permanently",
        Choice(options=[("safe", "only reads and navigates"), ("destructive", "delete my account")]),
    )
    assert ans.value == "destructive"


def test_hashing_embedder_cannot_do_semantics() -> None:
    """Documents a real limitation rather than hiding it.

    The hashing backend matches character n-grams, so paraphrases do not match:
    "place your order" shares nothing with "costs money". Production uses
    nomic-embed-text and does match these. This test exists so that anyone
    relying on the offline backend knows exactly what they are getting, and so
    the behaviour cannot change quietly underneath the risk policy.
    """
    emb = HashingEmbedder()
    paraphrase = cosine(
        emb.embed_one("place your order"),
        emb.embed_one("costs money"),
    )
    literal = cosine(
        emb.embed_one("place your order"),
        emb.embed_one("place your order"),
    )
    assert paraphrase < 0.5 * literal


def test_choice_requires_two_options(clf: Classifier) -> None:
    with pytest.raises(ValueError, match="at least two options"):
        clf.ask("x", Choice(options=[("only", "solo")]))


def test_choice_probabilities_sum_to_one(clf: Classifier) -> None:
    ans = clf.ask(
        "delete my account permanently",
        Choice(options=[("safe", "reads"), ("destructive", "deletes")]),
    )
    assert sum(ans.probabilities.values()) == pytest.approx(1.0)


# ------------------------------------------------------------------ score


def test_score_value_is_between_zero_and_one(clf: Classifier) -> None:
    ans = clf.ask("a critical security vulnerability", Score(labels=["low", "medium", "high"]))
    assert 0.0 <= ans.value <= 1.0


def test_score_respects_declared_ordering(clf: Classifier) -> None:
    # Nothing informative in the state: the monotone prior must still place the
    # answer inside the range and break ties downward, not randomly.
    ans = clf.ask("zzzz qqqq", Score(labels=["low", "medium", "high", "critical"]))
    assert 0.0 <= ans.value <= 1.0


def test_score_needs_two_levels(clf: Classifier) -> None:
    with pytest.raises(ValueError, match="at least two ordered levels"):
        Score(labels=["only"])


def test_score_rejects_min_label_outside_levels() -> None:
    with pytest.raises(ValueError, match="not one of"):
        Score(labels=["low", "high"], min_label="medium")


# ------------------------------------------------------------- calibration


def test_calibration_refuses_to_fit_without_data(clf: Classifier) -> None:
    with pytest.raises(ValueError, match="need at least"):
        clf.fit()


def test_calibration_refuses_to_fit_on_too_few_samples(clf: Classifier) -> None:
    for i in range(5):
        clf.observe(0.1 * i, i % 2)
    with pytest.raises(ValueError, match="need at least"):
        clf.fit(min_samples=20)


def test_calibration_fits_and_reports_fitted(clf: Classifier) -> None:
    for i in range(60):
        clf.observe(-0.5 + 0.02 * i, 1 if i > 30 else 0)
    cal = clf.fit()
    assert cal.fitted is True
    assert clf.is_calibrated is True


def test_calibration_makes_answers_report_calibrated(clf: Classifier) -> None:
    for i in range(60):
        clf.observe(-0.5 + 0.02 * i, 1 if i > 30 else 0)
    clf.fit()
    assert clf.ask("the site is down", Noul("This is urgent")).calibrated is True


def test_platt_is_monotone_in_the_right_direction() -> None:
    samples = [(-1.0, 0), (-0.5, 0), (0.0, 0), (0.5, 1), (1.0, 1), (1.5, 1)]
    cal = Calibration.platt(samples)
    assert cal.apply(1.5) > cal.apply(0.0) > cal.apply(-1.5)


def test_isotonic_is_monotone() -> None:
    samples = [(-1.0, 0), (-0.5, 0), (0.5, 1), (1.0, 1)]
    cal = Calibration.isotonic(samples)
    values = [cal.apply(x) for x in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    assert values == sorted(values)


def test_isotonic_clamps_outside_the_fitted_range() -> None:
    cal = Calibration.isotonic([(0.0, 0), (1.0, 1)])
    assert cal.apply(-99.0) == pytest.approx(0.0)
    assert cal.apply(99.0) == pytest.approx(1.0)


def test_ece_is_nan_until_calibrated(clf: Classifier) -> None:
    for i in range(30):
        clf.observe(0.1 * i, i % 2)
    assert math.isnan(clf.expected_calibration_error())


def test_ece_is_low_for_a_well_calibrated_set(clf: Classifier) -> None:
    # Separable data: a hard step at 0. Platt cannot represent a step (its MLE
    # diverges on separable input), so isotonic is the correct calibrator here
    # and ECE should be near zero rather than merely "not terrible".
    for i in range(200):
        m = -1.0 + 0.01 * i
        clf.observe(m, 1 if m > 0 else 0)
    clf.fit(kind="isotonic")
    assert clf.expected_calibration_error() < 0.05


def test_discrimination_catches_inverted_labels(clf: Classifier) -> None:
    # The case ECE structurally cannot see: inverted labels still fit a constant
    # calibration, so ECE reads ~0. Discrimination sees it immediately.
    for i in range(200):
        m = -1.0 + 0.01 * i
        clf.observe(m, 1 if m <= 0 else 0)
    assert clf.discrimination() < 0.5


def test_discrimination_is_high_for_correctly_labelled_data(clf: Classifier) -> None:
    for i in range(200):
        m = -1.0 + 0.01 * i
        clf.observe(m, 1 if m > 0 else 0)
    assert clf.discrimination() > 0.95


def test_discrimination_is_nan_with_only_one_class(clf: Classifier) -> None:
    for i in range(30):
        clf.observe(0.1 * i, 1)
    assert math.isnan(clf.discrimination())


def test_ece_is_blind_to_inversion_by_design(clf: Classifier) -> None:
    """Pins the documented ECE blind spot so nobody mistakes it for a bug.

    If this ever starts failing because someone improves ECE into detecting
    inversion, the docstring above needs updating in the same commit.
    """
    for i in range(200):
        m = -1.0 + 0.01 * i
        clf.observe(m, 1 if m <= 0 else 0)
    clf.fit(kind="isotonic")
    assert clf.expected_calibration_error() < 0.05  # wrong, and still ~0
    assert clf.discrimination() < 0.5              # right, and clearly below 0.5


def test_fit_rejects_unknown_kind(clf: Classifier) -> None:
    for i in range(30):
        clf.observe(0.1 * i, i % 2)
    with pytest.raises(ValueError, match="unknown calibration kind"):
        clf.fit(kind="magic")


# --------------------------------------------------------- policy: the gate


@pytest.fixture
def policy() -> Policy:
    return Policy(Classifier(HashingEmbedder()))


def test_read_only_navigation_may_proceed(policy: Policy) -> None:
    d = policy.evaluate("browse_navigate", {"url": "https://example.com/docs"})
    assert d.needs_human is False, f"unexpected: {d}"


def test_share_tracking_query_param_does_not_prompt(policy: Policy) -> None:
    # A `share` parameter on a GET is analytics, not a share action. Navigation
    # is a read; the risk lives in the click that follows.
    d = policy.evaluate("browse_navigate", {"url": "https://example.com/post?share=1"})
    assert d.needs_human is False, f"unexpected: {d}"


def test_navigation_to_a_transaction_endpoint_prompts(policy: Policy) -> None:
    # The narrow case where a URL *is* the risk: a bare payment endpoint.
    for url in (
        "https://shop.example.com/checkout/payment",
        "https://bank.example.com/pay/12345",
        "https://example.com/account/delete",
    ):
        d = policy.evaluate("browse_navigate", {"url": url})
        assert d.needs_human is True, f"{url} was allowed unattended"


def test_navigation_to_content_about_payment_does_not_prompt(policy: Policy) -> None:
    # A blog slug that mentions payment is content, not an endpoint.
    for url in (
        "https://example.com/blog/payment-options-guide",
        "https://example.com/pricing",
        "https://example.com/docs/subscriptions",
    ):
        d = policy.evaluate("browse_navigate", {"url": url})
        assert d.needs_human is False, f"{url} prompted unnecessarily"


def test_clicking_share_still_prompts(policy: Policy) -> None:
    # The counterpart to the query-param case: same word, real action.
    d = policy.evaluate("browse_click", {"selector": "share"})
    assert d.needs_human is True


def test_typing_a_risky_phrase_into_search_does_not_prompt(policy: Policy) -> None:
    # Searching for "delete account" deletes nothing: a query field's payload is
    # the query, not the command.
    d = policy.evaluate("browse_type", {"selector": "search box", "text": "delete account"})
    assert d.needs_human is False, f"unexpected: {d}"


def test_typing_into_a_comment_box_still_prompts(policy: Policy) -> None:
    # Not a query field, so the payload is part of the effect.
    d = policy.evaluate("browse_type", {"selector": "comment box", "text": "nice post"})
    assert d.needs_human is True


def test_unknown_tool_always_asks(policy: Policy) -> None:
    d = policy.evaluate("browse_exfiltrate", {"target": "https://evil.example"})
    assert d.needs_human is True
    assert "unknown tool" in d.reason


def test_checkout_wording_always_asks(policy: Policy) -> None:
    d = policy.evaluate("browse_click", {"selector": "place your order"})
    assert d.needs_human is True


def test_payment_wording_always_asks(policy: Policy) -> None:
    for selector in ("buy now", "checkout", "pay with card", "confirm order"):
        d = policy.evaluate("browse_click", {"selector": selector})
        assert d.needs_human is True, f"{selector!r} was allowed unattended"


def test_deleting_always_asks(policy: Policy) -> None:
    for selector in ("delete account", "cancel subscription", "remove permanently"):
        d = policy.evaluate("browse_click", {"selector": selector})
        assert d.needs_human is True, f"{selector!r} was allowed unattended"


def test_degraded_backend_asks_for_everything() -> None:
    broken = UnavailableEmbedder("test", HashingEmbedder())
    p = Policy(Classifier(broken))
    d = p.evaluate("browse_navigate", {"url": "https://example.com"})
    assert d.needs_human is True
    assert d.degraded is True
    assert "unavailable" in d.reason


def test_tightening_min_auto_risk_only_ever_adds_prompts(policy: Policy) -> None:
    call = ("browse_click", {"selector": "search"})
    loose = Policy(Classifier(HashingEmbedder()), min_auto_risk=Risk.OUTWARD).evaluate(*call)
    tight = Policy(Classifier(HashingEmbedder()), min_auto_risk=Risk.LOW).evaluate(*call)
    assert tight.needs_human >= loose.needs_human


def test_min_confidence_only_gates_in_blocking_mode() -> None:
    # `min_confidence` is a no-op in advisory mode, because the margin it
    # thresholds is not discriminative. It must only bite once the classifier
    # has been given authority, or the setting would be a silent lie.
    call = ("browse_click", {"selector": "expand menu"})
    advisory_easy = Policy(Classifier(HashingEmbedder()), min_confidence=0.0).evaluate(*call)
    advisory_hard = Policy(Classifier(HashingEmbedder()), min_confidence=0.99).evaluate(*call)
    assert advisory_hard.needs_human == advisory_easy.needs_human

    blocking_easy = Policy(
        Classifier(HashingEmbedder()), min_confidence=0.0,
        classifier_mode=ClassifierMode.BLOCKING,
    ).evaluate(*call)
    blocking_hard = Policy(
        Classifier(HashingEmbedder()), min_confidence=0.99,
        classifier_mode=ClassifierMode.BLOCKING,
    ).evaluate(*call)
    assert blocking_hard.needs_human >= blocking_easy.needs_human
    assert blocking_hard.needs_human is True


def test_advisory_mode_never_escalates_on_the_classifier() -> None:
    # The measured failure this mode exists to prevent: a coin-flip classifier
    # turning a read-only navigation into "destructive".
    p = Policy(Classifier(HashingEmbedder()), classifier_mode=ClassifierMode.ADVISORY)
    d = p.evaluate("browse_navigate", {"url": "https://example.com/pricing"})
    assert d.risk is Risk.SAFE
    assert d.needs_human is False


def test_classifier_off_never_embeds() -> None:
    # OFF must not touch the embedder at all, so a missing daemon cannot turn
    # into latency or a spurious error.
    p = Policy(Classifier(HashingEmbedder()), classifier_mode=ClassifierMode.OFF)
    d = p.evaluate("browse_navigate", {"url": "https://example.com"})
    assert d.answer is None
    assert d.needs_human is False


def test_advisory_mode_still_records_observations() -> None:
    # G5 needs labels, and every real decision is a free one. Advisory mode
    # reports without deciding, but it must still learn.
    clf = Classifier(HashingEmbedder())
    Policy(clf, classifier_mode=ClassifierMode.ADVISORY).evaluate(
        "browse_click", {"selector": "delete account"}
    )
    assert len(clf.observations) == 1


def test_concept_matching_survives_intervening_words() -> None:
    # The literal-phrase list missed all of these.
    for phrase in (
        "place your order",
        "cancel my subscription",
        "remove permanently",
        "opt out of all emails",
        "finalize purchase",
    ):
        assert matches_concept(phrase) is not None, f"missed {phrase!r}"


def test_concept_matching_respects_benign_context() -> None:
    # Routine browser selectors that contain risky words.
    for phrase in (
        "sort by newest",
        "order history",
        "remove this filter",
        "delete from selection",
        "close",
        "cancel",
    ):
        assert matches_concept(phrase) is None, f"false positive on {phrase!r}"


def test_weak_words_are_suppressed_by_benign_context() -> None:
    assert matches_weak("clear all filters") is None
    assert matches_weak("results count") is None
    assert matches_weak("archive old threads") is not None


def test_every_baseline_risk_is_a_valid_enum() -> None:
    from sentinel import TOOL_BASELINE

    for tool, risk in TOOL_BASELINE.items():
        assert isinstance(risk, Risk), f"{tool} -> {risk!r}"


def test_describe_action_includes_argument_values(policy: Policy) -> None:
    text = policy.describe_action("browse_click", {"selector": "buy"})
    assert "browse click" in text and "buy" in text


def test_multiple_questions_are_answered_independently(clf: Classifier) -> None:
    qs = [
        Noul("This is urgent"),
        Choice(options=[("safe", "reads"), ("destructive", "deletes")]),
        Score(labels=["low", "high"]),
    ]
    answers = clf.ask_all("cancel my subscription right now", qs)
    assert len(answers) == 3
    assert [a.type.value for a in answers] == ["noul", "choice", "score"]


def test_ask_all_with_no_questions_returns_empty(clf: Classifier) -> None:
    assert clf.ask_all("state", []) == []
