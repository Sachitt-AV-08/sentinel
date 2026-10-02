"""Transactional URL detection, and the measurement guard in `full_report`.

Both exist because a measurement lied.

`_is_transactional_url` used to be ten hardcoded path prefixes. That list
missed `/cart/checkout-step-2` and `/billing` - real URLs, found by running the
*live* gate the way `api.py` builds it and noticing what it waved through.

`full_report` used to silently produce "18 of 34 prompted unnecessarily"
whenever the embedder was unavailable, because a degraded embedder makes the
policy prompt on everything. It read as a catastrophic regression of the rules.
It was Ollama being down. Both tests below are written to fail against the old
behaviour.
"""

from __future__ import annotations

import pytest

from sentinel.policy import (
    _CONTENT_SECTIONS,
    _is_transactional_url as transactional,
)


# ------------------------------------------------------- the missed URLs ----


@pytest.mark.parametrize(
    "url",
    [
        "https://shop.example/checkout",
        # The two that the old ten-prefix list missed outright.
        "https://shop.example/cart/checkout-step-2",
        "https://shop.example/billing",
        "https://shop.example/basket/checkout/confirm",
        "https://shop.example/pay/12345",
        "https://shop.example/payment/method",
        "https://shop.example/account/delete",
        "https://shop.example/account/cancel-plan",
        "https://shop.example/settings/subscription/cancel",
        "https://shop.example/wallet/transfer-funds",
        "https://bank.example/accounts/close",
        "https://shop.example/logout",
    ],
)
def test_transaction_endpoints_are_caught(url: str) -> None:
    assert transactional(url) is True, f"missed a transaction endpoint: {url}"


@pytest.mark.parametrize(
    "url",
    [
        # The two failure modes that make precision matter. A reader looking for
        # advice, and a reader looking at history.
        "https://example.com/blog/payment-options-guide",
        "https://shop.example/help/billing-faq",
        "https://shop.example/blog/checkout-tips",
        "https://shop.example/support/billing-help",
        "https://shop.example/pricing",
        "https://shop.example/orders",
        "https://shop.example/account/orders/history",
        "https://shop.example/checkout-guide",
        "https://shop.example/reviews",
        "https://shop.example/plans",
        "https://shop.example/cart",
        "https://shop.example/products/widget",
        "https://shop.example/account/settings",
        "https://example.com/legal/terms",
        "https://example.com/",
    ],
)
def test_pages_about_money_are_left_alone(url: str) -> None:
    assert transactional(url) is False, f"flagged a read as a transaction: {url}"


def test_paypal_is_not_pay() -> None:
    """Whole-token matching, so a payment *provider* is not a payment."""
    assert transactional("https://shop.example/paypal") is False
    assert transactional("https://shop.example/pay") is True


def test_a_content_section_disqualifies_the_whole_url() -> None:
    """Once a URL is under /blog/, nothing inside it is going to be an action.

    This is why the section check runs before per-segment matching rather than
    being folded into it: `/blog/checkout-tips` has no content word in the
    segment `checkout-tips`, only in the segment before it.
    """
    assert "blog" in _CONTENT_SECTIONS
    assert transactional("https://x.example/blog/checkout") is False
    assert transactional("https://x.example/checkout") is True


def test_separator_style_does_not_change_the_verdict() -> None:
    """Sites spell the same page `/checkout-step-2`, `_` or `.`."""
    forms = [
        "https://shop.example/cart/checkout-step-2",
        "https://shop.example/cart/checkout_step_2",
        "https://shop.example/cart/checkout.step.2",
    ]
    assert all(transactional(u) is True for u in forms), forms


def test_a_malformed_url_is_not_a_transaction() -> None:
    """Fail toward "not a URL", which for this function means "not evidence"."""
    assert transactional("") is False
    assert transactional("not a url at all") is False


def test_orders_history_is_a_read_not_a_purchase() -> None:
    """`order` is deliberately not a transactional word.

    Treating `/orders` as a purchase would gate every visit to a history page,
    which is the kind of friction that makes people approve without reading.
    """
    assert transactional("https://shop.example/orders") is False
    assert transactional("https://shop.example/cart/confirm-order") is False


def test_recall_and_precision_on_a_labelled_set() -> None:
    """One number each, so a future edit cannot quietly trade one for the other.

    The accepted miss is `/cart/confirm-order`: navigating there charges nothing,
    and the page-URL rule plus the confirm button itself still catch the act. A
    false positive is worse than that miss - it trains people to click through.
    """
    risky = [
        "https://shop.example/checkout",
        "https://shop.example/cart/checkout-step-2",
        "https://shop.example/basket/checkout/confirm",
        "https://shop.example/billing",
        "https://shop.example/pay/12345",
        "https://shop.example/payment/method",
        "https://shop.example/account/delete",
        "https://shop.example/account/cancel-plan",
        "https://shop.example/settings/subscription/cancel",
        "https://shop.example/logout",
        "https://shop.example/wallet/transfer-funds",
        "https://bank.example/accounts/close",
    ]
    safe = [
        "https://example.com/blog/payment-options-guide",
        "https://example.com/help/billing-faq",
        "https://shop.example/pricing",
        "https://shop.example/orders",
        "https://shop.example/account/orders/history",
        "https://shop.example/paypal",
        "https://shop.example/cart",
        "https://shop.example/products/widget",
        "https://shop.example/support/billing-help",
        "https://example.com/legal/terms",
        "https://shop.example/account/settings",
        "https://shop.example/reviews",
        "https://shop.example/plans",
        "https://shop.example/blog/checkout-tips",
        "https://example.com/",
    ]
    tp = [u for u in risky if transactional(u)]
    fp = [u for u in safe if transactional(u)]

    recall = len(tp) / len(risky)
    precision = len(tp) / (len(tp) + len(fp))
    assert recall >= 0.95, f"recall dropped to {recall:.2f}: {set(risky) - set(tp)}"
    assert precision >= 0.98, f"precision dropped to {precision:.2f}: {fp}"


# --------------------------------------------- the live-gate consequence ----


def test_the_rules_alone_catch_what_the_page_url_would_have() -> None:
    """Why fixing this matters even though the live gate never sees a page.

    `api.py` builds the gate with no `page_url`, so
    `_is_transactional_url(page_url)` cannot fire in production. The argument-URL
    path is the only one that works there, which is why its recall is the
    number worth holding a line on.
    """
    from sentinel import Classifier, ClassifierMode, HashingEmbedder, Policy

    policy = Policy(Classifier(HashingEmbedder()), classifier_mode=ClassifierMode.OFF)
    for url in ("https://shop.example/checkout", "https://shop.example/billing"):
        decision = policy.evaluate("browse_navigate", {"url": url})
        assert decision.needs_human is True, url


# ----------------------------------------------- the measurement guard ----


def dead_embedder():
    """The real thing `default_embedder()` hands back when Ollama is down.

    Not a hand-rolled fake: the guard is keyed on `Policy.degraded`, which is an
    `isinstance` check against `UnavailableEmbedder`. A fake with a `dim`
    attribute would pass right through it, and the test would prove nothing.
    """
    from sentinel.embed import UnavailableEmbedder

    return UnavailableEmbedder("ConnectError: nothing is listening")


def test_full_report_refuses_to_measure_with_a_dead_embedder() -> None:
    """The measurement must not lie about a dead service.

    With a degraded embedder the policy takes its fail-closed branch and prompts
    on every action. Reporting that as "18 of 34 prompted unnecessarily, 0
    allowed correctly" is a true statement about the wrong thing, and it reads
    as a rules regression rather than a missing service.
    """
    from sentinel.eval_report import full_report

    with pytest.raises(RuntimeError, match="unavailable embedder"):
        full_report(dead_embedder())


def test_the_guard_message_says_how_to_measure_offline() -> None:
    """A guard that only says "no" makes people guess."""
    from sentinel.eval_report import full_report

    with pytest.raises(RuntimeError) as exc:
        full_report(dead_embedder())
    message = str(exc.value)
    assert "HashingEmbedder" in message, message
    assert "ollama serve" in message.lower(), message


def test_the_guard_keys_on_the_policy_not_on_the_embedder_name() -> None:
    """A substring match on the description could drift; this cannot.

    `degraded` is the same flag `evaluate` uses to take its prompt-on-everything
    branch, so the guard and the behaviour it is guarding cannot disagree.
    """
    from sentinel import Classifier, ClassifierMode, HashingEmbedder, Policy

    live = Policy(Classifier(HashingEmbedder()), classifier_mode=ClassifierMode.OFF)
    dead = Policy(Classifier(dead_embedder()), classifier_mode=ClassifierMode.OFF)

    assert live.degraded is False
    assert dead.degraded is True
    # And the consequence the guard is protecting, stated directly.
    assert dead.evaluate("browse_click", {"selector": "Next page"}).needs_human is True
    assert live.evaluate("browse_click", {"selector": "Next page"}).needs_human is False


def test_full_report_still_works_with_a_live_embedder() -> None:
    from sentinel import ClassifierMode, HashingEmbedder
    from sentinel.eval_report import full_report

    report = full_report(HashingEmbedder(), classifier_mode=ClassifierMode.OFF)
    assert report["dev_set"]["allowed_unsafely"] == 0
    assert report["dev_set"]["prompted_unnecessarily"] == 0
    assert report["adversarial_set"]["allowed_unsafely"] == 0
    assert report["adversarial_set"]["prompted_unnecessarily"] == 0