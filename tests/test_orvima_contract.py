"""Contract tests against the real orvima tool surface.

These import `orvima.tools` directly rather than a copy of the list. A literal
copy is a snapshot that quietly rots: someone adds a tool, Sentinel keeps
answering for the old set, and the new tool is "unknown" - which is fail-closed
and therefore safe, but degrades into a prompt on every call, which is the kind
of friction that gets a feature switched off.

Skipped, not failed, when orvima is not importable, so Sentinel's own test suite
still runs standalone.
"""

from __future__ import annotations

import pytest

from sentinel import TOOL_BASELINE, Classifier, ClassifierMode, HashingEmbedder, Policy, Risk

from sentinel import (
    TOOL_BASELINE,
    Classifier,
    ClassifierMode,
    HashingEmbedder,
    Policy,
    Risk,
    describe_item,
    make_item_resolver,
)

orvima = pytest.importorskip("orvima.tools", reason="orvima not on path")


@pytest.fixture
def policy() -> Policy:
    return Policy(Classifier(HashingEmbedder()))


def test_tool_table_matches_orvima_exactly() -> None:
    """No tool may be missing (fail-closed prompts forever) or stale (dead config)."""
    assert set(TOOL_BASELINE) == set(orvima.TOOL_NAMES), {
        "missing_from_sentinel": sorted(set(orvima.TOOL_NAMES) - set(TOOL_BASELINE)),
        "stale_in_sentinel": sorted(set(TOOL_BASELINE) - set(orvima.TOOL_NAMES)),
    }


def test_new_tools_are_risk_appropriate(policy: Policy) -> None:
    """Tools added to orvima after the table was written.

    The contract test above catches that they are *missing*; this catches that
    they were added at the right tier. `browse_set_files` uploads local files to
    a page, so it is data leaving the machine and must never be routine;
    `browse_download` is a read from the web and a reversible write to disk.
    """
    upload = policy.evaluate("browse_set_files", {"paths": ["C:/resume.pdf"]})
    assert upload.needs_human is True
    assert upload.risk is Risk.OUTWARD

    download = policy.evaluate("browse_download", {"selector": "invoice.pdf"})
    assert download.needs_human is False, f"a plain download prompted: {download}"


def test_uploaded_sensitive_files_are_caught_even_when_the_target_is_benign(
    policy: Policy,
) -> None:
    """A file input labelled "Attach" is still an exfiltration route.

    The label says nothing dangerous, so this is the case where the *paths*
    carry all the signal - a passport scan leaving the machine should not be
    allowed out on the strength of a button that says "Attach".
    """
    d = policy.evaluate(
        "browse_set_files",
        {"paths": ["C:/docs/passport.pdf"], "selector": "Attach"},
    )
    assert d.needs_human is True


def test_eval_audit_trail_is_a_read(policy: Policy) -> None:
    """Reading the record of what ran is how you find out what went wrong.

    Gating the audit trail would mean the one tool you reach for after an
    incident is the one that stops.
    """
    d = policy.evaluate("browse_eval_audit", {})
    assert d.needs_human is False
    assert d.risk is Risk.SAFE


def test_reading_cookies_via_eval_is_destructive_anyway(policy: Policy) -> None:
    """`browse_eval` is pinned by tool, so the expression cannot lower it."""
    d = policy.evaluate("browse_eval", {"expression": "document.cookie"})
    assert d.risk is Risk.DESTRUCTIVE
    assert d.needs_human is True


def test_browse_eval_is_destructive(policy: Policy) -> None:
    """`browse_eval` runs arbitrary JavaScript in the page.

    It is the single most dangerous call in the surface - it can read cookies,
    rewrite the DOM, or submit anything - and no argument inspection makes it
    safe, so it is pinned at the top tier rather than inferred per call.
    """
    d = policy.evaluate("browse_eval", {"expression": "document.cookie"})
    assert d.risk is Risk.DESTRUCTIVE
    assert d.needs_human is True


def test_browse_eval_is_destructive_even_for_a_harmless_expression(policy: Policy) -> None:
    # The argument is a script, not a description, so there is no wording that
    # can lower the floor. A seemingly innocent script is still code execution.
    d = policy.evaluate("browse_eval", {"expression": "1 + 1"})
    assert d.needs_human is True
    assert d.risk is Risk.DESTRUCTIVE


def test_read_only_tools_never_prompt(policy: Policy) -> None:
    for tool in (
        "browse_snapshot", "browse_screenshot", "browse_extract", "browse_list_tabs",
        "browse_scroll", "browse_wait", "browse_go_back", "browse_hover",
        "browse_open_tab", "browse_switch_tab",
    ):
        d = policy.evaluate(tool, {})
        assert d.needs_human is False, f"{tool} prompted unnecessarily: {d}"


def test_pressing_enter_on_a_checkout_page_prompts(policy: Policy) -> None:
    """The same call is typing on one page and paying on another.

    `browse_press` with Enter is shaped identically everywhere, so only the page
    it happens on distinguishes committing a purchase from pressing Enter. The
    page context is what closes that hole.
    """
    call = ("browse_press", {"key": "Enter"})
    assert policy.evaluate(*call, page_url="https://shop.example.com/checkout").needs_human is True
    assert policy.evaluate(*call, page_url="https://example.com/docs").needs_human is False


def test_typing_a_card_number_on_a_checkout_page_prompts(policy: Policy) -> None:
    d = policy.evaluate(
        "browse_type",
        {"selector": "#cc-number", "text": "4111 1111 1111 1111"},
        page_url="https://shop.example.com/checkout/payment",
    )
    assert d.needs_human is True


def test_page_context_does_not_prompt_reads_on_a_checkout_page(policy: Policy) -> None:
    """A snapshot taken while shopping is still just a read.

    Applying the page rule to every tool would mean you cannot look at a checkout
    page without a human, which is both annoying and pointless.
    """
    d = policy.evaluate("browse_snapshot", {}, page_url="https://shop.example.com/checkout")
    assert d.needs_human is False


def test_page_context_does_not_prompt_read_only_navigation(policy: Policy) -> None:
    d = policy.evaluate(
        "browse_navigate", {"url": "https://example.com/docs"}, page_url="https://shop.example.com/checkout"
    )
    assert d.needs_human is False


# ------------------------------------------------------------------ refs ----
#
# orvima's snapshot assigns refs by document order and renumbers them on every
# call, so a ref is only meaningful against the snapshot it came from. These
# tests cover the two ways that goes wrong.


def test_ref_without_a_resolver_is_refused(policy: Policy) -> None:
    """`ref e12` carries no text, so there is nothing to classify.

    orvima's own docstring says to prefer refs, which makes this the common path
    rather than an edge case. Defaulting to the bare `browse_click` baseline here
    would auto-approve clicks on the strength of no information at all.
    """
    d = policy.evaluate("browse_click", {"ref": "e12"})
    assert d.needs_human is True
    assert "no resolver" in d.reason


def test_resolved_ref_is_classified_by_element_text(policy: Policy) -> None:
    """Once resolved, the gate sees the real text and behaves normally."""
    d = policy.evaluate(
        "browse_click",
        {"ref": "e12"},
        resolver=lambda ref: ({"label": "Place your order", "tag": "button", "role": "button"}, False),
        item_resolver=make_item_resolver(),
    )
    assert d.needs_human is True
    assert d.risk is not Risk.LOW


def test_resolver_may_return_plain_text(policy: Policy) -> None:
    """A caller with only text should not have to fake a snapshot item."""
    d = policy.evaluate(
        "browse_click", {"ref": "e12"}, resolver=lambda ref: ("Place your order", False)
    )
    assert d.needs_human is True


def test_resolved_benign_ref_does_not_prompt(policy: Policy) -> None:
    d = policy.evaluate(
        "browse_click", {"ref": "e7"}, resolver=lambda ref: ("Next page", False)
    )
    assert d.needs_human is False


def test_password_field_is_detected_from_the_redacted_value(policy: Policy) -> None:
    """A password box is a credential sink whatever it is labelled.

    orvima's snapshot carries no `type` field and leaves `role` empty for every
    `<input>`, so nothing in the item says "password" except the `***` that its
    own redaction substituted for the value. That redaction - a privacy feature -
    is the only reliable marker, and a neutral label like "Your details" gives the
    matcher nothing at all.
    """
    item = {
        "ref": "e4", "tag": "input", "role": "",
        "label": "Your details", "text": "", "value": "***",
    }
    d = policy.evaluate(
        "browse_type",
        {"ref": "e4", "text": "hunter2"},
        resolver=lambda ref: (item, False),
        item_resolver=make_item_resolver(),
    )
    assert d.needs_human is True


def test_search_field_is_not_treated_as_a_credential_sink(policy: Policy) -> None:
    """The counterpart: an ordinary textbox, with a real value, is routine."""
    item = {
        "ref": "e2", "tag": "input", "role": "",
        "label": "Search", "text": "", "value": "",
    }
    d = policy.evaluate(
        "browse_type",
        {"ref": "e2", "text": "winter jacket"},
        resolver=lambda ref: (item, False),
        item_resolver=make_item_resolver(),
    )
    assert d.needs_human is False


def test_empty_password_field_still_counts(policy: Policy) -> None:
    """`***` appears only once something was typed, but the field is still a sink.

    orvima only redacts a non-empty value (`el.value ? ... : ""`), so a fresh
    password box has `value: ""` and looks like any other textbox. An explicit
    `type` is the only remaining signal, which is why the resolver honours it.
    """
    item = {"ref": "e5", "tag": "input", "type": "password", "role": "", "label": "Sign in"}
    assert make_item_resolver()(item).startswith("password field")


def test_card_field_value_is_seen_even_with_a_neutral_label(policy: Policy) -> None:
    """The filled value is part of the risk, not just the field's name.

    A card number typed into a field labelled "Payment method" is sensitive
    input regardless of how bland the label is.
    """
    item = {
        "ref": "e9", "tag": "input", "role": "", "label": "Payment method",
        "text": "", "value": "4111 1111 1111 1111",
    }
    d = policy.evaluate(
        "browse_type", {"ref": "e9", "text": "4111 1111 1111 1111"},
        resolver=lambda ref: (item, False),
        item_resolver=make_item_resolver(),
    )
    assert d.needs_human is True


def test_describe_item_keeps_value_and_label() -> None:
    text = describe_item({"role": "button", "label": "Buy", "text": "Buy now", "value": ""})
    assert "Buy" in text and "Buy now" in text


def test_describe_item_survives_an_empty_item() -> None:
    assert isinstance(describe_item({}), str)


def test_item_resolver_infers_roles_from_tags() -> None:
    resolve = make_item_resolver()
    assert "password" in resolve({"tag": "input", "type": "password"})
    assert "textbox" in resolve({"tag": "input", "type": "text"})
    assert "textbox" in resolve({"tag": "textarea"})
    assert "combobox" in resolve({"tag": "select"})
    assert "button" in resolve({"tag": "button"})


def test_item_resolver_detects_a_redacted_password_value() -> None:
    """`***` means orvima redacted a password field's value."""
    assert make_item_resolver()({"tag": "input", "value": "***"}).startswith("password field")


def test_item_resolver_treats_triple_asterisk_as_a_password_field() -> None:
    """`***` in the value is orvima's redaction marker, so it means password field.

    Known imprecision, accepted deliberately: an input whose real displayed value
    is literally `***` - a masked preview of some kind - is indistinguishable
    from a redaction and gets escalated. The false positive is one extra prompt
    on a field nobody types a password into, which is the cheap direction to be
    wrong in. Inventing a way to tell them apart would mean trusting a second
    source of truth about the DOM that orvima does not expose.
    """
    assert make_item_resolver()({"tag": "input", "value": "***"}).startswith("password field")


def test_item_resolver_prefers_an_explicit_role() -> None:
    resolve = make_item_resolver()
    assert resolve({"tag": "div", "role": "button"}).startswith("button")


def test_stale_ref_prompts_even_when_the_text_looks_safe(policy: Policy) -> None:
    """A ref that moved is a different button than the one that was reviewed.

    The resolved text describes what the element *was*. If the page shifted
    under the agent, the click lands somewhere else entirely, so the decision
    cannot be trusted no matter how benign the remembered text looks.
    """
    d = policy.evaluate(
        "browse_click",
        {"ref": "e7"},
        resolver=lambda ref: ({"label": "Next page", "tag": "button"}, True),
        item_resolver=make_item_resolver(),
    )
    assert d.needs_human is True
    assert "moved" in d.reason


def test_stale_ref_prompts_even_when_the_text_looks_risky(policy: Policy) -> None:
    d = policy.evaluate(
        "browse_click", {"ref": "e3"}, resolver=lambda ref: ("Delete account", True)
    )
    assert d.needs_human is True


def test_resolver_returning_none_is_treated_as_unresolved(policy: Policy) -> None:
    d = policy.evaluate("browse_click", {"ref": "e99"}, resolver=lambda ref: None)
    assert d.needs_human is True
    assert "no resolver" in d.reason


def test_resolver_that_raises_is_treated_as_unresolved(policy: Policy) -> None:
    """An exception inside the gate must never become permission."""

    def boom(ref: str):
        raise RuntimeError("page went away mid-snapshot")

    d = policy.evaluate("browse_click", {"ref": "e5"}, resolver=boom)
    assert d.needs_human is True


def test_selector_wins_when_both_are_supplied(policy: Policy) -> None:
    """orvima rejects ref+selector together, but be explicit rather than lucky."""
    d = policy.evaluate(
        "browse_click",
        {"ref": "e3", "selector": "Delete account"},
        resolver=lambda ref: ("Something harmless", False),
    )
    assert d.needs_human is True


def test_ref_resolution_is_not_needed_for_non_element_tools(policy: Policy) -> None:
    """`browse_press` and friends take no ref, so no resolver is needed."""
    d = policy.evaluate("browse_press", {"key": "Tab"})
    assert d.needs_human is False


def test_every_tool_is_classifiable_without_a_resolver() -> None:
    """No tool may require a ref, or it would be unreachable when unresolvable."""
    for fn in orvima.TOOLS:
        import inspect

        params = inspect.signature(fn).parameters
        if "ref" not in params:
            continue
        # A tool that can only be called by ref is fine, as long as the policy
        # handles it - which the tests above assert. This records the shape.
        assert params["ref"].default is None


def test_classifier_off_mode_never_embeds() -> None:
    """The deterministic path must not depend on the daemon at all.

    This is what makes the gate usable as a hard prerequisite: 0.13ms p95 and no
    network, so it cannot be taken offline by a model that failed to load.
    """
    p = Policy(Classifier(HashingEmbedder()), classifier_mode=ClassifierMode.OFF)
    for tool, args, should_prompt in (
        ("browse_click", {"selector": "Delete account"}, True),
        ("browse_click", {"selector": "Next page"}, False),
        ("browse_navigate", {"url": "https://example.com"}, False),
        ("browse_eval", {"expression": "1"}, True),
    ):
        d = p.evaluate(tool, args)
        assert d.needs_human is should_prompt, f"{tool} {args}: {d}"
        assert d.answer is None
