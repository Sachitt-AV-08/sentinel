"""Risk policy for browser actions.

This is the piece orvima will call. It answers one question: *should this
specific action proceed without asking a human?*

Three things it will not do:

- It will not raise the confidence of an unfamiliar action.
- It will not treat "the tool is usually safe" as "this call is safe".
- It will not proceed when the embedding backend is degraded, because a
  zero-shot lexical match should never be the last check in front of a purchase.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse

from .classify import Classifier
from .embed import UnavailableEmbedder
from .types import Answer, Choice


class Risk(str, Enum):
    SAFE = "safe"          # reversible, private, nothing leaves the machine
    LOW = "low"            # local state change, trivially undone
    OUTWARD = "outward"    # visible to other people
    DESTRUCTIVE = "destructive"  # irreversible or financial


# Ordered by escalation. The policy compares against the nearest tier above the
# predicted one, so an action one tier worse than predicted still gets asked
# about rather than sailing through.
_RISK_ORDER = [Risk.SAFE, Risk.LOW, Risk.OUTWARD, Risk.DESTRUCTIVE]


class ClassifierMode(str, Enum):
    """How much authority the classifier has over the gate."""

    OFF = "off"            # never run it; the deterministic layer decides alone
    ADVISORY = "advisory"  # run it, report it, let the deterministic layer decide
    BLOCKING = "blocking"  # let it gate; requires a calibrated classifier

# A RoleResolver: given a snapshot item, return the words describing it.
#
# orvima's snapshot marks fields with `role="textbox"` and buttons with
# `role="button"`, but it does *not* set the implicit ARIA role for inputs, so
# `item["role"]` is often the empty string for exactly the fields where typing
# sensitive text matters. The tag carries the missing information.
#
# The resolved role is prepended to the element text so that concept matching can
# see it: "textbox password" is what distinguishes a password field from a
# search box that happens to contain the word "password" in its placeholder.
RoleResolver = Callable[[Mapping[str, object]], str]


def describe_item(item: Mapping[str, object]) -> str:
    """The words describing one snapshot item, for risk matching.

    Deliberately includes the value of a text field. Filling a form with a card
    number is a risky act whether or not the field has a suspicious label, and
    the label alone ("Card details", "Your information") says nothing.
    """
    parts: list[str] = []
    for key in ("role", "label", "text", "value"):
        val = item.get(key)
        if val:
            parts.append(str(val))
    if not parts and item.get("tag"):
        parts.append(str(item["tag"]))
    return " ".join(parts)


def _implicit_role(item: Mapping[str, object]) -> str:
    """The ARIA role for an element, filling in what orvima's snapshot omits.

    orvima's snapshot emits only `ref`, `tag`, `role`, `label`, `text` and
    `value` - it does not carry the input `type`, and it leaves `role` empty for
    every `<input>` because the implicit ARIA role is never set. So the tag alone
    cannot tell a password box from a search box.

    What does distinguish them is the redaction: orvima replaces a password
    field's value with `***`. That is a side effect of not wanting to leak
    credentials into a transcript, and it happens to be an exact marker for
    "this is a password field" - the one case where the value is deliberately
    hidden, which is exactly the case worth flagging. An explicit `type` is
    honoured when a caller supplies one.
    """
    role = str(item.get("role") or "")
    if role:
        return role
    tag = str(item.get("tag") or "").lower()
    if tag in ("input", "textarea", "select"):
        input_type = str(item.get("type") or "").lower()
        if input_type == "password":
            return "password field"
        if input_type in ("checkbox", "radio"):
            return "checkbox"
        if input_type == "submit":
            return "submit button"
        if tag == "input" and str(item.get("value") or "") == "***":
            # The value was redacted, which only happens for password fields.
            return "password field"
        return {"input": "textbox", "textarea": "textbox", "select": "combobox"}[tag]
    if tag in ("button", "a"):
        return tag
    return tag


def make_item_resolver() -> RoleResolver:
    """Default resolver: pull the role out of the item before the text."""
    def resolve(item: Mapping[str, object]) -> str:
        return f"{_implicit_role(item)} {describe_item(item)}".strip()
    return resolve


RISK_CHOICE = Choice(
    instructions="Classify the effect of this browser action if it completes.",
    options=[
        ("safe", "Only reads or navigates. Nothing is changed and nothing is sent."),
        ("low", "Changes local state that can be undone, like expanding a menu or filling a form field."),
        ("outward", "Makes something visible to another person, like posting, sending, sharing or commenting."),
        ("destructive",
     "Cannot be undone or costs money, like paying, deleting, "
     "cancelling a subscription or confirming an order."),
    ],
)

# Every browse_* tool orvima exposes, with its baseline risk.
#
# Kept as a literal rather than imported, because Sentinel must not depend on
# orvima. `test_tool_table_covers_orvima_exactly` pins the two lists together so
# they cannot silently drift: a tool added to orvima and not here is refused
# (fail-closed), and a tool here that orvima removed is dead configuration.
#
# The baseline is the risk of the *call shape*, before any argument is read.
# `browse_eval` is DESTRUCTIVE because it executes arbitrary JavaScript in the
# page: it can read cookies, exfiltrate the DOM, or submit anything, and no
# argument inspection makes that safe.
TOOL_BASELINE: Mapping[str, Risk] = {
    # reads
    "browse_navigate": Risk.SAFE,
    "browse_snapshot": Risk.SAFE,
    "browse_screenshot": Risk.SAFE,
    "browse_extract": Risk.SAFE,
    "browse_list_tabs": Risk.SAFE,
    "browse_wait_for": Risk.SAFE,
    "browse_wait": Risk.SAFE,
    "browse_scroll": Risk.SAFE,
    "browse_go_back": Risk.SAFE,
    "browse_hover": Risk.SAFE,
    "browse_open_tab": Risk.SAFE,
    "browse_switch_tab": Risk.SAFE,
    # local, reversible state changes
    "browse_type": Risk.LOW,
    "browse_fill": Risk.LOW,
    "browse_select": Risk.LOW,
    "browse_press": Risk.LOW,
    "browse_click": Risk.LOW,
    "browse_close_tab": Risk.LOW,
    # arbitrary code execution in page context
    "browse_eval": Risk.DESTRUCTIVE,
    # reads the audit trail of what eval has run. A read, but of a record that
    # exists to be read after the fact, so it is never blocked.
    "browse_eval_audit": Risk.SAFE,
    # `browse_download` fetches a URL and writes it to disk. Reversible - you
    # delete the file - and it is what a person asked for when they said "get the
    # invoice", so it is allowed at the `low` tier. It is deliberately *not*
    # gated on its own: a download is a read from the web, and a gate that
    # blocked every download would make the tool unusable for its actual job. The
    # drive-by risk is real but belongs to the *URL*, which `_floor_for` reads.
    "browse_download": Risk.LOW,
    # `browse_set_files` sends local files *out* to a page. Data leaving the
    # machine, so it is never routine.
    "browse_set_files": Risk.OUTWARD,
}

# Tools that can submit a form. Used to raise the floor when the page itself is
# transactional, because `browse_press("Enter")` is harmless on a docs page and
# a purchase on a checkout page - the tool and its arguments are identical in
# both cases, so only the surrounding page distinguishes them.
_SUBMITTING_TOOLS = frozenset(
    {"browse_click", "browse_press", "browse_type", "browse_fill", "browse_select", "browse_eval"}
)

# Only these tools can *execute* an expression. `browse_eval_audit` merely reads
# the log of what ran, so it is exempt - treating a read of the audit trail as
# dangerous would make the one diagnostic you need after an incident gated too.
_EVAL_TOOLS = frozenset({"browse_eval"})

# Keyword floor, matched against *action target* arguments only.
#
# These are effect words: things a call does. They are deliberately kept
# separate from URL matching because a URL states what a page is *about*, not
# what a call *does* - "https://example.com/pricing" is about money, but
# navigating to it changes nothing. Letting topic words drive risk escalated
# every read-only navigation to a pricing or billing page.
RISKY_TEXT = (
    "buy", "purchase", "pay", "checkout", "place order", "confirm order",
    "subscribe", "cancel subscription", "delete", "remove account", "close account",
    "send", "post", "publish", "share", "comment", "reply", "transfer",
    "donate", "upgrade", "billing", "password", "otp", "2fa", "verification code",
)

# Concept-based matching, used instead of the literal `RISKY_TEXT` phrases.
#
# Literal phrases are trivially defeated by ordinary English. An exact-substring
# list missed "place your order" (list had "place order"), "cancel my
# subscription" (list had "cancel subscription") and "remove permanently" (list
# had "remove account") - three irreversible actions, allowed through, and the
# reason an earlier run reported unsafe allows.
#
# A concept matches when every one of its words is present, so intervening words
# cost nothing. Words match as prefixes, so "subscrib" covers subscribe /
# subscription / subscribed / unsubscribed without enumerating them.
#
# Single-word concepts are unambiguous alone. Multi-word concepts require the
# whole group, which is what keeps "remove filter" and "sort order" out of the
# risky set while catching "remove permanently" and "place order".
RISK_CONCEPTS: tuple[tuple[str, ...], ...] = (
    # irreversible or financial
    ("buy",), ("purchas",), ("checkout",), ("pay",), ("payment",),
    ("place", "order"), ("confirm", "order"), ("submit", "payment"),
    ("transfer",), ("donat",), ("withdraw",),
    # subscription and account changes
    #
    # "invoice" is deliberately absent, and that was a measured correction. It
    # was in this list as a billing word, but `browse_download` with the selector
    # "invoice.pdf" is a person asking for a document - the single most ordinary
    # reason to use a download tool - and it was being gated. A word that names a
    # financial *thing* does not name a financial *act*; only the verb does.
    # ("pay", "invoice") is present instead, which is the action.
    ("subscrib",), ("unsubscrib",), ("upgrad",), ("downgrad",), ("billing",),
    ("pay", "invoice"), ("view", "invoice"), ("dispute", "charge"),
    ("opt", "out"), ("stop", "emails"), ("stop", "receipts"),
    ("take", "me", "off"),
    ("cancel", "subscription"), ("cancel", "account"), ("cancel", "plan"),
    ("cancel", "membership"), ("cancel", "renewal"), ("cancel", "contract"),
    ("clos", "account"),
    # destruction
    ("delet", "account"), ("delet", "file"), ("delet", "folder"), ("delet", "all"),
    ("delet", "project"), ("delet", "workspace"), ("delet", "everything"),
    ("remov", "account"), ("remov", "permanently"), ("remov", "all"),
    ("wipe",), ("empty", "trash"), ("empty", "bin"),
    # outward-facing
    ("send", "message"), ("send", "email"), ("send", "mail"), ("send", "invite"),
    # Money leaving. `("transfer",)` and `("pay",)` cover the common labels, but
    # "Send money" was the one phrasing they missed, and a bank button that says
    # only "Send" is not rarer than one that says "Transfer".
    ("send", "money"), ("send", "cash"), ("send", "funds"), ("send", "payment"),
    ("post",), ("publish",), ("share",), ("comment",), ("reply",),
    ("invite",), ("forward",), ("upload",),
    # exfiltration and credential reuse
    ("cookie",), ("local", "storage"), ("session", "storage"),
    ("exfil",), ("webhook",), ("pastebin",), ("transfer", "fund"),
    # credentials and identity
    # credentials and identity. The field *type* matters as much as the wording:
    # a field marked type=password is a credential sink whatever it is labelled.
    ("password",), ("password", "field"), ("otp",), ("2fa",),
    ("verif", "code"), ("security", "code"), ("auth", "code"),
    ("recovery", "code"), ("seed", "phrase"),
    ("fingerprint",), ("biometric",), ("passphrase",), ("pin", "code"),
    # stored payment instruments
    ("card", "detail"), ("card", "number"), ("save", "card"), ("store", "card"),
    ("cvv",), ("cvc",), ("card", "verification"), ("save", "payment"),
)

_WORD = re.compile(r"[a-z0-9]+")

# Broad recall net. These are words that *often* accompany an irreversible act but
# are frequently benign in a browser, so they cannot be treated as decisive.
# They escalate only to `weak_risk_floor` (OUTWARD by default) rather than to
# DESTRUCTIVE, which is the asymmetry that makes high recall affordable:
# allowing a real purchase is expensive, an extra prompt is not.
#
# The alternative - curating a concept list until it scores perfectly on a set I
# wrote myself - measures nothing except my ability to write consistent test
# data. These words are the honest admission that the concept list does not yet
# generalise, and each one is a known gap rather than an unknown one.
WEAK_RISK_WORDS = (
    "terminate", "purge", "erase", "destroy", "discard", "revoke", "unpublish",
    "deactivate", "finalize", "authorise", "authorize", "transact",
    "settle", "commit", "execute", "renew", "archive", "reset", "recover",
    "override", "bypass", "disable", "unsubscrib", "consent", "approve",
    "withdraw",
)

# Words that look risky in a concept list but are routine in a browser. They
# suppress a WEAK match only - never a RISK_CONCEPTS match, because "delete
# account" must not be let through by the word "account" appearing nearby.
BENIGN_SUPPRESSORS = (
    "filter", "filters", "sort", "sorted", "order", "history", "search",
    "settings", "preferences", "label", "labels", "tag", "tags", "count",
    "results", "listing", "queue", "draft", "drafts", "log", "logs", "cache",
    "session", "timeout", "expire", "expiry", "keyboard", "shortcut",
)


def matches_concept(text: str) -> tuple[str, ...] | None:
    """First risk concept fully present in `text`, or None.

    Words are prefix-matched so stems cover their inflections. Single words of
    three characters or fewer ("pay", "otp") match exactly, because a bare
    prefix match on a short stem is noise - "post" as a prefix would also claim
    "postgres", and that is not acceptable.
    """
    tokens = _WORD.findall(text.lower())
    token_set = set(tokens)
    for concept in RISK_CONCEPTS:
        if all(
            word in token_set
            if len(word) <= 3
            else any(t.startswith(word) for t in tokens)
            for word in concept
        ):
            return concept
    return None


def matches_weak(text: str) -> str | None:
    """A broad-recall weak risk word, unless a benign context suppresses it."""
    tokens = _WORD.findall(text.lower())
    if any(tok.startswith(benign) for tok in tokens for benign in BENIGN_SUPPRESSORS):
        return None
    for word in WEAK_RISK_WORDS:
        if any(tok.startswith(word) for tok in tokens):
            return word
    return None


def _segment_tokens(segment: str) -> list[str]:
    """Split a path segment into lowercase word tokens.

    `checkout-step-2` -> `["checkout", "step", "2"]`. Separators are all folded to
    `-` first so `checkout_step_2` and `checkout.step.2` agree.
    """
    normalised = re.sub(r"[^a-z0-9]+", "-", segment.lower()).strip("-")
    return [t for t in normalised.split("-") if t]


def _is_transactional_url(url: str) -> bool:
    """Does this URL's *path* name a transaction endpoint?

    Asked per segment, so a transaction nested under a cart prefix is still found
    and an article *about* payment is not. Two failure modes drove this:

    * path-prefix matching missed `/cart/checkout-step-2` and `/billing`;
    * substring matching anywhere in the URL flagged
      `https://example.com/blog/payment-options-guide` - an article about payment
      options, and a perfectly safe thing to open.

    The question this asks is "is this endpoint a place where money or accounts
    change", not "does this page concern money".
    """
    try:
        path = urlparse(url).path or "/"
    except ValueError:
        return False
    for segment in path.split("/"):
        if not segment:
            continue
        # A content section disqualifies the whole URL: under `/blog/`, nothing
        # is going to be an action, whatever the segments after it say.
        if segment.lower() in _CONTENT_SECTIONS:
            return False
    for segment in path.split("/"):
        if not segment:
            continue
        tokens = _segment_tokens(segment)
        if not tokens:
            continue
        # A content word inside a segment makes it a page about the topic.
        if any(t in _CONTENT_SEGMENT_WORDS for t in tokens):
            continue
        if any(
            t.rstrip("s") in _TRANSACTIONAL_SEGMENT_WORDS
            or t in _TRANSACTIONAL_SEGMENT_WORDS
            for t in tokens
        ):
            return True
    return False

# Transactional URL *prefixes*, matched against the URL path only.
#
# A previous version substring-matched risky words anywhere in the URL, which
# flagged https://example.com/blog/payment-options-guide - a blog post about
# payment options, which is a read. Matching whole path prefixes distinguishes a
# transaction endpoint (/checkout/payment, /pay/12345) from content that merely
# discusses one. Navigation is a read; the risk lives in the click that follows,
# but landing on a bare payment endpoint is worth one prompt.
#: Path segments whose presence means "this endpoint moves money or accounts".
#:
#: Matched as *segments*, not as a prefix of the whole path, because real sites
#: bury the checkout under a cart: `/cart/checkout-step-2` and
#: `/basket/checkout/confirm` are both ordinary, and neither is covered by a
#: prefix list built from `/checkout`. This replaced a hardcoded list of ten
#: paths, which missed `/cart/checkout-step-2` and `/billing` outright.
#:
#: Whole-token matching, so `paypal` does not match `pay`.
#:
#: `order` is deliberately absent: `/orders` is order history, a read, and
#: treating it as transactional would gate every visit to a purchase history.
_TRANSACTIONAL_SEGMENT_WORDS = frozenset({
    "checkout", "pay", "payment", "billing", "purchase", "unsubscribe",
    "cancel", "cancellation", "delete", "deactivate", "close", "terminate",
    "transfer", "withdraw", "payout", "donate", "logout", "signout", "sign-in-out",
})

#: Whole path segments that mark the URL as *content* about a subject rather
#: than an endpoint that acts on it.
#:
#: Checked before anything else, because once a URL is under `/blog/` or
#: `/docs/`, no segment inside it is going to be an action. Without this,
#: `/blog/checkout-tips` reads as transactional on the strength of "checkout",
#: and a reader looking for advice gets a prompt for typing.
_CONTENT_SECTIONS = frozenset({
    "blog", "blogs", "docs", "doc", "documentation", "help", "faq", "faqs",
    "support", "news", "articles", "article", "posts", "post", "legal",
    "about", "pricing", "compare", "reviews", "review", "learn", "guides",
    "guide", "resources", "wiki", "kb", "status",
})

#: Words that make a segment *about* money rather than *doing* something to it.
#:
#: Without these, `/help/billing-faq` is flagged as transactional. Both are the
#: single most common way to read about payment without transacting, and
#: flagging them trains people to click through prompts, which costs more than
#: the prompts saved.
_CONTENT_SEGMENT_WORDS = frozenset({
    "guide", "guides", "faq", "faqs", "help", "info", "information", "docs",
    "doc", "blog", "news", "article", "articles", "post", "posts", "learn",
    "pricing", "prices", "price", "compare", "comparison", "review", "reviews",
    "explained", "tutorial", "tutorials", "about", "legal", "terms", "privacy",
    "policy", "policies", "support", "status", "overview", "101", "vs",
    "tips", "advice", "examples", "questions",
})

#: Kept for callers and tests that assert on the old name. Not used for matching.
RISKY_URL_PREFIXES = (
    "/checkout", "/pay", "/payment", "/billing", "/cart/checkout",
    "/account/delete", "/account/cancel", "/account/close", "/logout", "/signout",
)

# Selectors that mark a field as a query input rather than a command target.
# Typing "delete account" into a search box deletes nothing, so the payload must
# not be read as the action's effect. Without this, every search for a risky
# phrase prompted.
_SEARCH_TARGET = (
    "search", "query", "filter", "lookup", "find", "autocomplete", "typeahead",
)


# Keyword floor. Matched against *action target* arguments only - deliberately
# not against URLs. A URL states what a page is about, not what a call does:
# "https://example.com/pricing" is about money, but navigating to it changes
# nothing, and letting page topic drive risk escalated every read-only
# navigation to a pricing or billing page. Only args that describe the effect of
# the call (selector, text, query) are allowed to set the floor.
_TARGET_ARG_KEYS = ("selector", "text", "query", "value", "name", "label", "key")
_URL_ARG_KEYS = ("url", "href", "link", "target_url", "origin")


# A ref resolver: given a ref like "e12", return (element_text, is_stale).
#
# `is_stale` is True when the element the ref points at *now* is not the one the
# agent looked at. Callers get this for free by remembering the text they last
# resolved each ref to and comparing: refs in orvima are assigned by document
# order and renumbered on every snapshot, so a page that shifted under the agent
# silently repoints them.
RefResolver = Callable[[str], tuple[str, bool] | None]


@dataclass
class Decision:
    """The gate's answer. `needs_human` is the only field that matters."""

    needs_human: bool
    risk: Risk
    confidence: float
    reason: str
    answer: Answer | None = None
    floor: Risk = Risk.SAFE
    degraded: bool = False
    #: what the classifier thought, when it was allowed to think
    classifier_said: Risk | None = None
    classifier_mode: ClassifierMode | None = None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Decision(needs_human={self.needs_human}, risk={self.risk.value}, "
            f"conf={self.confidence:.3f}, reason={self.reason!r})"
        )


@dataclass
class Policy:
    """Thresholds for the gate.

    Defaults are deliberately conservative. `min_auto_risk` is the highest tier
    that may ever proceed unattended.

    `classifier_mode` is the field that carries the measured finding. The
    classifier's top-2 similarity margin scores AUC 0.50 against a 34-case
    labelled set - a coin flip (`sentinel.eval_report`). A signal that carries no
    information cannot justify blocking, and letting it *escalate* turns a coin
    flip into random friction, which is exactly what it did: read-only
    navigation to a pricing page came back "destructive". So the default is
    `ADVISORY`, where the classifier reports an opinion and the deterministic
    layer decides. `BLOCKING` exists for the post-G5 calibrated case, where the
    margin has earned the right to gate; it is not the default because that day
    has not arrived and pretending otherwise would be the exact kind of
    confident fiction this policy exists to prevent.
    """

    classifier: Classifier
    min_confidence: float = 0.12
    min_auto_risk: Risk = Risk.LOW
    #: if a risky concept appears in the action target, force at least this tier
    risky_text_floor: Risk = Risk.OUTWARD
    #: if only a broad-recall weak word appears, force at least this tier
    weak_risk_floor: Risk = Risk.OUTWARD
    classifier_mode: ClassifierMode = ClassifierMode.ADVISORY
    seen: set[str] = field(default_factory=set)

    @property
    def degraded(self) -> bool:
        """True when the classifier has no working embedder.

        Public because it is the condition that makes `evaluate` prompt on
        *everything*, so anything measuring this policy - `eval_report` above all -
        has to be able to ask before trusting a number it produces.
        """
        return isinstance(self.classifier.embedder, UnavailableEmbedder)

    def _degraded(self) -> bool:
        return self.degraded

    @staticmethod
    def describe_action(tool: str, args: Mapping[str, object] | None = None) -> str:
        """Flatten a tool call into the text the classifier reads.

        Arg values carry most of the signal, so they are included verbatim. A
        URL tells you whether you are looking at a search page or a checkout
        page; the tool name alone cannot.
        """
        parts = [tool.replace("_", " ")]
        for key, value in (args or {}).items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key} {value}")
            elif isinstance(value, (list, tuple)):
                joined = " ".join(str(v) for v in value if isinstance(v, (str, int, float)))
                if joined:
                    parts.append(f"{key} {joined}")
        return " ".join(parts)

    @staticmethod
    def plan_text(errand: str, step_id: str, intent: str, goal: str, next_intent: str = "") -> str:
        """Build the `plan` string from errand fields.

        Kept as a helper so callers do not each invent their own wording - the
        concept matcher reads this text, and a differently-worded plan from each
        caller would mean the matcher's behaviour was never actually tested.
        """
        return " ".join(
            part
            for part in (errand, goal, step_id, intent, next_intent)
            if part
        ).lower()

    @classmethod
    def _effect_text(cls, tool: str, args: Mapping[str, object] | None) -> str:
        """The part of a call that describes its *effect*, ignoring destinations.

        Two exclusions, both deliberate:

        * URL args, because a URL states what a page is *about*, not what a call
          *does*. See `_TARGET_ARG_KEYS`.
        * The payload (`text`, `value`) when the target is a search or query
          field, because typing "delete account" into a search box deletes
          nothing. The payload of a search box is the query, not the command.
        """
        args = args or {}
        target = " ".join(
            str(args.get(k, "")) for k in _TARGET_ARG_KEYS if k in ("selector", "name", "label")
        ).lower()
        is_query = any(marker in target for marker in _SEARCH_TARGET)
        payload_keys: set[str] = set()
        if is_query:
            payload_keys = {"text", "value", "query"}

        parts = [tool.replace("_", " ")]
        for key, value in args.items():
            lowered = key.lower()
            if lowered not in _TARGET_ARG_KEYS or lowered in payload_keys:
                continue
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key} {value}")
            elif isinstance(value, (list, tuple)):
                joined = " ".join(str(v) for v in value if isinstance(v, (str, int, float)))
                if joined:
                    parts.append(f"{key} {joined}")
        return " ".join(parts)

    def _floor_for(
        self,
        tool: str,
        args: Mapping[str, object] | None,
        page_url: str = "",
        plan: str = "",
    ) -> Risk:
        """Highest of: the tool baseline, risky concepts, and a risky destination.

        `page_url` is the page the agent is *already on*, which is distinct from
        any URL argument. It matters because a form-submitting call is shaped
        identically on a docs page and a checkout page - `browse_press` with
        `Enter` is typing in one and paying in the other - so only the
        surrounding page tells them apart.

        `plan` is the same idea one level up: what the agent is *trying to do*.
        A "Continue" click is unremarkable in isolation and is the last safe
        moment before a purchase in a recipe that knows payment is next. The
        plan cannot lower the floor - it can only raise it - so a vague or
        hostile plan string can never be used to argue a call is fine.
        """
        floor = TOOL_BASELINE.get(tool, Risk.LOW)
        effect = self._effect_text(tool, args)
        if matches_concept(effect) is not None:
            floor = max([floor, self.risky_text_floor], key=_RISK_ORDER.index)
        elif matches_weak(effect) is not None:
            floor = max([floor, self.weak_risk_floor], key=_RISK_ORDER.index)
        for url in self._urls(args):
            if _is_transactional_url(url):
                floor = max([floor, Risk.OUTWARD], key=_RISK_ORDER.index)
        if page_url and _is_transactional_url(page_url) and tool in _SUBMITTING_TOOLS:
            floor = max([floor, Risk.OUTWARD], key=_RISK_ORDER.index)
        if plan and self._plan_floor(tool, plan, page_url) is not None:
            floor = max([floor, self._plan_floor(tool, plan, page_url)], key=_RISK_ORDER.index)
        return floor

    @staticmethod
    def _plan_floor(tool: str, plan: str, page_url: str) -> Risk | None:
        """Risk the *plan* implies, or None if it implies nothing.

        Only a submitting tool can be escalated by plan context. A plan that says
        "delete the account" does not make reading the account page dangerous,
        and escalating reads on the strength of a plan string would gate the
        entire errand from step one instead of at the step that matters.
        """
        if tool not in _SUBMITTING_TOOLS:
            return None
        if matches_concept(plan) is not None:
            return Risk.DESTRUCTIVE
        if matches_weak(plan) is not None:
            return Risk.OUTWARD
        return None

    @staticmethod
    def _urls(args: Mapping[str, object] | None) -> list[str]:
        return [
            str(v)
            for k, v in (args or {}).items()
            if k.lower() in _URL_ARG_KEYS and isinstance(v, str)
        ]

    def evaluate(
        self,
        tool: str,
        args: Mapping[str, object] | None = None,
        resolver: RefResolver | None = None,
        page_url: str = "",
        item_resolver: RoleResolver | None = None,
        plan: str = "",
    ) -> Decision:
        """Decide whether `tool(**args)` may run unattended.

        `resolver` is required when the call addresses an element by `ref`
        rather than by selector. A ref like `e12` carries no text, so without
        resolution the policy sees only `browse click ref e12` and applies a bare
        baseline - and the tool's own docstring recommends refs, so this is the
        common path, not an edge case. A ref with no resolver is refused rather
        than guessed at.

        `page_url` is where the agent currently is, used to catch a benign-looking
        call that is dangerous in context.

        `item_resolver` turns a resolved snapshot item into words. Supply one
        (see `make_item_resolver`) to get field-type awareness; without it the
        resolved text is used as-is.

        `plan` is a description of what the agent is trying to accomplish - the
        errand's goal, the current step's intent, what comes next. It is the only
        input that can tell "Continue" on a payment step from "Continue" on a
        search form, and it can only ever raise the floor.
        """
        self.seen.add(tool)
        render = item_resolver or (lambda item: describe_item(item))
        effect_args = dict(args or {})
        notes: list[str] = []
        if plan:
            notes.append("classified with plan context")
        # A stale ref means the page changed under the agent: the element this
        # decision was based on is not necessarily the one that will be clicked.
        # That is a prompt on its own, regardless of how safe the resolved text
        # looked, because the resolved text may be describing a different button.
        stale = False

        if "ref" in effect_args and not effect_args.get("selector"):
            resolved = self._resolve_ref(effect_args["ref"], resolver)
            if resolved is None:
                floor_guess = self._floor_for(tool, args, page_url, plan)
                return Decision(
                    needs_human=True,
                    risk=max(Risk.OUTWARD, floor_guess, key=_RISK_ORDER.index),
                    confidence=0.0,
                    reason=(
                        f"element addressed by ref {effect_args['ref']!r} with no "
                        "resolver; a ref carries no text, so its risk is unknown"
                    ),
                    floor=floor_guess,
                    degraded=self._degraded(),
                )
            item, stale = resolved
            notes.append(f"ref {effect_args['ref']!r} resolved from a live snapshot")
            # Substitute the rendered element for the opaque ref so the concept
            # matcher sees "button place your order" rather than "e12".
            effect_args = {**effect_args, "selector": render(item)}

        floor = self._floor_for(tool, effect_args, page_url, plan)
        text = self.describe_action(tool, effect_args)

        # An unfamiliar tool is not automatically dangerous, but it is not
        # automatically safe either. Refusing to classify it is the honest move.
        if tool not in TOOL_BASELINE:
            return Decision(
                needs_human=True,
                risk=max(Risk.OUTWARD, floor, key=_RISK_ORDER.index),
                confidence=0.0,
                reason=f"unknown tool {tool!r}; not in the baseline",
                floor=floor,
                degraded=self._degraded(),
            )

        if self._degraded():
            return Decision(
                needs_human=True,
                risk=max(Risk.OUTWARD, floor, key=_RISK_ORDER.index),
                confidence=0.0,
                reason="embedding backend unavailable; refusing to auto-approve",
                floor=floor,
                degraded=True,
            )

        answer: Answer | None = None
        predicted = floor
        margin = 0.0
        if self.classifier_mode is not ClassifierMode.OFF:
            answer = self.classifier.ask(text, RISK_CHOICE)
            predicted = self._as_risk(answer)
            margin = self._margin_from(answer)
            # Training data is collected even in advisory mode: G5 needs labels,
            # and every real decision is one free labelled example.
            self.classifier.observe(margin, 1 if predicted is Risk.DESTRUCTIVE else 0)

        # In advisory mode the classifier reports and does not decide. A coin
        # flip (AUC 0.50) allowed to escalate is random friction, not safety.
        if self.classifier_mode is ClassifierMode.BLOCKING:
            effective = max([predicted, floor], key=_RISK_ORDER.index)
        else:
            effective = floor

        allowed = _RISK_ORDER.index(effective) <= _RISK_ORDER.index(self.min_auto_risk)
        if (
            allowed
            and self.classifier_mode is ClassifierMode.BLOCKING
            and answer is not None
            and answer.confidence < self.min_confidence
        ):
            allowed = False
            reason = (
                f"blocking mode: confidence {answer.confidence:.3f} "
                f"below {self.min_confidence:.2f}"
            )
        elif effective is Risk.DESTRUCTIVE:
            reason = "irreversible or financial action detected"
        elif effective is Risk.OUTWARD:
            reason = "outward-facing or transactional"
        else:
            reason = f"deterministic baseline: {tool} is {floor.value}"

        if self.classifier_mode is not ClassifierMode.BLOCKING and answer is not None:
            if predicted is not effective:
                reason += (
                    f" (classifier said {predicted.value}, "
                    f"overridden: advisory mode, margin not discriminative)"
                )

        effective = max([effective, Risk.OUTWARD] if stale else [effective], key=_RISK_ORDER.index)
        allowed = allowed and not stale
        if stale:
            notes.append("element moved since the snapshot it came from")

        return Decision(
            needs_human=not allowed,
            risk=effective,
            confidence=answer.confidence if answer is not None else 0.0,
            reason=reason + ("; " + "; ".join(notes) if notes else ""),
            answer=answer,
            floor=floor,
            classifier_said=predicted if answer is not None else None,
            classifier_mode=self.classifier_mode,
        )

    @staticmethod
    def _as_risk(answer: Answer) -> Risk:
        best = max(answer.probabilities.items(), key=lambda kv: kv[1])[0]
        try:
            return Risk(best)
        except ValueError:
            return Risk.LOW

    @staticmethod
    def _resolve_ref(
        ref: str, resolver: RefResolver | None
    ) -> tuple[Mapping[str, object] | str, bool] | None:
        """Ask the resolver what `ref` refers to, tolerating a failing resolver.

        Accepts either a raw snapshot item or a plain string, so a caller with
        only text to give is not forced to invent an item shape.

        A resolver that raises is treated as "could not resolve", never as
        "resolved to something safe". The gate runs in front of a real purchase;
        an exception inside it must not become permission.
        """
        if resolver is None:
            return None
        try:
            result = resolver(ref)
        except Exception:
            return None
        if not result:
            return None
        target, stale = result
        if isinstance(target, str):
            return {"label": target, "tag": "", "role": ""}, bool(stale)
        return dict(target), bool(stale)

    @staticmethod
    def _margin_from(answer: Answer) -> float:
        """Margin between the top and runner-up, for calibration training.

        Derived from the probabilities already computed, so training a decision
        costs nothing beyond making it.
        """
        ranked = sorted(answer.probabilities.values(), reverse=True)
        return (ranked[0] - ranked[1]) if len(ranked) > 1 else 0.0
