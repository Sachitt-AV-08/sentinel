# Sentinel

Local classification for deciding whether an agent action needs a human.

No LLM in the decision path. No API keys. No network calls except to a local
Ollama daemon, and even those are optional.

```python
from sentinel import Classifier, Policy, default_embedder

policy = Policy(Classifier(default_embedder()))

d = policy.evaluate("browse_navigate", {"url": "https://example.com/docs"})
d.needs_human   # False - read-only navigation

d = policy.evaluate("browse_click", {"selector": "place your order"})
d.needs_human   # True  - irreversible and financial
d.reason        # "irreversible or financial action detected"
```

---

## What it actually does

Given a tool name and its arguments, Sentinel returns `needs_human: bool` plus
the reasoning. It does not plan, generate, or explain.

Two question types exist for general use beyond gating:

```python
from sentinel import Choice, Noul, Score, Classifier

clf = Classifier()
clf.ask("the site returned a 500", Noul("This is an outage"))     # P(true)
clf.ask("please wire the funds", Choice(options=[("ok", "proceed"),
                                                 ("stop", "hold")]))
clf.ask("auth bypass in prod", Score(labels=["low", "med", "high", "crit"]))
```

---

## Measured results

Numbers from this machine (RTX 5070 Laptop, 8151 MiB VRAM, `nomic-embed-text`
on GPU), reproducible with `sentinel eval` and `sentinel bench`.

### Gate correctness — 76/76

| set | cases | prompted correctly | prompted needlessly | allowed unsafely |
|---|---|---|---|---|
| dev (`eval_report.EVAL_SET`) | 34 | 16 | 0 | **0** |
| held-out (`adversarial.ADVERSARIAL_SET`) | 42 | 26 | 0 | **0** |

The dev set was written first and the concept list built against it, so it
scores 34/34 and proves very little. The adversarial set was written afterwards
specifically to break it — paraphrases, and benign selectors that contain risky
words (`sort by newest`, `remove this filter`, `order history`). It found four
real bugs; all four are fixed and the failures are now regression tests.

Both sets were written by the same author as the code. **76/76 on self-authored
data is weak evidence.** Treat it as a floor, not a proof.

### Test suite & coverage

| metric | value |
|---|---|
| tests collected | 130 |
| test files | 4 |
| source lines | ~2.1k |
| line coverage (gate) | 75% |
| CI matrix | 2 OS × 4 Python versions |

Run: `python -m pytest --cov=sentinel` (exits 1 below 75%).

### Latency — 27.7 ms p95 against a 50 ms target

| stage | p50 | p95 | p99 | max |
|---|---|---|---|---|
| full decision, classifier advisory | 20.5 ms | **27.7 ms** | 37.2 ms | 57.8 ms |
| deterministic gate only (`--mode off`) | 0.08 ms | 0.13 ms | — | — |

Cold first call, model load included: 108 ms.

The daemon was never the bottleneck. Two client-side mistakes were:

- Building an `httpx.Client` per call cost ~250 ms of a ~940 ms decision.
- The legacy `/api/embeddings` endpoint carries ~300 ms of fixed overhead per
  request; the batch `/api/embed` endpoint costs ~19 ms per vector. A previous
  comment in this repo claimed the batch endpoint was unreliable — that was
  wrong, and batching alone was worth 18x.

The first benchmark I wrote was also wrong: it reused the same selectors each
round, so the classifier's own text cache answered every call and reported a
1 ms p50. The benchmark now appends a nonce per round to force genuine misses.
If you change the benchmark, keep doing that.

### The classifier does not work as a gate — and is not used as one

Measured on 34 labelled cases with the real embedder:

```
benign margin range   +0.0012 .. +0.1021
risky  margin range   +0.0006 .. +0.1997
margin AUC            0.5035
separable?            no, at any threshold
```

A coin flip. The top-2 similarity margin cannot tell a benign action from a
risky one, so the gate does not use it. Default is `ClassifierMode.ADVISORY`:
the classifier runs, records an observation for G5, reports its opinion in
`reason`, and the deterministic layer decides. `ClassifierMode.BLOCKING` exists
for after G5, when a calibrated classifier has earned the right to gate.

This was found by measurement, not assumed. The original design gated on
`min_confidence=0.12` and produced 11 prompts on 18 benign actions — pure
friction, no safety, because the number being thresholded was noise.

---

## How the gate decides

1. **Tool baseline** — an explicit table. `navigate`, `snapshot`, `extract`,
   `scroll`, `hover` are read-only. `type`, `select`, `press`, `click` change
   local state. A tool not in the table is refused, because an unfamiliar tool
   is not automatically safe.
2. **Risk concepts** — matched against the *action target* only, as groups of
   words that must all be present. `("place", "order")` catches "place your
   order", "place order" and "place the order" alike. Literal phrase lists were
   tried first and let through "place your order", "cancel my subscription" and
   "remove permanently" — all irreversible.
3. **Weak net** — a broader word list (`purge`, `terminate`, `revoke`) that
   escalates only to `outward`. This is the honest admission that the concept
   list does not generalise: high recall is cheap, a missed purchase is not.
4. **Transactional URL prefixes** — `/checkout`, `/pay`, `/account/delete`.
   Matched on path prefix, not substring, so a blog post about payment options
   is content rather than an endpoint.

### Two distinctions that carry a lot of weight

**A URL says what a page is about, not what a call does.** Navigating to
`/pricing` is a read no matter what the page discusses. Letting page topic drive
risk escalated every read-only navigation to a pricing or billing page.

**A query field's payload is a query, not a command.** Typing "delete account"
into a search box deletes nothing. Without this rule, every search for a risky
phrase prompted.

---

## Installation

```powershell
git clone https://github.com/Sachitt-AV-08/sentinel
cd sentinel
pip install -e .          # core has no dependencies
pip install -e ".[ollama]"  # adds httpx for the Ollama backend
```

Needs a local Ollama with `nomic-embed-text` for the semantic path:

```powershell
ollama pull nomic-embed-text
```

Without it, Sentinel falls back to a deterministic hashing embedder so tests
run anywhere. That backend is lexical: it matches character n-grams, so
"place your order" and "costs money" are unrelated to it. It is pinned by a test
(`test_hashing_embedder_cannot_do_semantics`) so nobody mistakes it for the real
thing. In advisory mode this costs little, because the deterministic layer is
what decides.

## CLI

```powershell
sentinel doctor                                    # which embedder is live
sentinel gate browse_click --args '{"selector":"buy now"}'   # exit 10 = needs a human
sentinel eval                                      # confusion matrix + separability
sentinel bench --rounds 5                          # latency, exits 1 if over target
sentinel ask "the site is down" --noul "This is an outage"
sentinel score "auth bypass in prod" --levels low,med,high,crit
```

`gate` exits 0 (may proceed) or 10 (needs a human), so it drops into a shell
conditional. `eval` exits 1 if anything unsafe was allowed. `bench` exits 1 if
p95 misses the target.

## Calibration (G5, not yet wired)

`Classifier.observe()` records every decision, and `fit("platt")` /
`fit("isotonic")` turn margins into probabilities. `fit()` refuses to run below
20 observations, and `Answer.calibrated` is `False` until it succeeds, so
downstream code never has to guess whether a probability means anything.

`expected_calibration_error()` is documented as blind to a globally inverted
calibration — a constant predictor on a 50/50 set scores ~0 however wrong it
is. `discrimination()` (AUC) is what catches inversion, and there is a test
pinning the blind spot so it cannot be mistaken for a bug.

## Tests

```powershell
python -m pytest --cov=sentinel -q     # 130 tests, 68% coverage gate
```

## Known limits

- 76 labelled cases, all self-authored. Precision and recall on real browsing
  traffic are unknown, and the concept list is English-only and will miss
  phrasings I did not think of.
- Only the 14 tools in `TOOL_BASELINE` are classified. Anything else is refused,
  which is safe but means adding a tool requires a decision about its risk.
- Real selectors are CSS and accessibility text, not "place your order". The
  concept matcher sees whatever orvima passes through, so its behaviour on live
  pages is untested.
- The weak word net is tuned against my own errors, so it is biased toward the
  phrasings I tend to use.

## License

MIT
