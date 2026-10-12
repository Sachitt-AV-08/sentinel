"""Pluggable embedding backends.

The classifier only needs `text -> vector`. Production uses Ollama's
`nomic-embed-text`; tests and offline use use `HashingEmbedder`, which is
deterministic, dependency-free and good enough to exercise every code path.

The hashing backend is not a toy gradient: it is a bag-of-character-ngrams
projection, which means it still separates lexically similar texts and still
reproduces exactly. It is simply weaker than a trained model on paraphrase,
and `describe()` reports which one is live so a caller never has to guess
whether it is looking at real semantic vectors.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

DEFAULT_MODEL = "nomic-embed-text"
DEFAULT_DIM = 768


class Embedder(Protocol):
    """Anything that turns text into a fixed-length vector."""

    dim: int
    name: str

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        ...


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity. Assumes unit vectors, but does not require them."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    num = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


_TOKEN = re.compile(r"[a-z0-9]+")


def _ngrams(text: str) -> Counter:
    """Character 3-grams plus whole words.

    Word features give the vector some semantics; character n-grams keep it
    robust to the typos and the CSS selectors that show up in tool arguments.
    """
    lowered = text.lower()
    feats: Counter = Counter(_TOKEN.findall(lowered))
    padded = f" {lowered} "
    for i in range(len(padded) - 2):
        feats[padded[i : i + 3]] += 1
    return feats


class HashingEmbedder:
    """Deterministic hashing embedder. No network, no model, no surprises."""

    name = "hashing-v1"

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        if dim < 8:
            raise ValueError("dim too small to separate anything")
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_one(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for feature, count in _ngrams(text).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[idx] += sign * (1.0 + math.log(count))
        return _normalise(vec)


class OllamaEmbedder:
    """Embeds via a local Ollama daemon. No API key, no cloud.

    Uses the batch `/api/embed` endpoint, which is worth roughly 18x over the
    legacy single-prompt `/api/embeddings`: measured 19 ms per vector batched
    against 337 ms one-at-a-time, because the legacy path pays ~300 ms of fixed
    per-request overhead. If the batch endpoint is missing (old daemon) it falls
    back to the legacy one rather than failing, and `name` records which is live
    so a slow `name` is diagnosable instead of mysterious.

    `keep_alive` is sent explicitly. Without it a daemon under memory pressure
    can unload and reload the model between decisions, which is what turns a
    30 ms gate into a multi-second one.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://127.0.0.1:11434",
        timeout: float = 30.0,
        dim: int = DEFAULT_DIM,
        keep_alive: str = "30m",
        prefer_batch: bool = True,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.dim = dim
        self.keep_alive = keep_alive
        self.prefer_batch = prefer_batch
        self._batch_ok = prefer_batch
        self.name = f"ollama:{model}"
        # One HTTP client per thread, created lazily and kept for the process.
        # Building an httpx.Client per call cost ~250 ms of the ~940 ms a
        # decision took; reusing one brings a single-text embed to ~36 ms. On a
        # per-decision hot path that is the whole difference between missing the
        # 50 ms gate by 19x and meeting it. httpx clients are not thread-safe,
        # hence per-thread rather than a single shared one.
        self._local = threading.local()

    def _get_client(self):
        client = getattr(self._local, "client", None)
        if client is None:
            import httpx

            client = httpx.Client(timeout=self.timeout)
            self._local.client = client
        return client

    def close(self) -> None:
        """Release this thread's HTTP connection. Safe to call more than once."""
        client = getattr(self._local, "client", None)
        if client is not None:
            client.close()
            self._local.client = None

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "OllamaEmbedder needs httpx; install it or use HashingEmbedder"
            ) from exc

        client = self._get_client()
        if self._batch_ok:
            try:
                return self._embed_batch(client, texts)
            except (httpx.HTTPError, KeyError, ValueError):
                # Old daemon without /api/embed. Downgrade once and retry.
                self._batch_ok = False
                self.name = f"ollama:{self.model}/legacy"
        return self._embed_legacy(client, texts)

    def _embed_batch(self, client, texts: Sequence[str]) -> list[list[float]]:
        resp = client.post(
            f"{self.host}/api/embed",
            json={"model": self.model, "input": list(texts), "keep_alive": self.keep_alive},
        )
        resp.raise_for_status()
        vecs = resp.json()["embeddings"]
        out = [[float(v) for v in vec] for vec in vecs]
        if len(out) != len(texts):
            raise ValueError(f"daemon returned {len(out)} vectors for {len(texts)} inputs")
        self.dim = len(out[0])
        return out

    def _embed_legacy(self, client, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            resp = client.post(
                f"{self.host}/api/embeddings",
                json={"model": self.model, "prompt": text, "keep_alive": self.keep_alive},
            )
            resp.raise_for_status()
            out.append([float(v) for v in resp.json()["embedding"]])
        self.dim = len(out[0])
        return out

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class UnavailableEmbedder:
    """Stands in when the real backend cannot be reached.

    This exists so a missing daemon degrades into a *visibly* weaker mode
    instead of an import error three frames deep. `degraded` is surfaced to the
    caller because a zero-shot hashing classifier should not be trusted for a
    high-risk gate without saying so.
    """

    name = "unavailable"

    def __init__(self, reason: str, fallback: Embedder | None = None) -> None:
        self.reason = reason
        self._fallback = fallback or HashingEmbedder()
        self.dim = self._fallback.dim
        self.name = f"unavailable({self._fallback.name})"

    @property
    def degraded(self) -> bool:
        return True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self._fallback.embed(texts)

    def embed_one(self, text: str) -> list[float]:
        return self._fallback.embed_one(text)


def default_embedder(prefer_ollama: bool = True) -> Embedder:
    """Best embedder that actually works here.

    Tries Ollama, falls back to hashing, and reports which it picked rather
    than pretending the fallback is equivalent.
    """
    if prefer_ollama:
        try:
            emb = OllamaEmbedder()
            emb.embed_one("connectivity probe")
            return emb
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            return UnavailableEmbedder(f"{type(exc).__name__}: {exc}")
    return HashingEmbedder()


def describe(embedder: Embedder) -> str:
    """One line a CLI or log can print, including any degradation."""
    if isinstance(embedder, UnavailableEmbedder):
        return f"{embedder.name} - DEGRADED ({embedder.reason})"
    return f"{embedder.name} (dim={embedder.dim})"
