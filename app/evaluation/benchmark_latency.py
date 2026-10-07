###
from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.embeddings.vector_store import ChromaVectorStore

EMBEDDING_DIM = 768  

def _percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile: always returns an actual observed value, so
    p50 <= p95 <= p99 <= max always holds regardless of sample size.

    statistics.quantiles() was tried first and rejected: its default
    interpolation method can extrapolate ABOVE the observed max when the
    requested quantile count (e.g. n=100 for p99) is large relative to the
    number of samples — exactly the case here with n_queries typically in
    the tens, not thousands.
    """
    if not sorted_values:
        return 0.0
    rank = max(0, min(len(sorted_values) - 1, math.ceil(pct / 100 * len(sorted_values)) - 1))
    return sorted_values[rank]


@dataclass
class LatencyReport:
    n_chunks: int
    n_queries: int
    insert_seconds: float
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.latencies_ms)

    @property
    def p50_ms(self) -> float:
        return _percentile(sorted(self.latencies_ms), 50)

    @property
    def p95_ms(self) -> float:
        return _percentile(sorted(self.latencies_ms), 95)

    @property
    def p99_ms(self) -> float:
        return _percentile(sorted(self.latencies_ms), 99)

    @property
    def max_ms(self) -> float:
        return max(self.latencies_ms)


def _random_vector(dim: int, rng: random.Random) -> list[float]:
    return [rng.uniform(-1.0, 1.0) for _ in range(dim)]


def benchmark_vector_query_latency(
    store_path: str | Path,
    n_chunks: int = 250_000,
    n_queries: int = 50,
    dim: int = EMBEDDING_DIM,
    batch_size: int = 5000,
    top_k: int = 5,
    seed: int = 42,
) -> LatencyReport:
    rng = random.Random(seed)
    store = ChromaVectorStore(path=str(store_path), collection_name="latency_benchmark")

    insert_start = time.perf_counter()
    inserted = 0
    while inserted < n_chunks:
        batch_n = min(batch_size, n_chunks - inserted)
        ids = [f"chunk-{inserted + i}" for i in range(batch_n)]
        embeddings = [_random_vector(dim, rng) for _ in range(batch_n)]
        documents = [f"synthetic chunk {inserted + i}" for i in range(batch_n)]
        metadatas = [{"doc_id": 1, "chunk_id": inserted + i} for i in range(batch_n)]
        store.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        inserted += batch_n
    insert_seconds = time.perf_counter() - insert_start

    latencies_ms = []
    for _ in range(n_queries):
        query_vector = _random_vector(dim, rng)
        start = time.perf_counter()
        store.query(query_vector, top_k=top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    return LatencyReport(n_chunks=n_chunks, n_queries=n_queries, insert_seconds=insert_seconds, latencies_ms=latencies_ms)
