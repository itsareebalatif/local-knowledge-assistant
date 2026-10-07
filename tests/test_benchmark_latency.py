"""Smoke test for the latency benchmark tool itself — small scale, fast,
just proves the insert/query/percentile machinery is correct. The actual
NFR-2 claim (<0.4s at 250,000 chunks) is validated by running this for real
at full scale via `kengine benchmark-latency`, not as part of the default
test suite — a 250k-vector insert has no business running on every `pytest`
invocation.
"""

from __future__ import annotations

from app.evaluation.benchmark_latency import LatencyReport, _percentile, benchmark_vector_query_latency


def test_percentile_never_exceeds_the_actual_max_with_few_samples():
    # Regression test: statistics.quantiles(n=100) can extrapolate above the
    # observed max when there are far fewer than 100 samples — _percentile
    # must not, since it only ever returns an actual observed value.
    values = sorted([3.47, 2.58, 2.49, 2.98, 14.27, 21.06, 3.20, 6.15, 4.51, 4.28])
    assert _percentile(values, 50) <= _percentile(values, 95) <= _percentile(values, 99) <= max(values)


def test_percentile_handles_empty_list():
    assert _percentile([], 95) == 0.0


def test_percentile_handles_single_value():
    assert _percentile([7.0], 99) == 7.0


def test_benchmark_runs_and_reports_sane_percentiles(tmp_path):
    report = benchmark_vector_query_latency(tmp_path, n_chunks=200, n_queries=10, batch_size=50)

    assert report.n_chunks == 200
    assert len(report.latencies_ms) == 10
    assert report.insert_seconds > 0
    assert 0 <= report.p50_ms <= report.p95_ms <= report.p99_ms <= report.max_ms
    assert report.mean_ms > 0


def test_benchmark_respects_batch_size_smaller_than_total(tmp_path):
    # n_chunks not evenly divisible by batch_size — exercises the partial
    # final-batch path in the insert loop.
    report = benchmark_vector_query_latency(tmp_path, n_chunks=130, n_queries=3, batch_size=50)
    assert report.n_chunks == 130
