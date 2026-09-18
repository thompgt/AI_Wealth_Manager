"""Test verification of the load and soak test harness.

Covers Production Readiness Item 15:
- Load harness execution
- Percentile latency calculation
- Error rate and throughput metrics
"""

from scripts.load_harness import BenchmarkResult, run_benchmark


def test_benchmark_result_percentiles_calculation():
    res = BenchmarkResult(
        total_requests=100,
        successful_requests=100,
        failed_requests=0,
        duration_seconds=1.0,
        latencies_ms=[float(i) for i in range(1, 101)],
    )
    assert res.requests_per_second == 100.0
    assert res.error_rate == 0.0
    # p50 should be ~50.5
    assert 50.0 <= res.percentile(50) <= 51.0
    # p90 should be ~90.1
    assert 89.0 <= res.percentile(90) <= 91.0
    # p99 should be ~99.0
    assert 98.0 <= res.percentile(99) <= 100.0

    summary = res.summary()
    assert summary["total_requests"] == 100
    assert summary["failed_requests"] == 0
    assert summary["p50_latency_ms"] > 0


def test_benchmark_harness_smoke_execution():
    """Run a small benchmark cycle through the harness and verify clean metrics."""
    res = run_benchmark(concurrency=2, iterations=10)
    assert res.total_requests == 50  # 10 iterations * 5 operations
    assert res.successful_requests == 50
    assert res.failed_requests == 0
    assert res.error_rate == 0.0
    assert res.requests_per_second > 0.0
    assert res.duration_seconds > 0.0
    assert len(res.latencies_ms) == 50
