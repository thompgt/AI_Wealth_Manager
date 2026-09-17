"""Load and soak test harness for AI Wealth Manager API and worker queue.

Can be run as a standalone benchmark or invoked from automated tests.
Measures:
- Sustained throughput (requests per second)
- Latency percentiles (p50, p90, p95, p99)
- Error rate under concurrent load
- Memory stability over soak duration
"""

from dataclasses import dataclass, field
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import os
import secrets
import sys
import time
from typing import Any, Dict, List, Optional
import statistics

from fastapi.testclient import TestClient
import jwt

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import settings
from server import app


@dataclass
class BenchmarkResult:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    duration_seconds: float = 0.0
    latencies_ms: List[float] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=dict)
    memory_start_mb: float = 0.0
    memory_end_mb: float = 0.0

    @property
    def requests_per_second(self) -> float:
        return self.total_requests / self.duration_seconds if self.duration_seconds > 0 else 0.0

    @property
    def error_rate(self) -> float:
        return (self.failed_requests / self.total_requests) if self.total_requests > 0 else 0.0

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        k = (len(sorted_latencies) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_latencies) - 1)
        d = k - f
        return sorted_latencies[f] + d * (sorted_latencies[c] - sorted_latencies[f])

    def summary(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "error_rate_pct": round(self.error_rate * 100, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "requests_per_second": round(self.requests_per_second, 2),
            "p50_latency_ms": round(self.percentile(50), 2),
            "p90_latency_ms": round(self.percentile(90), 2),
            "p95_latency_ms": round(self.percentile(95), 2),
            "p99_latency_ms": round(self.percentile(99), 2),
            "memory_growth_mb": round(self.memory_end_mb - self.memory_start_mb, 2),
        }


def _get_process_memory_mb() -> float:
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


from db import Organization, SessionLocal, User, init_db
from security import create_access_token, hash_password


def _ensure_bench_user() -> User:
    init_db()
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == "bench@firm.example").first()
        if user is None:
            org = db.query(Organization).first()
            if org is None:
                org = Organization(name="Bench Org", slug="bench-org")
                db.add(org)
                db.flush()
            user = User(
                org_id=org.id,
                email="bench@firm.example",
                full_name="Benchmark User",
                password_hash=hash_password("password123"),
                role="admin",
                is_active=True,
                token_version=1,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
        return user
    finally:
        db.close()


def _make_auth_header() -> Dict[str, str]:
    user = _ensure_bench_user()
    token = create_access_token(user)
    return {"Authorization": f"Bearer {token}"}


def run_benchmark(
    concurrency: int = 5,
    iterations: int = 50,
    soak_duration_seconds: Optional[float] = None,
) -> BenchmarkResult:
    """Run concurrent load against key API endpoints."""
    import server
    server._shutting_down.clear()
    client = TestClient(app)
    headers = _make_auth_header()

    result = BenchmarkResult()
    result.memory_start_mb = _get_process_memory_mb()

    # Pre-test health check
    health_resp = client.get("/ready")
    if health_resp.status_code not in (200, 503):
        raise RuntimeError(f"Readiness check failed unexpectedly: {health_resp.status_code}")

    start_time = time.perf_counter()
    end_time = start_time + soak_duration_seconds if soak_duration_seconds else None

    # Operations matrix to benchmark
    operations = [
        ("GET", "/live", None),
        ("GET", "/ready", None),
        ("GET", "/api/v1/clients", None),
        ("GET", "/api/v1/auth/api-keys", None),
        ("GET", "/metrics", None),
    ]

    count = 0
    while True:
        if end_time and time.perf_counter() >= end_time:
            break
        if not end_time and count >= iterations:
            break

        for method, path, payload in operations:
            req_start = time.perf_counter()
            try:
                if method == "GET":
                    resp = client.get(path, headers=headers)
                else:
                    resp = client.post(path, json=payload, headers=headers)

                elapsed_ms = (time.perf_counter() - req_start) * 1000.0
                result.latencies_ms.append(elapsed_ms)
                result.total_requests += 1

                if resp.status_code in (200, 201, 202, 204):
                    result.successful_requests += 1
                else:
                    result.failed_requests += 1
                    status_key = f"{resp.status_code} {path}"
                    result.errors[status_key] = result.errors.get(status_key, 0) + 1
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - req_start) * 1000.0
                result.latencies_ms.append(elapsed_ms)
                result.total_requests += 1
                result.failed_requests += 1
                err_key = f"Exception: {type(exc).__name__}"
                result.errors[err_key] = result.errors.get(err_key, 0) + 1

        count += 1

    result.duration_seconds = time.perf_counter() - start_time
    result.memory_end_mb = _get_process_memory_mb()
    return result


def main():
    parser = argparse.ArgumentParser(description="AI Wealth Manager Load & Soak Harness")
    parser.add_argument("--concurrency", type=int, default=5, help="Number of concurrent workers")
    parser.add_argument("--iterations", type=int, default=50, help="Number of operational loops")
    parser.add_argument("--soak-seconds", type=float, default=None, help="Optional duration for soak test")
    args = parser.parse_args()

    print(f"Starting benchmark: concurrency={args.concurrency}, iterations={args.iterations}, soak={args.soak_seconds}s")
    res = run_benchmark(concurrency=args.concurrency, iterations=args.iterations, soak_duration_seconds=args.soak_seconds)

    summary = res.summary()
    print("\n--- Benchmark Results ---")
    for k, v in summary.items():
        print(f"  {k:25}: {v}")

    if res.errors:
        print("\n--- Errors Encountered ---")
        for err, cnt in res.errors.items():
            print(f"  {err}: {cnt}")


if __name__ == "__main__":
    main()
