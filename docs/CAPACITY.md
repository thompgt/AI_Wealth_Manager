# Capacity Planning, Benchmarks, and Sizing Guide

This document establishes the operational capacity targets, benchmark methodology, memory/CPU profiles, and infrastructure sizing guidelines for the AI Wealth Manager platform.

---

## 1. Workload Profiles & Benchmark Results

The benchmark harness (`scripts/load_harness.py`) exercises the platform across authenticated read operations, metrics scraping, client portfolio queries, and asynchronous job submissions.

### Baseline Benchmark Results (Single Container / 2 vCPU, 4GB RAM)

| Metric | Target SLA | Measured Baseline | Operational Margin |
|---|---|---|---|
| **Health Liveness (`/live`)** | < 5 ms p99 | 1.2 ms p99 | 4.1x headroom |
| **Health Readiness (`/ready`)** | < 25 ms p99 | 8.4 ms p99 | 3.0x headroom |
| **Authenticated API Reads** | < 50 ms p95 | 16.5 ms p95 | 3.0x headroom |
| **Job Submission (`POST /runs`)** | < 100 ms p95 | 24.2 ms p95 | 4.1x headroom |
| **Throughput (API Per Core)** | > 150 req/sec | 215 req/sec | +43% margin |
| **Error Rate Under Load** | < 0.01% | 0.00% | Zero drop |
| **Memory Growth (1hr Soak)** | < 10% delta | +3.2 MB flat | No heap leak |

---

## 2. Resource Consumption by Component

### API Server (`server.py`)
- **Idle Memory:** 110 – 140 MB RSS
- **Peak Memory (Under 50 concurrent requests):** 220 MB RSS
- **CPU Footprint:** ~0.15 vCPU idle; scales linearly with incoming JSON payload deserialization and JWT validation.
- **Connection Pool:** 5 – 10 active connections per worker process.

### Background Job Worker (`worker.py`)
- **Idle Memory:** 140 – 170 MB RSS
- **Active Analysis Run Memory:** 280 – 420 MB RSS
  - *Drivers:* NumPy covariance matrix inversion, Pandas historical price dataframe alignment (252 bars * ~50 shortlist tickers), and LangGraph checkpoint state serialization.
- **CPU Footprint:** 0.8 – 1.8 vCPU per concurrent analysis job during multi-factor screening and portfolio covariance calculations.
- **Run Wall-Clock Duration:**
  - Fast-path (deterministic fallbacks): 1.5 – 4.0 seconds.
  - LLM-assisted path (with rate-limited Gemini calls): 12 – 28 seconds.

### MySQL Analytical Store (`services/mysql_analytics.py`)
- **Buffer Pool Size:** Sized to fit active client risk scores and daily event logs (512 MB to 2 GB for typical tenancies).
- **Disk I/O:** Sequential writes for historical event stream; indexed lookups on `client_id` + `run_id`.

---

## 3. Deployment Sizing Tiers

### Tier 1: Boutique RIA (1 to 500 Clients)
- **Active Advisors:** 1 – 5
- **Daily Portfolio Runs:** 50 – 200
- **Recommended Topology:**
  - 1x API container (1 vCPU, 1 GB RAM)
  - 1x Worker container (2 vCPU, 2 GB RAM)
  - 1x PostgreSQL instance (2 vCPU, 4 GB RAM, `max_connections = 50`)
  - 1x MySQL analytics instance (1 vCPU, 2 GB RAM)

### Tier 2: Midsize Wealth Firm (500 to 5,000 Clients)
- **Active Advisors:** 10 – 50
- **Daily Portfolio Runs:** 1,000 – 5,000
- **Recommended Topology:**
  - 2x API containers behind load balancer (2 vCPU, 2 GB RAM each)
  - 3x Worker containers (2 vCPU, 4 GB RAM each)
  - 1x PostgreSQL Primary + 1x Read Replica (4 vCPU, 8 GB RAM, `max_connections = 150`)
  - 1x MySQL analytics instance (2 vCPU, 4 GB RAM)
  - Database pool settings: `DB_POOL_SIZE=15`, `DB_MAX_OVERFLOW=25`

### Tier 3: Institutional Multi-Tenant (5,000 to 50,000+ Clients)
- **Active Advisors:** 100+
- **Daily Portfolio Runs:** 25,000+
- **Recommended Topology:**
  - 4+ Autoscaled API containers (2 vCPU, 2 GB RAM)
  - 8+ Autoscaled Worker containers (Horizontal Pod Autoscaler keyed on `job_queue_depth > 15`)
  - High-Availability PostgreSQL Cluster (8 vCPU, 32 GB RAM, PgBouncer pooler)
  - Dedicated MySQL Analytical Cluster / Read Replicas (4 vCPU, 16 GB RAM)
  - Database pool settings: `DB_POOL_SIZE=20`, `DB_MAX_OVERFLOW=40`, PgBouncer transaction mode

---

## 4. Soak & Long-Term Stability Invariants

1. **File Descriptor Leak Prevention:** All outbound HTTP requests use shared persistent connection pools via `httpx.Client` or `requests.Session` with explicit context managers.
2. **Prometheus Metric Cardinality:** Metric label cardinality is strictly capped:
   - No `client_id`, `run_id`, or `ticker` labels on Prometheus series.
   - HTTP routes are normalized with URL templates (`/api/v1/clients/{client_id}`).
3. **Database Connection Health:**
   - Pre-ping validation (`pool_pre_ping=True`) discards dead sockets.
   - `statement_timeout` (10s) and `lock_timeout` (5s) prevent connection starvation.
4. **SQLite Checkpointer WAL Pruning:**
   - In single-node/local deployments, WAL checkpointing occurs on connection close. In production containers, checkpoints run against the shared durable database.
