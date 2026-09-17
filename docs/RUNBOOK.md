# Production Operations & Incident Response Runbook

This runbook provides step-by-step operational procedures for deploying, maintaining, backing up, restoring, and troubleshooting the AI Wealth Manager platform in production environments.

---

## 1. System Architecture & Component Overview

| Component | Image / Process | Port | Health Check | Dependencies |
|---|---|---|---|---|
| **API Gateway / App** | `server.py` (Uvicorn) | 8000 / 8001 | `GET /ready`, `GET /live` | PostgreSQL, Checkpointer |
| **Background Worker** | `worker.py` | N/A | Process heartbeat in DB | PostgreSQL, LLM Provider, Market Data |
| **Relational Database** | PostgreSQL 16 | 5432 | `pg_isready` | Persistent Volume |
| **Analytical Store** | MySQL 8.0 | 3306 | `mysqladmin ping` | Persistent Volume |
| **Metrics Collector** | Prometheus | 9090 | `/-/healthy` | API Gateway (`/metrics`) |
| **Monitoring Dashboard**| Grafana | 3000 | `/api/health` | Prometheus |

---

## 2. Deployment Procedures

### Standard Rolling Deployment (Zero-Downtime)

1. **Pre-flight Checks:**
   ```bash
   # Ensure git clean and dependencies audited
   git status
   pip-audit -r requirements.txt --desc on
   pytest -q
   ```

2. **Schema Migration Pre-Run:**
   Execute forward migrations *before* routing traffic to new application containers:
   ```bash
   alembic upgrade head
   alembic check
   ```
   *Note: All migrations must be backward-compatible (expand-contract pattern). Never drop columns in the same release that stops reading them.*

3. **Deploy Container Updates:**
   ```bash
   # Pull latest verified images
   docker compose pull api worker
   
   # Rolling restart of worker processes first
   docker compose up -d --no-deps worker
   
   # Restart API containers with zero-downtime health verification
   docker compose up -d --no-deps --scale api=2 api
   ```

4. **Post-Deployment Verification:**
   ```bash
   # Verify liveness and readiness
   curl -f http://localhost:8000/live
   curl -f http://localhost:8000/ready
   
   # Check active Prometheus alerts
   curl -s http://localhost:9090/api/v1/alerts | jq .
   ```

---

## 3. Rollback Procedures

If elevated error rates, alert rule firings (e.g. `HighSilentDegradationRate`), or readiness failures occur post-deploy:

1. **Immediate Binary / Container Rollback:**
   ```bash
   # Rollback to the previous stable release tag
   export PREVIOUS_IMAGE_TAG="v2.0.0-stable"
   docker compose up -d --no-deps api worker
   ```

2. **Database Rollback (If Required):**
   *Caution: Only downgrade migrations if the new schema introduced fatal blockers and the rollback migration is verified non-destructive.*
   ```bash
   # Inspect current migration revision
   alembic current
   
   # Revert exactly one revision
   alembic downgrade -1
   ```

3. **Reclaim Orphaned Jobs:**
   A rolled-back worker deployment may leave jobs in `running` state. Clean them up:
   ```bash
   python -c "from services.jobs import reclaim_orphaned_jobs; from db import SessionLocal; db = SessionLocal(); print(reclaim_orphaned_jobs(db, max_age_seconds=120)); db.close()"
   ```

---

## 4. Backup, Retention & Disaster Recovery

### PostgreSQL Operational Backups
- **Daily Full Logical Backup:**
  ```bash
  pg_dump -Fc -h "$DB_HOST" -U "$DB_USER" -d "$DB_NAME" > "/backups/pg_dump_$(date +%Y%m%d_%H%M%S).dump"
  ```
- **Continuous WAL Archiving:**
  WAL segments are streamed to secure cloud storage (AWS S3 or GCS) with bucket immutability (Object Lock) configured for 5 years per SEC Rule 17a-4.

### MySQL Analytical Store Backup
- **Daily Consistent Snapshot:**
  ```bash
  mysqldump -h "$MYSQL_HOST" -u "$MYSQL_USER" -p"$MYSQL_PASSWORD" \
    --single-transaction --quick --databases wealth_analytics > "/backups/mysql_analytics_$(date +%Y%m%d).sql"
  ```

### Encryption Key Management (`FIELD_ENCRYPTION_KEY`)
- Client PII fields (`email`, `phone`, `date_of_birth`, `notes`) are encrypted at rest using AES-128-CBC + HMAC-SHA256 via `services/encryption.py`.
- **Key Storage:** The master key `FIELD_ENCRYPTION_KEY` must be stored in AWS Secrets Manager or HashiCorp Vault.
- **Key Backup:** Store an offline, split-custody paper copy in the corporate vault. Losing this key renders client PII permanently unrecoverable.

---

## 5. Incident Response Playbooks

### Playbook A: Dead Worker & Stalled Job Queue
- **Symptoms:** Alert `JobQueueBacklogStalled` fires; `job_queue_depth` increases while `analysis_runs_finished_total` flatlines.
- **Triage Steps:**
  1. Inspect worker container logs:
     ```bash
     docker compose logs --tail=100 -f worker
     ```
  2. Verify if worker process is deadlocked or crashed on memory exhaustion:
     ```bash
     docker stats worker
     ```
  3. Restart worker container:
     ```bash
     docker compose restart worker
     ```
  4. The worker automatically invokes `reclaim_orphaned_jobs()` on startup, re-queuing any tasks abandoned by dead process IDs.

### Playbook B: LLM Provider Outage or Quota Exhaustion
- **Symptoms:** Alert `LLMQuotaExhaustedOrProviderDown` or `HighSilentDegradationRate` firing; LLM calls returning fallback.
- **Triage Steps:**
  1. Inspect fallback metric labels:
     ```bash
     curl -s http://localhost:8000/metrics | grep llm_calls_total
     ```
  2. If quota exceeded, verify if daily org spend cap was hit via `services/spend.py`.
  3. The system gracefully continues generating recommendations via deterministic quantitative heuristics (`agents/suitability.py` and `agents/rebalance.py`). No trade execution is corrupted.
  4. If upstream API key expired, rotate `GEMINI_API_KEY` in secrets manager and reload configuration:
     ```bash
     docker compose exec api kill -HUP 1
     ```

### Playbook C: Market Data Provider Circuit Breaker Open
- **Symptoms:** Alert `MarketDataCircuitBreakerOpen` firing for provider `yfinance` or `polygon`.
- **Triage Steps:**
  1. Check provider health:
     ```bash
     curl -s http://localhost:8000/metrics | grep market_data_circuit_open
     ```
  2. Circuit breaker auto-resets after 300 seconds of quiet time. To force a reset or switch fallback provider, verify `POLYGON_API_KEY` or `TIINGO_API_KEY` in `.env`.
  3. If quotes are flagged stale (`stale_quotes_total > 0`), investigate network proxy or egress firewalls.

### Playbook D: Audit Trail Verification Failure
- **Symptoms:** Periodic integrity job or regulator audit detects a broken hash chain in `audit_events`.
- **Triage Steps:**
  1. Run the audit integrity verification script:
     ```bash
     python -c "from services.audit import verify_chain; from db import SessionLocal; db = SessionLocal(); valid, idx = verify_chain(db); print(f'Chain Valid: {valid}, Broken Index: {idx}'); db.close()"
     ```
  2. If broken, immediately preserve database WAL logs and flag the incident to Compliance.
  3. Audit records are append-only; an integrity mismatch indicates manual SQL update or unauthorized tampering.
