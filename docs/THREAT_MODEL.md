# Formal Threat Model (STRIDE Framework)

## 1. System Context & Data Classification

AI Wealth Manager processes sensitive personal, financial, and regulatory records. The system classifies data into four sensitivity tiers:

| Tier | Description | Examples | Protection at Rest | Protection in Transit |
|---|---|---|---|---|
| **Tier 1: Client PII** | Direct personally identifiable information | Email, phone number, date of birth, client notes | Fernet AES-128-CBC + HMAC-SHA256 (`EncryptedString`) | TLS 1.3 |
| **Tier 2: Financial Assets** | Portfolio, account balances, tax lots, orders | Net worth, cash, positions, cost basis, order quantities | PostgreSQL encryption, scoped tenant isolation | TLS 1.3 |
| **Tier 3: Mandate / Policy** | Versioned Investment Policy Statements (IPS) | Asset class limits, risk tiers, exclusions | Hash-chained audit records, versioned rows | TLS 1.3 |
| **Tier 4: System Secrets** | Authentication & signing keys | API keys, JWT secret, database credentials, LLM keys | Argon2/Bcrypt hashes, Secret Manager | Protected memory |

---

## 2. Trust Boundaries & Architecture

```
[ External User / Advisor Browser ]
               │
      (HTTPS / TLS 1.3)
               ▼
┌────────────────────────────────────────────────────────┐
│ TRUST BOUNDARY 1: API Gateway (server.py)             │
│ - JWT / API Key Authentication & Rate Limiting         │
│ - RFC 9457 Safe Error Sanitization                    │
│ - Tenancy Context Injection (Scoped Queries)          │
└───────────────────────┬────────────────────────────────┘
                        │
      (Internal RPC / Database Queue)
                        ▼
┌────────────────────────────────────────────────────────┐
│ TRUST BOUNDARY 2: Worker Engine (worker.py)            │
│ - Heartbeat Runner & Dead-Worker Recovery             │
│ - Wall-Clock & Daily Org LLM Spend Budgets            │
│ - LangGraph Orchestrator & Multi-Agent Pipeline       │
└───────────┬─────────────────────────────────┬──────────┘
            │                                 │
 (Fenced Prompts)                     (Deterministic Logic)
            ▼                                 ▼
┌───────────────────────┐         ┌──────────────────────────────┐
│ UNTRUSTED BOUNDARY 3: │         │ TRUST BOUNDARY 4:            │
│ LLM & News Providers  │         │ Deterministic Guardrails     │
│ - External Headlines  │         │ - Zero LLM in Decision Path  │
│ - Gemini LLM API      │         │ - Suitability Sizing & Caps  │
│ - Provider Fallbacks  │         │ - Tax-Awareness Wash-Sale    │
└───────────────────────┘         └──────────────┬───────────────┘
                                                 │
                                     (Encrypted Persistence)
                                                 ▼
┌────────────────────────────────────────────────────────────────┐
│ TRUST BOUNDARY 5: Persistence Layer                            │
│ - PostgreSQL: Tenant Rows + PII Field Encryption               │
│ - MySQL: Immutable Analytics Event Stream                     │
│ - Audit Trail: Hash-Chained Append-Only Log                   │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. STRIDE Threat Analysis & Mitigations

### 1. Spoofing (Identity & Authentication)
- **Threat:** Attacker impersonates an advisor, compliance officer, or another firm's API client.
- **Mitigations:**
  - JWT tokens signed with SHA256, carrying short-lived expirations (`ACCESS_TOKEN_TTL_MINUTES=15`) and per-user `token_version` for instant revocation.
  - API keys hashed with SHA256 before database comparison; raw secrets never logged or stored.
  - Zero-downtime key rotation with automatic grace periods.

### 2. Tampering (Data Integrity)
- **Threat:** Malicious actor modifies an approved trade, changes client risk tolerance, or alters regulatory audit trails.
- **Mitigations:**
  - IPS policies are versioned immutable records. Runs bind to specific policy versions.
  - Audit trail uses a cryptographic hash-chain (`prev_hash || row_data -> sha256`). Any back-dating or row deletion invalidates subsequent chain hashes.
  - Trade approvals require explicit `compliance` role separate from the proposing `advisor`.

### 3. Repudiation (Accountability)
- **Threat:** Advisor denies initiating an unsuitable recommendation or altering an exclusion list.
- **Mitigations:**
  - Correlation ID is propagated through HTTP headers, database job records, and background agent execution contexts.
  - Hash-chained audit log records `actor_id`, `actor_role`, `org_id`, `ip_address`, and deterministic snapshot parameters.

### 4. Information Disclosure (Confidentiality)
- **Threat:** SQL injection or database dump exposes client PII; API error dumps stack traces containing holdings.
- **Mitigations:**
  - Field-level encryption on client PII at rest (`email`, `phone`, `date_of_birth`, `notes`) using distinct Fernet keys.
  - Strict RFC 9457 application error model strips stack traces, file paths, and echoes of sensitive input values.
  - Cross-tenant lookups return HTTP 404 rather than 403, preventing resource enumeration.

### 5. Denial of Service (Availability)
- **Threat:** Tenant floods the background queue with long-running analysis runs; runaway LLM prompt drains the budget.
- **Mitigations:**
  - Daily per-organization spend limits (`services/spend.py`) and per-run wall-clock budgets (`services/deadline.py`).
  - Database connection pool timeout guards (`statement_timeout=10s`, `lock_timeout=5s`).
  - Heartbeat-driven worker reclamation restarts orphaned jobs if a worker node crashes.

### 6. Elevation of Privilege & Prompt Injection
- **Threat:** Attacker embeds prompt-injection payload in client notes or financial news headline to bypass suitability limits.
- **Mitigations:**
  - **Zero LLM in the control path:** All suitability screens, maximum position caps, sector caps, and wash-sale prunings are implemented in pure, deterministic Python (`agents/suitability.py`, `agents/tax_awareness.py`).
  - Untrusted data (news headlines, user notes) is sanitized and enclosed in XML boundary tags (`<untrusted_data>`) with illegal tag escaping.
  - Adversarial red-team test suite continuously verifies prompt injection resilience.
