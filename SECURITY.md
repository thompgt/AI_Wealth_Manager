# Security Policy

## 1. Supported Versions

Security updates and patches are applied to the following release branches:

| Version | Supported | Notes |
|---|---|---|
| `2.0.x` | :white_check_mark: | Current production release |
| `1.x.x` | :x: | Deprecated legacy architecture |

---

## 2. Reporting a Vulnerability

We take the security of wealth management data, client PII, and financial audit integrity seriously. If you identify a potential security vulnerability in AI Wealth Manager, please report it responsibly:

- **Email:** `security@wealthmanager.example` (or through our coordinated disclosure portal).
- **Encryption:** Use our GPG public key (Key ID: `0xDEADBEEFCAFE1234`) for sensitive attachments.
- **Please DO NOT** open public GitHub issues for security vulnerabilities.

### What to Include in Your Report
1. Detailed description of the vulnerability and its potential security impact.
2. Step-by-step reproduction instructions or proof-of-concept script.
3. Affected components (e.g. API authentication, prompt injection, encryption at rest).
4. Any potential mitigations you have identified.

### Response Timelines & SLA
- **Initial Acknowledgment:** Within 24 hours.
- **Triage & Severity Assessment:** Within 72 hours.
- **Patch Development & Release:** Critical vulnerabilities are patched within 7 business days.
- **Public Disclosure:** Coordinated after all affected production systems have applied the remediation.

---

## 3. Cryptographic Standards & Design Principles

1. **Zero LLM in the Control Path:** Machine learning models are treated as untrusted research generators. Guardrails, suitability limits, trade sizing, and wash-sale rules are 100% deterministic code.
2. **Field-Level Encryption at Rest:** Sensitive client PII is encrypted with AES-128-CBC + HMAC-SHA256 (Fernet) prior to persistence.
3. **Immutability of Audit Trails:** The audit log is cryptographically chained via SHA-256 hashes, providing mathematical tamper-evidence.
4. **Principle of Least Privilege:** Tenancy is enforced by construction using scoped database sessions; RBAC strictly separates advisors from compliance officers.
