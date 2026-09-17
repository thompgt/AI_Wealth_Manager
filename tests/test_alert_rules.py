"""Tests for Prometheus alert rules.

Verifies Production Readiness Item 14:
- Alert rules for the silent failure modes that return HTTP 200:
  * Silent agent degradation
  * LLM provider exhaustion/fallback
  * Budget headroom exhaustion
  * Market data circuit breaker open
  * Stale quote usage in sizing
  * Guardrail block spikes
  * Job queue backlog stalling
"""

from pathlib import Path
import re
import yaml
import pytest

import metrics


def test_prometheus_configuration_and_alert_rule_files():
    prom_cfg_path = Path("monitoring/prometheus/prometheus.yml")
    assert prom_cfg_path.exists(), "prometheus.yml must exist"

    with open(prom_cfg_path, "r", encoding="utf-8") as f:
        prom_cfg = yaml.safe_load(f)

    assert "rule_files" in prom_cfg
    assert "alerts.yml" in prom_cfg["rule_files"]


def test_alerts_yaml_syntax_and_structure():
    alerts_path = Path("monitoring/prometheus/alerts.yml")
    assert alerts_path.exists(), "alerts.yml must exist"

    with open(alerts_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    assert "groups" in content
    assert len(content["groups"]) > 0

    group = content["groups"][0]
    assert group.get("name") == "silent_degradation_alerts"
    assert "rules" in group
    assert len(group["rules"]) >= 6

    alert_names = set()
    for rule in group["rules"]:
        assert "alert" in rule
        name = rule["alert"]
        alert_names.add(name)

        assert "expr" in rule, f"Rule {name} must have an expr"
        assert "labels" in rule, f"Rule {name} must have labels"
        assert "severity" in rule["labels"]
        assert rule["labels"]["severity"] in ("warning", "critical")

        assert "annotations" in rule, f"Rule {name} must have annotations"
        assert "summary" in rule["annotations"]
        assert "description" in rule["annotations"]

    expected_alerts = {
        "HighSilentDegradationRate",
        "LLMQuotaExhaustedOrProviderDown",
        "LowLLMBudgetHeadroom",
        "MarketDataCircuitBreakerOpen",
        "StaleQuotesUsedInPortfolioSizing",
        "ExcessiveGuardrailBlocks",
        "JobQueueBacklogStalled",
    }
    assert expected_alerts.issubset(alert_names)


def test_alert_rules_reference_valid_registered_metrics():
    """Verify that metric names referenced in alert expressions exist in the metrics registry."""
    alerts_path = Path("monitoring/prometheus/alerts.yml")
    with open(alerts_path, "r", encoding="utf-8") as f:
        content = yaml.safe_load(f)

    registered_metric_names = set(metrics.REGISTRY._names_to_collectors.keys())

    # Check key domain metrics are present in registry
    assert "agent_degraded_total" in registered_metric_names
    assert "llm_calls_total" in registered_metric_names
    assert "market_data_circuit_open" in registered_metric_names
    assert "stale_quotes_total" in registered_metric_names
    assert "guardrail_blocks_total" in registered_metric_names
    assert "job_queue_depth" in registered_metric_names

    for rule in content["groups"][0]["rules"]:
        expr = rule["expr"]
        # Extract metric names like agent_degraded_total from promql expr
        found_metrics = re.findall(r"([a-z_][a-z0-9_]+_total|[a-z_][a-z0-9_]+_usd|[a-z_][a-z0-9_]+_open|job_queue_depth)", expr)
        for m in found_metrics:
            assert m in registered_metric_names, f"Metric '{m}' in alert '{rule['alert']}' must be in metrics.REGISTRY"
