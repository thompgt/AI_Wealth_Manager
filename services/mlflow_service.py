"""MLflow service for tracking machine learning, quantitative portfolio metrics, and LLM inference.

Observability and reproducibility for model and quant decisions:
1. Portfolio optimization & risk metrics (Sharpe ratio, volatility, max drawdown, weights).
2. Market regime classification (regime label, confidence, macro indicator signals).
3. LLM generation metadata (model, tokens, latency, temperature).

Fails open: if MLflow is not installed or tracking is disabled, calls degrade cleanly
to no-ops without throwing exceptions or interrupting graph execution.
"""

import threading
from typing import Any, Dict, Optional

from config import settings
from logging_setup import get_logger

logger = get_logger(__name__)

_mlflow = None
_init_lock = threading.Lock()
_initialized = False
_active_runs: Dict[str, str] = {}  # awm_run_id -> mlflow_run_id


def is_available() -> bool:
    """True if MLflow tracking is enabled and the library is installed."""
    if not getattr(settings, "MLFLOW_ENABLED", True):
        return False
    global _mlflow, _initialized
    if not _initialized:
        with _init_lock:
            if not _initialized:
                try:
                    import mlflow
                    _mlflow = mlflow
                    _mlflow.set_tracking_uri(getattr(settings, "MLFLOW_TRACKING_URI", "file:./mlruns"))
                    _mlflow.set_experiment(getattr(settings, "MLFLOW_EXPERIMENT_NAME", "ai-wealth-manager"))
                    logger.info("MLflow tracking initialized at %s", settings.MLFLOW_TRACKING_URI)
                except ImportError:
                    logger.debug("MLflow library not installed; tracking disabled.")
                    _mlflow = None
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not initialize MLflow tracking: %s", exc)
                    _mlflow = None
                finally:
                    _initialized = True
    return _mlflow is not None


def start_run(run_id: str, tags: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Start an MLflow tracking run associated with this analysis run."""
    if not is_available():
        return None
    try:
        run_tags = {"awm_run_id": run_id}
        if tags:
            run_tags.update({str(k): str(v) for k, v in tags.items() if v is not None})

        mlflow_run = _mlflow.start_run(run_name=f"run-{run_id[:8]}", tags=run_tags, nested=True)
        ml_id = mlflow_run.info.run_id
        _active_runs[run_id] = ml_id
        logger.debug("Started MLflow run %s for analysis run %s", ml_id, run_id)
        return ml_id
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow start_run error: %s", exc)
        return None


def end_run(run_id: str, status: str = "FINISHED") -> None:
    """End the MLflow run for this analysis run."""
    if not is_available():
        return
    ml_id = _active_runs.pop(run_id, None)
    if not ml_id:
        return
    try:
        _mlflow.end_run(status=status)
        logger.debug("Ended MLflow run %s (status=%s)", ml_id, status)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow end_run error: %s", exc)


def log_portfolio_metrics(
    run_id: str,
    stats: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
) -> None:
    """Log quantitative portfolio optimization and risk metrics."""
    if not is_available():
        return
    try:
        metrics_to_log = {}
        for key in ("sharpe_ratio", "annual_return", "annual_volatility", "max_drawdown", "portfolio_beta", "diversification_score"):
            val = stats.get(key)
            if isinstance(val, (int, float)):
                metrics_to_log[key] = float(val)

        if weights:
            for symbol, weight in weights.items():
                if isinstance(weight, (int, float)):
                    metrics_to_log[f"weight_{symbol}"] = float(weight)

        if metrics_to_log:
            _mlflow.log_metrics(metrics_to_log)
            logger.debug("Logged %d portfolio metrics to MLflow for run %s", len(metrics_to_log), run_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow log_portfolio_metrics error: %s", exc)


def log_regime_inference(
    run_id: str,
    regime_label: str,
    confidence: float,
    signals: Optional[Dict[str, Any]] = None,
) -> None:
    """Log macro market regime classification outcomes and supporting indicators."""
    if not is_available():
        return
    try:
        _mlflow.log_param(f"regime_label_{run_id[:8]}", regime_label)
        _mlflow.log_metric("regime_confidence", float(confidence))

        if signals:
            ratios = signals.get("ratio_signals") or {}
            for name, val in ratios.items():
                if isinstance(val, dict) and "change_pct" in val:
                    pct = val["change_pct"]
                    if isinstance(pct, (int, float)):
                        clean_name = name.split()[0].replace("/", "_")
                        _mlflow.log_metric(f"macro_ratio_{clean_name}", float(pct))

            solos = signals.get("ticker_pct_change") or {}
            for ticker, change in solos.items():
                if isinstance(change, (int, float)):
                    clean_ticker = ticker.replace("^", "").replace("=", "_").replace(".", "_")
                    _mlflow.log_metric(f"macro_trend_{clean_ticker}", float(change))

        logger.debug("Logged regime inference (%s, conf=%.2f) to MLflow", regime_label, confidence)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow log_regime_inference error: %s", exc)


def log_llm_generation(
    run_id: str,
    node: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    latency_ms: float,
) -> None:
    """Log LLM token usage and latency metrics."""
    if not is_available():
        return
    try:
        _mlflow.log_metrics(
            {
                f"{node}_prompt_tokens": float(prompt_tokens),
                f"{node}_completion_tokens": float(completion_tokens),
                f"{node}_latency_ms": float(latency_ms),
            }
        )
        logger.debug("Logged LLM generation metrics for %s to MLflow", node)
    except Exception as exc:  # noqa: BLE001
        logger.debug("MLflow log_llm_generation error: %s", exc)
