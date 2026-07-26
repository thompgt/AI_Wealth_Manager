"""Multi-factor screening over the securities universe.

Replaces the previous approach, which was keyword matching against flaw
strings ("does this flaw text contain the word 'sector'?") followed by a
below-median P/E filter within whatever survived. That produced defensible
picks by accident: the median was computed over the handful of names that
passed the keyword step, so "cheap" meant cheap relative to an arbitrary
subset of six stocks, and a name could win purely by being the least expensive
thing in a small room.

The design here:

* **Hard constraints first, scoring second.** Market cap, liquidity, asset
  class, restrictions and explicit exclusions are eligibility questions with
  a yes/no answer. Blending them into a score would let a spectacular
  valuation buy its way past a liquidity floor, which is exactly what a floor
  exists to prevent.

* **Cross-sectional z-scores, ranked within peer group.** A P/E of 22 is
  expensive for a utility and cheap for a software company. Scoring against
  the whole universe reliably recommends whichever sector is currently
  cheapest and calls it stock selection. Factors are standardized within
  sector (or asset class, for funds) so the comparison is like-for-like.

* **Winsorized inputs.** One name with a P/E of 900 after an earnings
  collapse drags the mean and standard deviation enough to compress every
  other name's score toward zero. Clipping at the 5th/95th percentile keeps
  one outlier from silently disabling the factor for everyone else.

* **Missing is not average.** A name with no data for a factor gets no score
  from it and has its weights renormalized over what is known — rather than
  being imputed to the mean, which quietly rewards names for having nothing
  reported.

Scores rank candidates. They do not size positions and they do not approve
anything: that is the suitability guardrail's job, and keeping the optimizer
away from the constraint checker is deliberate.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db import Security, to_float
from logging_setup import get_logger

logger = get_logger(__name__)


# Factor definitions: (security column, higher_is_better). Kept declarative so
# a factor can be added, reweighted or dropped without touching the scoring
# code -- and so the weights that produced a recommendation can be recorded.
FACTORS: Dict[str, List[Tuple[str, bool]]] = {
    "value": [
        ("pe_ratio", False),
        ("pb_ratio", False),
        ("ps_ratio", False),
        ("peg_ratio", False),
    ],
    "quality": [
        ("return_on_equity", True),
        ("profit_margin", True),
        ("debt_to_equity", False),
        ("free_cash_flow", True),
    ],
    "momentum": [
        ("momentum_12m", True),
        ("momentum_6m", True),
    ],
    "growth": [
        ("revenue_growth", True),
        ("earnings_growth", True),
    ],
    "yield": [
        ("dividend_yield", True),
    ],
    "low_volatility": [
        ("volatility_1y", False),
        ("max_drawdown_1y", True),  # less negative is better
    ],
}

# Regime-dependent factor weights. The evidence for factor timing is thin, so
# these are deliberately mild tilts around a balanced base rather than
# aggressive rotations -- a system that swings its whole methodology on a
# low-confidence regime label is a system whose recommendations are mostly
# noise about the regime call.
REGIME_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Bull":        {"value": 0.15, "quality": 0.20, "momentum": 0.30, "growth": 0.20, "yield": 0.05, "low_volatility": 0.10},
    "Early-cycle": {"value": 0.25, "quality": 0.15, "momentum": 0.25, "growth": 0.20, "yield": 0.05, "low_volatility": 0.10},
    "Late-cycle":  {"value": 0.25, "quality": 0.30, "momentum": 0.05, "growth": 0.05, "yield": 0.15, "low_volatility": 0.20},
    "Bear":        {"value": 0.20, "quality": 0.30, "momentum": 0.00, "growth": 0.00, "yield": 0.20, "low_volatility": 0.30},
    "Volatile":    {"value": 0.20, "quality": 0.30, "momentum": 0.05, "growth": 0.05, "yield": 0.15, "low_volatility": 0.25},
}
DEFAULT_WEIGHTS = {"value": 0.20, "quality": 0.25, "momentum": 0.15, "growth": 0.15, "yield": 0.10, "low_volatility": 0.15}

# Risk tier nudges applied on top of the regime weights. A conservative
# mandate should not be handed the highest-momentum name in the universe just
# because momentum happens to be in favour.
TIER_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "Conservative": {"low_volatility": +0.15, "yield": +0.10, "quality": +0.05, "momentum": -0.15, "growth": -0.15},
    "Moderate": {},
    "Aggressive": {"momentum": +0.10, "growth": +0.10, "low_volatility": -0.10, "yield": -0.10},
}


@dataclass
class ScreenCriteria:
    """Hard eligibility rules. Everything here is pass/fail, never scored."""

    asset_classes: Optional[List[str]] = None
    min_market_cap: float = 2_000_000_000
    min_avg_dollar_volume: float = 5_000_000
    max_beta: Optional[float] = None
    min_dividend_yield: Optional[float] = None
    max_volatility: Optional[float] = None
    excluded_tickers: List[str] = field(default_factory=list)
    excluded_sectors: List[str] = field(default_factory=list)
    # Names already held. Excluded from *new* ideas but tracked separately,
    # because "you already own this" is a different answer from "this fails
    # our standards" and the two should not be conflated in a report.
    held_tickers: List[str] = field(default_factory=list)
    require_sector: Optional[List[str]] = None
    security_types: Optional[List[str]] = None


@dataclass
class ScreenResult:
    ticker: str
    name: Optional[str]
    asset_class: str
    sector: Optional[str]
    security_type: str
    composite_score: float
    factor_scores: Dict[str, Optional[float]]
    metrics: Dict[str, Optional[float]]
    coverage: float  # fraction of factor weight backed by real data
    peer_group: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "ticker": self.ticker,
            "name": self.name,
            "asset_class": self.asset_class,
            "sector": self.sector,
            "security_type": self.security_type,
            "composite_score": round(self.composite_score, 4),
            "factor_scores": {k: (round(v, 3) if v is not None else None)
                              for k, v in self.factor_scores.items()},
            "metrics": self.metrics,
            "coverage": round(self.coverage, 3),
            "peer_group": self.peer_group,
        }


@dataclass
class ScreenReport:
    """Results plus why everything else was excluded.

    The rejection counts are not diagnostics for developers; they are what
    lets the system say "nothing passed because your minimum market cap
    excluded 94% of the universe" rather than returning an empty list and
    leaving the advisor to guess.
    """

    results: List[ScreenResult]
    universe_size: int
    eligible_size: int
    rejections: Dict[str, int]
    weights: Dict[str, float]

    def summary(self) -> str:
        if not self.rejections:
            return f"{self.eligible_size} of {self.universe_size} securities were eligible."
        top = sorted(self.rejections.items(), key=lambda kv: -kv[1])[:3]
        detail = ", ".join(f"{reason} ({count})" for reason, count in top)
        return (
            f"{self.eligible_size} of {self.universe_size} securities were eligible; "
            f"main exclusions: {detail}."
        )


def resolve_weights(regime_label: Optional[str], risk_tier: Optional[str]) -> Dict[str, float]:
    """Blend regime and risk-tier tilts into a normalized weight vector."""
    weights = dict(REGIME_WEIGHTS.get(regime_label or "", DEFAULT_WEIGHTS))
    for factor, delta in TIER_ADJUSTMENTS.get(risk_tier or "Moderate", {}).items():
        weights[factor] = max(0.0, weights.get(factor, 0.0) + delta)
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {factor: weight / total for factor, weight in weights.items()}


def _winsorized_z(values: pd.Series) -> pd.Series:
    """Z-score after clipping the tails.

    Without the clip, a single P/E of 900 sets the standard deviation and
    compresses every other name into a band around zero -- the factor stops
    discriminating at all, silently, and only for the groups containing an
    outlier.
    """
    clean = values.dropna()
    if len(clean) < 3:
        return pd.Series(np.nan, index=values.index)
    low, high = clean.quantile(0.05), clean.quantile(0.95)
    clipped = values.clip(lower=low, upper=high)
    std = clipped.std()
    if not std or std != std or std == 0:
        return pd.Series(0.0, index=values.index).where(values.notna())
    return (clipped - clipped.mean()) / std


# Below this many members, a z-score says more about the group's size than
# about the security. In a group of three, the best name is mechanically ~1.2
# standard deviations above the mean whatever its actual merit -- which is how
# a thinly-populated corner of the universe ends up topping every screen.
MIN_PEER_GROUP = 6


def _peer_group(security: Security) -> str:
    """The cohort a name is scored against.

    Funds are grouped by asset class and individual equities by sector. A P/E
    of 22 is expensive for a utility and cheap for software; scoring across
    the whole universe reliably picks whichever sector is cheapest right now
    and presents it as stock selection.
    """
    if security.security_type in ("etf", "fund"):
        return f"fund:{security.asset_class}"
    return f"equity:{security.sector or 'Unknown'}"


def _assign_peer_groups(securities: Sequence[Security]) -> Dict[str, str]:
    """Peer group per ticker, collapsing groups too small to standardize.

    A sector with two names cannot support a meaningful cross-sectional
    z-score, so its members are pooled into a broader cohort rather than being
    scored against each other. Pooling loses some like-for-like precision; a
    two-member group loses all of it.
    """
    def broader(security: Security) -> str:
        kind = "fund" if security.security_type in ("etf", "fund") else "equity"
        return f"{kind}:{security.asset_class}"

    natural = {s.ticker: _peer_group(s) for s in securities}
    fallback = {s.ticker: broader(s) for s in securities}

    natural_counts: Dict[str, int] = {}
    fallback_counts: Dict[str, int] = {}
    for security in securities:
        natural_counts[natural[security.ticker]] = natural_counts.get(natural[security.ticker], 0) + 1
        fallback_counts[fallback[security.ticker]] = fallback_counts.get(fallback[security.ticker], 0) + 1

    resolved: Dict[str, str] = {}
    for security in securities:
        ticker = security.ticker
        if natural_counts[natural[ticker]] >= MIN_PEER_GROUP:
            resolved[ticker] = natural[ticker]
        elif fallback_counts[fallback[ticker]] >= MIN_PEER_GROUP:
            resolved[ticker] = fallback[ticker]
        else:
            # If even the asset-class cohort is thin, compare against the
            # whole universe. A wide comparison is imprecise; a comparison
            # against two peers is arbitrary.
            resolved[ticker] = "all"
    return resolved


def _eligible(security: Security, criteria: ScreenCriteria) -> Optional[str]:
    """None if the security passes, otherwise the reason it did not."""
    ticker = security.ticker.upper()

    if security.is_restricted:
        return "firm restricted list"
    if ticker in {t.upper() for t in criteria.excluded_tickers}:
        return "excluded by policy or this run"
    if ticker in {t.upper() for t in criteria.held_tickers}:
        return "already held"
    if criteria.asset_classes and security.asset_class not in criteria.asset_classes:
        return "asset class not permitted by policy"
    if criteria.security_types and security.security_type not in criteria.security_types:
        return "security type not requested"
    if security.sector and security.sector in set(criteria.excluded_sectors):
        return "sector excluded by policy"
    if criteria.require_sector and security.sector not in set(criteria.require_sector):
        return "outside the requested sector"

    # Funds have no market cap in the sense an equity does, and applying an
    # equity floor to them would exclude the entire core allocation -- the
    # instruments most likely to be the right answer.
    if security.security_type not in ("etf", "fund"):
        market_cap = to_float(security.market_cap) if security.market_cap is not None else None
        if market_cap is None:
            return "no verifiable market capitalisation"
        if market_cap < criteria.min_market_cap:
            return "below minimum market capitalisation"

    volume = to_float(security.avg_dollar_volume) if security.avg_dollar_volume is not None else None
    if volume is not None and volume < criteria.min_avg_dollar_volume:
        return "below minimum average dollar volume"

    if criteria.max_beta is not None and security.beta is not None and security.beta > criteria.max_beta:
        return "beta above the policy ceiling"
    if (
        criteria.max_volatility is not None
        and security.volatility_1y is not None
        and security.volatility_1y > criteria.max_volatility
    ):
        return "volatility above the policy ceiling"
    if (
        criteria.min_dividend_yield is not None
        and (security.dividend_yield or 0.0) < criteria.min_dividend_yield
    ):
        return "dividend yield below the requested minimum"
    return None


def screen(
    db: Session,
    criteria: ScreenCriteria,
    *,
    regime_label: Optional[str] = None,
    risk_tier: Optional[str] = None,
    weights: Optional[Dict[str, float]] = None,
    limit: int = 25,
) -> ScreenReport:
    """Score the eligible universe and return the top `limit` names."""
    universe = (
        db.query(Security)
        .filter(Security.is_active.is_(True))
        .all()
    )
    rejections: Dict[str, int] = {}
    eligible: List[Security] = []
    for security in universe:
        reason = _eligible(security, criteria)
        if reason is None:
            eligible.append(security)
        else:
            rejections[reason] = rejections.get(reason, 0) + 1

    resolved_weights = weights or resolve_weights(regime_label, risk_tier)

    if not eligible:
        return ScreenReport([], len(universe), 0, rejections, resolved_weights)

    peer_groups = _assign_peer_groups(eligible)
    frame = pd.DataFrame(
        [
            {
                "ticker": s.ticker,
                "peer_group": peer_groups[s.ticker],
                **{
                    column: (
                        to_float(getattr(s, column))
                        if getattr(s, column) is not None
                        else np.nan
                    )
                    for column in _all_columns()
                },
            }
            for s in eligible
        ]
    ).set_index("ticker")

    # A negative P/E is not "very cheap"; it means the company lost money, and
    # ranking on it ascending puts the biggest losses at the top of the value
    # factor. Treat non-positive multiples as missing.
    for column in ("pe_ratio", "pb_ratio", "ps_ratio", "peg_ratio", "forward_pe"):
        if column in frame:
            frame.loc[frame[column] <= 0, column] = np.nan

    factor_frame = _score_factors(frame)
    composite, coverage = _composite(factor_frame, resolved_weights)

    by_ticker = {s.ticker: s for s in eligible}
    results: List[ScreenResult] = []
    for ticker in composite.sort_values(ascending=False).index:
        security = by_ticker[ticker]
        results.append(
            ScreenResult(
                ticker=ticker,
                name=security.name,
                asset_class=security.asset_class,
                sector=security.sector,
                security_type=security.security_type,
                composite_score=float(composite[ticker]),
                factor_scores={
                    factor: (None if pd.isna(factor_frame.loc[ticker, factor])
                             else float(factor_frame.loc[ticker, factor]))
                    for factor in FACTORS
                },
                metrics=_metric_snapshot(security),
                coverage=float(coverage[ticker]),
                peer_group=peer_groups[ticker],
            )
        )
        if len(results) >= limit:
            break

    return ScreenReport(results, len(universe), len(eligible), rejections, resolved_weights)


def _all_columns() -> List[str]:
    columns: List[str] = []
    for definitions in FACTORS.values():
        for column, _ in definitions:
            if column not in columns:
                columns.append(column)
    return columns


def _score_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-factor z-scores, standardized within peer group."""
    scores = pd.DataFrame(index=frame.index, columns=list(FACTORS), dtype=float)

    for factor, definitions in FACTORS.items():
        component_scores = []
        for column, higher_is_better in definitions:
            if column not in frame:
                continue
            per_group = []
            for _, group in frame.groupby("peer_group"):
                z = _winsorized_z(group[column])
                per_group.append(z if higher_is_better else -z)
            if not per_group:
                continue
            component_scores.append(pd.concat(per_group).reindex(frame.index))
        if not component_scores:
            continue
        stacked = pd.concat(component_scores, axis=1)
        # Mean of the components that exist, so a name missing one of four
        # value metrics still receives a value score from the other three
        # rather than being dropped from the factor entirely.
        scores[factor] = stacked.mean(axis=1, skipna=True)
    return scores


def _composite(
    factor_frame: pd.DataFrame, weights: Dict[str, float]
) -> Tuple[pd.Series, pd.Series]:
    """Weighted composite plus the fraction of weight actually backed by data.

    Renormalizing over available factors is the alternative to imputing
    missing values to the group mean. Imputation would quietly reward a name
    for having nothing reported -- it scores average on every factor it lacks
    and above average on the ones it has.
    """
    composite = pd.Series(0.0, index=factor_frame.index)
    coverage = pd.Series(0.0, index=factor_frame.index)

    for factor, weight in weights.items():
        if factor not in factor_frame or weight <= 0:
            continue
        values = factor_frame[factor]
        present = values.notna()
        composite[present] += values[present] * weight
        coverage[present] += weight

    # Guard against dividing by zero for a name with no factor data at all.
    normalized = composite / coverage.replace(0.0, np.nan)
    return normalized.fillna(0.0), coverage


def _metric_snapshot(security: Security) -> Dict[str, Optional[float]]:
    """The raw numbers behind a score, for the report and the audit record.

    A composite score is not an explanation. Carrying the inputs means the
    recommendation can say "12x earnings, 18% ROE, 22% below its high" rather
    than "scored 1.8".
    """
    def value(attr: str) -> Optional[float]:
        raw = getattr(security, attr, None)
        if raw is None:
            return None
        number = to_float(raw)
        return round(number, 4) if abs(number) < 1e6 else round(number, 0)

    return {
        "market_cap": value("market_cap"),
        "pe_ratio": value("pe_ratio"),
        "forward_pe": value("forward_pe"),
        "pb_ratio": value("pb_ratio"),
        "peg_ratio": value("peg_ratio"),
        "dividend_yield": value("dividend_yield"),
        "beta": value("beta"),
        "return_on_equity": value("return_on_equity"),
        "profit_margin": value("profit_margin"),
        "debt_to_equity": value("debt_to_equity"),
        "revenue_growth": value("revenue_growth"),
        "momentum_6m": value("momentum_6m"),
        "momentum_12m": value("momentum_12m"),
        "volatility_1y": value("volatility_1y"),
        "max_drawdown_1y": value("max_drawdown_1y"),
        "expense_ratio": value("expense_ratio"),
    }


def correlation_to_portfolio(
    candidate_prices: pd.DataFrame,
    portfolio_weights: Dict[str, float],
    candidates: Iterable[str],
) -> Dict[str, Optional[float]]:
    """Each candidate's return correlation to the existing portfolio.

    The single most useful diversification number and the one the previous
    screen had no notion of: a name can be cheap, high quality and in a
    different sector while still moving in lockstep with what the client
    already owns, in which case buying it does not reduce risk. Returned as a
    ranking input, never as a hard filter -- a high-correlation name can still
    be the right holding.
    """
    if candidate_prices.empty or not portfolio_weights:
        return {ticker: None for ticker in candidates}

    returns = candidate_prices.pct_change().dropna(how="all")
    held = [t for t in portfolio_weights if t in returns.columns]
    if not held:
        return {ticker: None for ticker in candidates}

    total = sum(portfolio_weights[t] for t in held)
    if total <= 0:
        return {ticker: None for ticker in candidates}

    portfolio_returns = sum(
        returns[t].fillna(0.0) * (portfolio_weights[t] / total) for t in held
    )

    output: Dict[str, Optional[float]] = {}
    for ticker in candidates:
        if ticker not in returns.columns:
            output[ticker] = None
            continue
        series = returns[ticker]
        joined = pd.concat([series, portfolio_returns], axis=1).dropna()
        # Fewer than ~30 overlapping observations gives a correlation whose
        # confidence interval spans most of [-1, 1]; reporting it as a number
        # implies a precision that is not there.
        if len(joined) < 30:
            output[ticker] = None
            continue
        value = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        output[ticker] = None if value != value else round(value, 3)
    return output
