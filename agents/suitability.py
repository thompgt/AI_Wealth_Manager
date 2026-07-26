"""Suitability guardrail: the gate between "good idea" and "client-facing".

Deterministic, rule-based, no LLM anywhere in this file. That is the single
most important design decision in the system. A model that is right 97% of the
time is a useful research analyst and an unacceptable control: the 3% is
exactly the case a control exists to catch, and "the model judged it suitable"
is not an answer to a regulator asking why a client was sold something
unsuitable.

Every rule here resolves from the client's versioned Investment Policy
Statement rather than a module constant, so the limits that produced a
decision can be recovered later, along with who approved them and when.

Failure direction is deliberate and differs by rule:

* **Security quality fails closed.** An unverifiable security is not
  recommended. If no provider can tell us the market cap or listing venue of
  something, that is a reason to decline, not to proceed hopefully.
* **Beta fails open.** Beta is genuinely missing for many legitimate
  instruments, so an absent value skips that check rather than rejecting the
  name. Failing closed here would exclude most bond funds from every
  conservative portfolio -- the opposite of the intent.

Sizing lives here too, because sizing *is* a suitability decision. Cash is
distributed by confidence, capped per position by policy, and constrained so
the result does not breach the sector limit or drain the cash floor -- the
three ways a correctly-chosen security can still be the wrong trade.
"""

from typing import Any, Dict, List, Optional, Tuple

from db import ClientProfile as ClientProfileModel
from db import SessionLocal
from logging_setup import get_logger
from metrics import guardrail_blocks
from services.market_data import get_security_info
from services.policy import resolve as resolve_policy
from services.portfolio import load_portfolio
from state import AgentState, Candidate, SuitabilityResult

from agents.runtime import finish, node_run

logger = get_logger(__name__)

NODE_NAME = "suitability"

# Exchange codes as actually returned by the provider layer. Verified against
# live tickers:
#   NMS/NGM/NCM = Nasdaq tiers, NYQ = NYSE, ASE = NYSE American,
#   PCX = NYSE Arca (most ETFs), BATS = Cboe BZX.
# The OTC tiers that must be rejected are PNK (pink sheets), OQX/OQB (OTCQX,
# OTCQB) and OID (foreign ADRs on OTC Markets).
MAJOR_EXCHANGES = {"NMS", "NGM", "NCM", "NYQ", "ASE", "PCX", "BATS", "ARCA", "NYSE", "NASDAQ"}

# The rule applies to clients at or near the point of drawing on the
# portfolio, where a deep drawdown cannot be waited out.
NEAR_RETIREMENT_AGE = 60
SHORT_HORIZON_YEARS = 5


def _reject(ticker: str, rule: str, reason: str) -> str:
    """Format a rejection.

    The `TICKER: rejected -- ...` shape is load-bearing: the guardrail gate
    parses the leading token to build the exclusion list that makes the retry
    loop do something different on its next pass. Changing this format
    silently turns the retry into an expensive no-op, which is exactly what
    happened before, so the parse is also defended on the gate side.
    """
    guardrail_blocks.labels(rule).inc()
    return f"{ticker}: rejected -- {reason}"


def _check_security_quality(candidate: Candidate, policy) -> Optional[str]:
    """Size, liquidity and listing venue. Fails closed on missing data."""
    ticker = candidate["ticker"]
    info = get_security_info(ticker)

    if info is None:
        return _reject(
            ticker, "unverifiable",
            "no market data provider could return verifiable reference data for this "
            "security, so its quality cannot be confirmed.",
        )

    is_fund = (info.quote_type or "").upper() in ("ETF", "MUTUALFUND")

    if not is_fund:
        if info.market_cap is None:
            return _reject(
                ticker, "unverifiable",
                "market capitalisation is unavailable, so security quality cannot be verified.",
            )
        if info.market_cap < policy.min_market_cap:
            return _reject(
                ticker, "min_market_cap",
                f"market cap of ${info.market_cap:,.0f} is below this client's "
                f"${policy.min_market_cap:,.0f} minimum.",
            )

    if info.exchange is None:
        return _reject(
            ticker, "unverifiable",
            "the listing venue is unavailable, so it cannot be confirmed to trade on a "
            "major exchange.",
        )
    if info.exchange.upper() not in MAJOR_EXCHANGES:
        return _reject(
            ticker, "exchange",
            f"it is listed on '{info.exchange}', not a major US exchange. Over-the-counter "
            f"listings carry disclosure and liquidity risks this mandate does not accept.",
        )

    if (
        info.avg_dollar_volume is not None
        and info.avg_dollar_volume < policy.min_avg_dollar_volume
    ):
        return _reject(
            ticker, "liquidity",
            f"average daily turnover of ${info.avg_dollar_volume:,.0f} is below the "
            f"${policy.min_avg_dollar_volume:,.0f} floor; a client-sized position could not "
            f"be exited without moving the price.",
        )
    return None


def _check_policy_exclusions(candidate: Candidate, policy) -> Optional[str]:
    """Explicit client and firm prohibitions."""
    ticker = candidate["ticker"]
    if ticker.upper() in {t.upper() for t in policy.excluded_tickers}:
        return _reject(ticker, "policy_exclusion", "it is on this client's exclusion list.")

    sector = candidate.get("sector")
    if sector and sector in set(policy.excluded_sectors):
        return _reject(
            ticker, "sector_exclusion",
            f"the {sector} sector is excluded by this client's policy.",
        )

    asset_class = candidate.get("asset_class")
    if policy.allowed_asset_classes and asset_class not in policy.allowed_asset_classes:
        return _reject(
            ticker, "asset_class",
            f"the {asset_class} asset class is not permitted by this client's policy.",
        )
    return None


def _check_risk_fit(candidate: Candidate, policy, profile: Dict[str, Any]) -> Optional[str]:
    """Beta ceiling for near-retirement or short-horizon clients.

    Skipped when beta is unavailable. Unlike the quality checks, failing
    closed here would reject most bond and international funds -- precisely
    the instruments a conservative mandate needs.
    """
    ticker = candidate["ticker"]
    info = get_security_info(ticker)
    beta = info.beta if info else None
    if beta is None:
        return None

    if policy.max_position_beta is not None and beta > policy.max_position_beta:
        return _reject(
            ticker, "beta_policy",
            f"beta of {beta:.2f} exceeds the {policy.max_position_beta:.2f} ceiling in this "
            f"client's {policy.risk_tier} policy.",
        )

    age = profile.get("age") or 0
    horizon = profile.get("time_horizon_years") or 99
    if age >= NEAR_RETIREMENT_AGE or horizon <= SHORT_HORIZON_YEARS:
        if beta > 1.5:
            return _reject(
                ticker, "beta_horizon",
                f"beta of {beta:.2f} is too high for a client aged {age} with a "
                f"{horizon}-year horizon; there is not enough time to recover from a "
                f"drawdown that a position this sensitive would deepen.",
            )
    return None


def _compute_allocations(
    candidates: List[Candidate],
    view,
    policy,
) -> Tuple[Dict[str, float], List[str]]:
    """Distribute investable cash across approved candidates.

    Water-filling, weighted by confidence: each candidate's share is capped by
    its remaining position headroom, and cash that overflows a capped name is
    redistributed to the others rather than abandoned. Converges in a handful
    of passes for realistic candidate counts.

    Two constraints beyond the position cap, both of which the previous
    version omitted and both of which turn a good pick into a bad trade:

    * The **cash floor** is reserved first. Deploying the entire balance
      creates the liquidity flaw diagnostics warns about on the next run.
    * The **sector cap** is enforced as the basket is built, so five names
      that are individually fine do not collectively breach a sector limit.
    """
    notes: List[str] = []
    tickers = [c["ticker"] for c in candidates]
    allocation = {t: 0.0 for t in tickers}
    if not tickers:
        return allocation, notes

    total = view.total_value
    investable = max(0.0, min(view.cash - policy.min_cash_pct * total, view.spendable_cash))
    if investable < policy.min_position_notional or total <= 0:
        notes.append(
            f"No allocation was made: ${investable:,.0f} is investable after reserving the "
            f"{policy.min_cash_pct:.0%} cash floor, below the "
            f"${policy.min_position_notional:,.0f} minimum position size."
        )
        return allocation, notes

    current_weights = view.weights()
    cap_dollars = policy.max_position_pct * total
    headroom = {
        t: max(0.0, cap_dollars - current_weights.get(t, 0.0) * total) for t in tickers
    }

    # Sector budget, tracked as the basket is assembled.
    invested = view.invested_value
    sector_value: Dict[str, float] = {}
    for holding in view.holdings:
        if view.is_diversified_fund(holding):
            continue
        key = holding.sector or "Unclassified"
        sector_value[key] = sector_value.get(key, 0.0) + holding.market_value

    sectors = {
        c["ticker"]: (c.get("sector") if c.get("security_type") not in ("etf", "fund") else None)
        for c in candidates
    }

    confidences = {c["ticker"]: max(float(c.get("confidence") or 0.0), 0.0) for c in candidates}
    total_confidence = sum(confidences.values())
    weight = (
        {t: v / total_confidence for t, v in confidences.items()}
        if total_confidence > 0
        # Equal weight when nothing has a confidence -- which happens on the
        # deterministic path, and is the honest response to having no ranking
        # signal rather than an arbitrary ordering.
        else {t: 1.0 / len(tickers) for t in tickers}
    )

    remaining = investable
    open_tickers = set(tickers)
    for _ in range(6):
        if remaining <= 1e-6 or not open_tickers:
            break
        open_weight = sum(weight[t] for t in open_tickers)
        if open_weight <= 0:
            break
        distributed = 0.0
        newly_capped = []
        for ticker in list(open_tickers):
            share = remaining * (weight[ticker] / open_weight)
            room = headroom[ticker] - allocation[ticker]

            sector = sectors.get(ticker)
            if sector:
                current = sector_value.get(sector, 0.0)
                base = invested + sum(allocation.values())
                # (current + x) / (base + x) <= cap
                cap = policy.max_sector_pct
                sector_room = (
                    max(0.0, (cap * base - current) / (1 - cap)) if cap < 1 else float("inf")
                )
                room = min(room, sector_room)

            give = max(0.0, min(share, room))
            allocation[ticker] += give
            if sector and give > 0:
                sector_value[sector] = sector_value.get(sector, 0.0) + give
            distributed += give
            if allocation[ticker] >= headroom[ticker] - 1e-9 or room <= 1e-9:
                newly_capped.append(ticker)
        remaining -= distributed
        for ticker in newly_capped:
            open_tickers.discard(ticker)
        if distributed <= 1e-9:
            break

    if remaining > policy.min_position_notional:
        notes.append(
            f"${remaining:,.0f} of investable cash was left undeployed because every "
            f"approved candidate reached its position or sector limit. Deploying it would "
            f"have breached a policy limit, so the run invests less rather than more."
        )
    return allocation, notes


def suitability_node(state: AgentState) -> dict:
    """Evaluate candidates against the client's policy and size the survivors."""
    result: Optional[SuitabilityResult] = None

    with node_run(NODE_NAME, state) as ctx:
        candidates: List[Candidate] = list(state.get("candidate_stocks") or [])
        profile = state.get("client_profile") or {}
        client_id = profile.get("client_id")

        db = SessionLocal()
        try:
            client = (
                db.query(ClientProfileModel).filter(ClientProfileModel.id == client_id).first()
            )
            if client is None:
                raise ValueError(f"No client profile with id {client_id}")

            policy = resolve_policy(db, client)
            view = load_portfolio(db, client)
            ctx.policy_version = policy.version
            ctx.input_snapshot = {
                "candidates": [c.get("ticker") for c in candidates],
                "policy_version": policy.version,
                "investable_cash": round(view.cash, 2),
            }

            violations: List[str] = []
            passed: List[Candidate] = []
            checks_applied = [
                "security_quality", "policy_exclusions", "risk_fit",
                "position_cap", "sector_cap", "minimum_size",
            ]

            for candidate in candidates:
                reason = (
                    _check_policy_exclusions(candidate, policy)
                    or _check_security_quality(candidate, policy)
                    or _check_risk_fit(candidate, policy, profile)
                )
                if reason is not None:
                    logger.info("[Suitability] %s", reason)
                    violations.append(reason)
                else:
                    passed.append(candidate)

            allocations, notes = _compute_allocations(passed, view, policy)
            violations.extend(notes)

            approved_recommendations: List[Candidate] = []
            total = view.total_value
            for candidate in passed:
                ticker = candidate["ticker"]
                amount = allocations.get(ticker, 0.0)
                if amount < policy.min_position_notional:
                    violations.append(
                        _reject(
                            ticker, "min_size",
                            f"only ${amount:,.0f} could be allocated after position and sector "
                            f"limits, below the ${policy.min_position_notional:,.0f} minimum "
                            f"position size.",
                        )
                    )
                    continue
                approved_recommendations.append(
                    {
                        **candidate,
                        "allocation_amount": round(amount, 2),
                        "allocation_pct": round(amount / total, 4) if total > 0 else 0.0,
                    }
                )
                logger.info(
                    "[Suitability] approved %s: $%s (%.1f%% of portfolio)",
                    ticker, f"{amount:,.0f}", (amount / total * 100) if total else 0,
                )

            result = SuitabilityResult(
                approved=bool(approved_recommendations),
                violations=violations,
                adjusted_recommendations=approved_recommendations,
                checks_applied=checks_applied,
            )

            allocated = sum(c["allocation_amount"] for c in approved_recommendations)
            ctx.output_snapshot = {
                "approved": len(approved_recommendations),
                "rejected": len(candidates) - len(approved_recommendations),
                "total_allocated": round(allocated, 2),
            }
            ctx.summary = (
                f"Evaluated {len(candidates)} candidate(s): {len(approved_recommendations)} "
                f"approved for ${allocated:,.0f}, {len(violations)} violation(s) recorded."
            )
            logger.info("[Suitability] %s", ctx.summary)
        finally:
            db.close()

    if result is None:
        # The node raised. Fail closed: recommending nothing is the safe
        # direction for a control that could not complete.
        result = SuitabilityResult(
            approved=False,
            violations=[
                "The suitability guardrail could not complete, so no recommendation is "
                "approved. This is a deliberate fail-closed: an unevaluated recommendation "
                "is not an approved one."
            ],
            adjusted_recommendations=[],
            checks_applied=[],
        )

    return finish(ctx, {"suitability_result": result})
