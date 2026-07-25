"""Shared risk limits and the portfolio base used to measure against them.

Both the Portfolio Diagnostics agent and the Suitability guardrail claimed in
their comments to use "these EXACT numbers ... so the two agents tell the
client a consistent story". They did use the same percentages -- but applied
them to *different denominators*:

  * Diagnostics measured concentration against `sum(holdings.values())`, the
    total market value of the portfolio.
  * Suitability capped positions against `client_profile["net_worth"]`.

Whenever those two differ (an unpriced holding, a net worth that includes a
house or a pension, cash held outside the managed account), the client gets
told a position is over the 25% limit by one agent while the other happily
sizes up to a *different* 25%. Worse, the numbers move in opposite
directions: a client whose net worth exceeds their managed portfolio would be
flagged for concentration and simultaneously allowed to add more.

The limits and the denominator now both live here, so "consistent story" is
enforced by construction rather than by a comment asking two files to agree.
"""

from typing import Any, Dict, Mapping

# Max fraction of the portfolio base allowed in a single non-cash position,
# by the client's stated risk tolerance.
RISK_TIER_POSITION_CAP: Dict[str, float] = {
    "Conservative": 0.15,
    "Moderate": 0.25,
    "Aggressive": 0.35,
}
DEFAULT_POSITION_CAP = RISK_TIER_POSITION_CAP["Moderate"]


def position_cap_fraction(risk_tolerance: Any) -> float:
    """The single-position cap for a risk tolerance, defaulting to the
    Moderate tier for any unrecognized value."""
    return RISK_TIER_POSITION_CAP.get(str(risk_tolerance), DEFAULT_POSITION_CAP)


def portfolio_base_value(client_profile: Mapping[str, Any]) -> float:
    """The denominator every percentage-of-portfolio limit is measured against.

    This is the total market value of the holdings the system actually
    manages, including the cash sleeve -- NOT `net_worth`, which may include
    assets this system neither sees nor can rebalance (property, pensions,
    private holdings). Sizing a position as a fraction of net worth would let
    the system deploy against wealth it has no visibility into.

    Falls back to `net_worth` only when there are no holdings at all (a
    brand-new client whose money hasn't arrived yet), so onboarding still
    produces sensible position sizes.
    """
    holdings = client_profile.get("holdings") or {}
    total = sum(v for v in holdings.values() if isinstance(v, (int, float)) and v > 0)
    if total > 0:
        return float(total)
    return float(client_profile.get("net_worth", 0.0) or 0.0)
