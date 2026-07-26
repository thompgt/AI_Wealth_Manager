"""Investment Policy Statement: resolution, versioning and defaults.

Every limit this system enforces used to be a module-level constant —
`RISK_TIER_POSITION_CAP`, `MIN_MARKET_CAP`, `MAX_BETA_NEAR_RETIREMENT`,
`SECTOR_CONCENTRATION_THRESHOLD`. They worked, but they made one question
unanswerable: *what limits were in force when this recommendation was made?*
The only way to find out was to read git history and hope the deploy timeline
matched. For a system that withholds recommendations and must justify doing
so, that is the wrong shape.

Limits now resolve from a versioned `investment_policies` row, and every run
stamps the version it used onto its audit record. A policy edit supersedes the
prior version rather than mutating it, so the row that authorised a March
recommendation still exists in April, unchanged, with the approver and
effective dates attached.

`resolve` never fails. A client with no policy gets tier-appropriate defaults,
marked as such, so the system is usable from the first minute of onboarding —
but the resolved object always knows whether it came from an approved policy
or a default, and the report says which.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from db import ClientProfile, InvestmentPolicy, Organization, to_float, utcnow
from logging_setup import get_logger

logger = get_logger(__name__)


# Tier defaults. These are the old hardcoded constants, kept as *defaults*
# rather than as *rules* -- the difference being that an advisor can now
# depart from them for a specific client, with a version and an approver
# recording that they did.
TIER_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "Conservative": {
        "max_position_pct": 0.15,
        "max_sector_pct": 0.25,
        "max_asset_class_pct": 0.60,
        "min_cash_pct": 0.05,
        "max_cash_pct": 0.25,
        "max_position_beta": 1.10,
        "max_portfolio_beta": 0.85,
        "max_portfolio_volatility": 0.12,
        "min_market_cap": 5_000_000_000,
        "min_avg_dollar_volume": 20_000_000,
        "target_allocation": {
            "us_equity": 0.28, "intl_equity": 0.10, "em_equity": 0.02,
            "fixed_income": 0.48, "real_assets": 0.05, "commodity": 0.02, "cash": 0.05,
        },
        "rebalance_frequency_days": 180,
    },
    "Moderate": {
        "max_position_pct": 0.25,
        "max_sector_pct": 0.35,
        "max_asset_class_pct": 0.75,
        "min_cash_pct": 0.03,
        "max_cash_pct": 0.20,
        "max_position_beta": 1.40,
        "max_portfolio_beta": 1.05,
        "max_portfolio_volatility": 0.18,
        "min_market_cap": 2_000_000_000,
        "min_avg_dollar_volume": 10_000_000,
        "target_allocation": {
            "us_equity": 0.42, "intl_equity": 0.15, "em_equity": 0.05,
            "fixed_income": 0.28, "real_assets": 0.05, "commodity": 0.02, "cash": 0.03,
        },
        "rebalance_frequency_days": 90,
    },
    "Aggressive": {
        "max_position_pct": 0.35,
        "max_sector_pct": 0.45,
        "max_asset_class_pct": 0.90,
        "min_cash_pct": 0.02,
        "max_cash_pct": 0.15,
        "max_position_beta": 1.80,
        "max_portfolio_beta": 1.30,
        "max_portfolio_volatility": 0.26,
        "min_market_cap": 1_000_000_000,
        "min_avg_dollar_volume": 5_000_000,
        "target_allocation": {
            "us_equity": 0.55, "intl_equity": 0.18, "em_equity": 0.08,
            "fixed_income": 0.12, "real_assets": 0.04, "commodity": 0.01, "cash": 0.02,
        },
        "rebalance_frequency_days": 60,
    },
}

# Drift tolerance before a rebalance is proposed. Wider on the volatile sleeves
# and narrower on cash: rebalancing costs money and taxes, so a band that is
# too tight trades constantly and gives the gains to the spread. These are
# absolute weight deviations, not relative.
DEFAULT_DRIFT_BANDS = {
    "us_equity": 0.05,
    "intl_equity": 0.04,
    "em_equity": 0.03,
    "fixed_income": 0.05,
    "real_assets": 0.03,
    "commodity": 0.02,
    "cash": 0.03,
    "alternative": 0.03,
}


@dataclass
class ResolvedPolicy:
    """The limits in force for one run, and where they came from."""

    client_id: int
    version: Optional[int]
    source: str  # "policy" | "org_default" | "tier_default"
    risk_tier: str

    max_position_pct: float
    max_sector_pct: float
    max_asset_class_pct: Optional[float]
    min_cash_pct: float
    max_cash_pct: float
    max_position_beta: Optional[float]
    max_portfolio_beta: Optional[float]
    max_portfolio_volatility: Optional[float]
    min_market_cap: float
    min_avg_dollar_volume: float
    min_position_notional: float

    target_allocation: Dict[str, float]
    drift_bands: Dict[str, float]
    allowed_asset_classes: List[str]
    excluded_tickers: List[str]
    excluded_sectors: List[str]

    lot_selection_method: str
    harvest_losses: bool
    max_short_term_gain_budget: Optional[float]

    benchmark_ticker: str
    rebalance_frequency_days: int
    approved_by_user_id: Optional[int] = None
    approved_at: Optional[Any] = None
    notes: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def is_approved(self) -> bool:
        """False when these limits are defaults nobody has signed off.

        Surfaced in the report rather than hidden: advice generated against
        unapproved defaults is still advice, but the client and the reviewer
        should know the mandate was inferred rather than agreed.
        """
        return self.source == "policy" and self.approved_by_user_id is not None

    def band_for(self, asset_class: str) -> float:
        return float(self.drift_bands.get(asset_class, DEFAULT_DRIFT_BANDS.get(asset_class, 0.05)))

    def target_for(self, asset_class: str) -> float:
        return float(self.target_allocation.get(asset_class, 0.0))

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["approved_at"] = self.approved_at.isoformat() if self.approved_at else None
        payload["is_approved"] = self.is_approved
        return payload

    def summary(self) -> str:
        origin = {
            "policy": f"approved IPS v{self.version}",
            "org_default": "firm default policy",
            "tier_default": f"built-in {self.risk_tier} defaults (no IPS on file)",
        }[self.source]
        return (
            f"Limits from {origin}: max {self.max_position_pct:.0%} per position, "
            f"{self.max_sector_pct:.0%} per sector, cash held between "
            f"{self.min_cash_pct:.0%} and {self.max_cash_pct:.0%}."
        )


def _defaults_for_tier(tier: str) -> Dict[str, Any]:
    return dict(TIER_DEFAULTS.get(tier, TIER_DEFAULTS["Moderate"]))


def get_active_policy(db: Session, client_id: int) -> Optional[InvestmentPolicy]:
    """The policy currently in force, if any."""
    return (
        db.query(InvestmentPolicy)
        .filter(
            InvestmentPolicy.client_id == client_id,
            InvestmentPolicy.status == "active",
        )
        .order_by(InvestmentPolicy.version.desc())
        .first()
    )


def get_policy_version(db: Session, client_id: int, version: int) -> Optional[InvestmentPolicy]:
    """A specific historical version.

    This is what makes a past recommendation reproducible: the run recorded a
    version number, and this returns the exact limits that number referred to.
    """
    return (
        db.query(InvestmentPolicy)
        .filter(InvestmentPolicy.client_id == client_id, InvestmentPolicy.version == version)
        .first()
    )


def resolve(db: Session, client: ClientProfile) -> ResolvedPolicy:
    """Limits in force for this client right now. Never raises.

    Resolution order: an active IPS, then the firm's default policy, then
    tier defaults. The `source` field records which applied, so a report can
    disclose that a recommendation was made against defaults nobody approved
    rather than presenting inferred limits as agreed ones.
    """
    tier = client.risk_tolerance or "Moderate"
    defaults = _defaults_for_tier(tier)
    warnings: List[str] = []

    policy = get_active_policy(db, client.id)
    if policy is not None:
        if not policy.target_allocation:
            warnings.append(
                f"IPS v{policy.version} defines no target allocation; "
                f"{tier} defaults are being used for rebalancing."
            )
        return ResolvedPolicy(
            client_id=client.id,
            version=policy.version,
            source="policy",
            risk_tier=tier,
            max_position_pct=policy.max_position_pct,
            max_sector_pct=policy.max_sector_pct,
            max_asset_class_pct=policy.max_asset_class_pct,
            min_cash_pct=policy.min_cash_pct,
            max_cash_pct=policy.max_cash_pct,
            max_position_beta=policy.max_position_beta,
            max_portfolio_beta=policy.max_portfolio_beta,
            max_portfolio_volatility=policy.max_portfolio_volatility,
            min_market_cap=to_float(policy.min_market_cap),
            min_avg_dollar_volume=to_float(policy.min_avg_dollar_volume),
            min_position_notional=to_float(policy.min_position_notional),
            target_allocation=policy.target_allocation or defaults["target_allocation"],
            drift_bands=policy.drift_bands or dict(DEFAULT_DRIFT_BANDS),
            allowed_asset_classes=policy.allowed_asset_classes or [],
            excluded_tickers=[t.upper() for t in (policy.excluded_tickers or [])],
            excluded_sectors=policy.excluded_sectors or [],
            lot_selection_method=policy.lot_selection_method,
            harvest_losses=bool(policy.harvest_losses),
            max_short_term_gain_budget=(
                to_float(policy.max_short_term_gain_budget)
                if policy.max_short_term_gain_budget is not None
                else None
            ),
            benchmark_ticker=policy.benchmark_ticker,
            rebalance_frequency_days=policy.rebalance_frequency_days,
            approved_by_user_id=policy.approved_by_user_id,
            approved_at=policy.approved_at,
            notes=policy.notes,
            warnings=warnings,
        )

    org = db.query(Organization).filter(Organization.id == client.org_id).first()
    org_defaults = (org.default_policy or {}) if org else {}
    source = "org_default" if org_defaults else "tier_default"
    if source == "tier_default":
        warnings.append(
            "No Investment Policy Statement is on file for this client. Limits below are "
            f"built-in {tier} defaults and have not been reviewed or approved."
        )

    merged = {**defaults, **org_defaults}

    return ResolvedPolicy(
        client_id=client.id,
        version=None,
        source=source,
        risk_tier=tier,
        max_position_pct=merged["max_position_pct"],
        max_sector_pct=merged["max_sector_pct"],
        max_asset_class_pct=merged.get("max_asset_class_pct"),
        min_cash_pct=merged["min_cash_pct"],
        max_cash_pct=merged["max_cash_pct"],
        max_position_beta=merged.get("max_position_beta"),
        max_portfolio_beta=merged.get("max_portfolio_beta"),
        max_portfolio_volatility=merged.get("max_portfolio_volatility"),
        min_market_cap=float(merged["min_market_cap"]),
        min_avg_dollar_volume=float(merged["min_avg_dollar_volume"]),
        min_position_notional=float(merged.get("min_position_notional", 100.0)),
        target_allocation=dict(merged["target_allocation"]),
        drift_bands=dict(merged.get("drift_bands", DEFAULT_DRIFT_BANDS)),
        allowed_asset_classes=list(merged.get("allowed_asset_classes", [])),
        excluded_tickers=[t.upper() for t in merged.get("excluded_tickers", [])],
        excluded_sectors=list(merged.get("excluded_sectors", [])),
        lot_selection_method=merged.get("lot_selection_method", "HIFO"),
        harvest_losses=bool(merged.get("harvest_losses", True)),
        max_short_term_gain_budget=merged.get("max_short_term_gain_budget"),
        benchmark_ticker=merged.get("benchmark_ticker", "SPY"),
        rebalance_frequency_days=int(merged.get("rebalance_frequency_days", 90)),
        warnings=warnings,
    )


def draft_policy(
    db: Session,
    client: ClientProfile,
    *,
    created_by_user_id: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> InvestmentPolicy:
    """Create the next draft version, seeded from what is in force today.

    Seeding from the current policy rather than from blank defaults means an
    edit changes only what the editor intended; starting from defaults would
    silently revert every prior customization the editor did not restate.
    """
    overrides = overrides or {}
    current = resolve(db, client)

    latest = (
        db.query(InvestmentPolicy)
        .filter(InvestmentPolicy.client_id == client.id)
        .order_by(InvestmentPolicy.version.desc())
        .first()
    )
    next_version = (latest.version + 1) if latest else 1

    fields: Dict[str, Any] = {
        "max_position_pct": current.max_position_pct,
        "max_sector_pct": current.max_sector_pct,
        "max_asset_class_pct": current.max_asset_class_pct,
        "min_cash_pct": current.min_cash_pct,
        "max_cash_pct": current.max_cash_pct,
        "max_position_beta": current.max_position_beta,
        "max_portfolio_beta": current.max_portfolio_beta,
        "max_portfolio_volatility": current.max_portfolio_volatility,
        "min_market_cap": current.min_market_cap,
        "min_avg_dollar_volume": current.min_avg_dollar_volume,
        "min_position_notional": current.min_position_notional,
        "target_allocation": current.target_allocation,
        "drift_bands": current.drift_bands,
        "allowed_asset_classes": current.allowed_asset_classes,
        "excluded_tickers": current.excluded_tickers,
        "excluded_sectors": current.excluded_sectors,
        "lot_selection_method": current.lot_selection_method,
        "harvest_losses": current.harvest_losses,
        "max_short_term_gain_budget": current.max_short_term_gain_budget,
        "benchmark_ticker": current.benchmark_ticker,
        "rebalance_frequency_days": current.rebalance_frequency_days,
        "notes": None,
    }
    fields.update({k: v for k, v in overrides.items() if k in fields})

    policy = InvestmentPolicy(
        org_id=client.org_id,
        client_id=client.id,
        version=next_version,
        status="draft",
        created_by_user_id=created_by_user_id,
        **fields,
    )
    db.add(policy)
    db.flush()
    return policy


def validate_policy(policy: InvestmentPolicy) -> List[str]:
    """Internal consistency checks, run before a policy can be activated.

    These catch the mistakes that would otherwise surface as a run that
    proposes nothing and cannot say why -- a minimum cash floor above the
    maximum, or a target allocation summing to 80%, are both silent
    no-recommendation conditions downstream.
    """
    problems: List[str] = []

    if not 0 < policy.max_position_pct <= 1:
        problems.append("max_position_pct must be between 0 and 1.")
    if not 0 < policy.max_sector_pct <= 1:
        problems.append("max_sector_pct must be between 0 and 1.")
    if policy.min_cash_pct >= policy.max_cash_pct:
        problems.append(
            f"min_cash_pct ({policy.min_cash_pct:.0%}) must be below max_cash_pct "
            f"({policy.max_cash_pct:.0%}); as written no cash level is permitted."
        )
    if policy.max_position_pct > policy.max_sector_pct:
        problems.append(
            f"max_position_pct ({policy.max_position_pct:.0%}) exceeds max_sector_pct "
            f"({policy.max_sector_pct:.0%}), so a single position could breach the sector "
            "limit it sits inside."
        )

    allocation = policy.target_allocation or {}
    if allocation:
        total = sum(float(v) for v in allocation.values())
        if abs(total - 1.0) > 0.01:
            problems.append(
                f"target_allocation sums to {total:.1%}, not 100%. Rebalancing against it "
                "would systematically under- or over-invest."
            )
        negative = [k for k, v in allocation.items() if float(v) < 0]
        if negative:
            problems.append(f"target_allocation has negative weights for {', '.join(negative)}.")
        cash_target = float(allocation.get("cash", 0.0))
        if cash_target < policy.min_cash_pct - 1e-9:
            problems.append(
                f"target cash ({cash_target:.0%}) is below the minimum cash floor "
                f"({policy.min_cash_pct:.0%}); rebalancing to target would breach the floor."
            )

    if policy.lot_selection_method not in ("HIFO", "FIFO", "LIFO"):
        problems.append(f"lot_selection_method {policy.lot_selection_method!r} is not supported.")

    return problems


def activate_policy(
    db: Session, policy: InvestmentPolicy, *, approved_by_user_id: int
) -> InvestmentPolicy:
    """Make a draft the policy in force, superseding the previous version.

    The prior version is marked superseded with an end date rather than
    deleted or edited. Runs that cite it must still be able to resolve exactly
    what it said.
    """
    problems = validate_policy(policy)
    if problems:
        raise ValueError("; ".join(problems))

    now = utcnow()
    previous = (
        db.query(InvestmentPolicy)
        .filter(
            InvestmentPolicy.client_id == policy.client_id,
            InvestmentPolicy.status == "active",
            InvestmentPolicy.id != policy.id,
        )
        .all()
    )
    for old in previous:
        old.status = "superseded"
        old.effective_to = now

    policy.status = "active"
    policy.effective_from = now
    policy.approved_by_user_id = approved_by_user_id
    policy.approved_at = now
    db.flush()
    logger.info(
        "Activated IPS v%d for client %d", policy.version, policy.client_id,
        extra={"client_id": policy.client_id},
    )
    return policy


def policy_from_assessment(tier: str, *, benchmark: str = "SPY") -> Dict[str, Any]:
    """Proposed policy fields for a freshly scored risk assessment.

    The output of onboarding: a questionnaire produces a tier, and a tier
    produces a concrete, editable draft mandate rather than leaving the
    advisor to invent one.
    """
    defaults = _defaults_for_tier(tier)
    return {
        **{k: v for k, v in defaults.items() if k != "target_allocation"},
        "target_allocation": dict(defaults["target_allocation"]),
        "drift_bands": dict(DEFAULT_DRIFT_BANDS),
        "benchmark_ticker": benchmark,
    }
