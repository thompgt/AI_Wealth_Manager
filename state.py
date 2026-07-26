"""The graph's shared state.

Two conventions that matter more than they look:

**Only `audit_trail` is a reducer.** Every other key is last-write-wins, which
is safe only because no two concurrently-running nodes write the same key.
`audit_trail` is annotated with `operator.add` because the parallel branches
all append to it; without that, LangGraph raises on the concurrent write, and
a node returning the whole accumulated list rather than just its own record
duplicates every prior entry.

**Degradation is state, not a log line.** `degradations` records every point
where a node produced output from a fallback path rather than its intended
one. The system is written to degrade rather than crash, which means a run
built on no market data, a failed regime call and template prose is shaped
exactly like a good one. This key is what lets the report, the API response
and the operator tell them apart.
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class ClientProfile(TypedDict, total=False):
    client_id: int
    org_id: int
    name: str
    age: Optional[int]
    risk_tolerance: str  # "Conservative" | "Moderate" | "Aggressive"
    time_horizon_years: int
    goals: List[str]
    net_worth: float
    notes: Optional[str]
    # Live positions, priced. Symbol -> dollar market value, cash included as
    # "CASH". Kept as a flat map because most consumers want exposure, not
    # account detail; per-account data lives in `accounts`.
    holdings: Dict[str, float]
    accounts: List[Dict[str, Any]]
    # Symbols held but unpriceable this run. Excluded from `holdings` rather
    # than valued at zero, and named here so the report can say so.
    unpriced_symbols: List[str]


class PolicySnapshot(TypedDict, total=False):
    """The limits in force for this run, captured at the start.

    Snapshotted rather than re-resolved per node: a policy activated
    mid-run must not have some nodes enforcing the old limits and others the
    new ones. `version` is written to every audit record.
    """

    version: Optional[int]
    source: str
    risk_tier: str
    max_position_pct: float
    max_sector_pct: float
    min_cash_pct: float
    max_cash_pct: float
    min_market_cap: float
    target_allocation: Dict[str, float]
    benchmark_ticker: str
    is_approved: bool
    summary: str


class PortfolioDiagnostics(TypedDict, total=False):
    total_value: float
    invested_value: float
    cash: float
    concentration: Dict[str, float]      # symbol -> fraction of portfolio
    sector_exposure: Dict[str, float]
    asset_class_exposure: Dict[str, float]
    sharpe_ratio: Optional[float]
    annual_return: Optional[float]
    annual_volatility: Optional[float]
    max_drawdown: Optional[float]
    portfolio_beta: Optional[float]
    diversification_score: float          # 0-100, higher is better diversified
    effective_positions: Optional[float]  # inverse HHI: "really" how many holdings
    # Average pairwise correlation of the holdings. A portfolio of twelve
    # names that all move together is one position wearing twelve hats, and
    # no count of holdings reveals that.
    average_correlation: Optional[float]
    correlation_clusters: List[List[str]]
    drift: List[Dict[str, Any]]
    breaches: List[Dict[str, Any]]
    flaws: List[str]
    market_data_tickers: List[str]


class MarketRegime(TypedDict, total=False):
    regime_label: str  # "Bull" | "Bear" | "Volatile" | "Late-cycle" | "Early-cycle"
    confidence: float  # 0-1
    supporting_signals: Dict[str, Any]
    news_sentiment: Optional[float]
    narrative: str


class Candidate(TypedDict, total=False):
    """A proposed position.

    `total=False` because research emits candidates without allocation
    fields -- those are filled in downstream by the guardrail, which is the
    only component allowed to decide how many dollars a position gets.
    Marking them required would be a lie about the shape research produces.
    """

    ticker: str
    name: Optional[str]
    asset_class: str
    sector: Optional[str]
    security_type: str
    price: Optional[float]
    composite_score: Optional[float]
    factor_scores: Dict[str, Optional[float]]
    valuation_metrics: Dict[str, Any]
    correlation_to_portfolio: Optional[float]
    addresses_flaw: str
    regime_fit_rationale: str
    confidence: float
    allocation_amount: float
    allocation_pct: float
    account_id: Optional[int]


class SuitabilityResult(TypedDict, total=False):
    approved: bool
    violations: List[str]
    adjusted_recommendations: List[Candidate]
    checks_applied: List[str]


class TaxAssessment(TypedDict, total=False):
    wash_sale_flags: List[str]
    wash_sale_detail: List[Dict[str, Any]]
    harvest_candidates: List[Dict[str, Any]]
    realized_ytd: Dict[str, float]
    estimated_tax_cost: float
    tax_efficiency_notes: List[str]
    # True when the assessment could not run. Distinct from "no flags found":
    # an empty flag list and an unverifiable tax position are opposite
    # conclusions, and the guardrail gate must treat the second as blocking.
    verification_failed: bool


class RebalancePlanState(TypedDict, total=False):
    proposals: List[Dict[str, Any]]
    deferred: List[Dict[str, Any]]
    notes: List[str]
    sell_count: int
    buy_count: int
    gross_notional: float
    estimated_tax_cost: float


class Degradation(TypedDict, total=False):
    """One place where output came from a fallback rather than the real path."""

    node: str
    reason: str          # stable vocabulary, e.g. "no_api_key", "rate_limited"
    detail: str
    impact: str          # what the client-visible output loses as a result


class AgentRunRecord(TypedDict, total=False):
    node_name: str
    started_at: str  # ISO timestamp
    completed_at: Optional[str]
    status: str  # "success" | "error" | "degraded"
    summary: str
    error_detail: Optional[str]
    degraded: bool
    # Reproducibility: which model, which prompt, what it cost, what data it
    # saw. Declared but never written in the previous version, which meant the
    # audit trail could say a recommendation happened but not what produced it.
    model_used: Optional[str]
    prompt_version: Optional[str]
    temperature: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cost_usd: Optional[float]
    duration_ms: Optional[int]
    policy_version: Optional[int]
    data_provenance: Optional[Dict[str, Any]]
    input_snapshot: Optional[Dict[str, Any]]
    output_snapshot: Optional[Dict[str, Any]]


class AgentState(TypedDict, total=False):
    run_id: str
    client_profile: ClientProfile
    policy: PolicySnapshot

    portfolio_diagnostics: Optional[PortfolioDiagnostics]
    market_regime: Optional[MarketRegime]
    candidate_stocks: Optional[List[Candidate]]
    suitability_result: Optional[SuitabilityResult]
    tax_assessment: Optional[TaxAssessment]
    rebalance_plan: Optional[RebalancePlanState]
    final_report: Optional[str]

    # Appended to by every node. Annotated with operator.add so LangGraph
    # concatenates each node's returned list instead of overwriting it --
    # required because the parallel branches write this key concurrently.
    # Each node returns {"audit_trail": [its_own_record]}, never the full list.
    audit_trail: Annotated[List[AgentRunRecord], operator.add]
    degradations: Annotated[List[Degradation], operator.add]

    # Human-in-the-loop control flow
    requires_human_approval: bool
    approval_reason: Optional[str]
    human_approved: Optional[bool]
    rejection_note: Optional[str]

    # Control flow: research <-> guardrail retry loop
    research_attempts: int
    needs_research_retry: bool

    # Retry feedback, written by the guardrail gate and read by research on
    # its next pass. Without these the retry loop is a no-op: research would
    # screen an identical universe with identical inputs and deterministically
    # reproduce the same rejected picks, burning three rounds of API calls to
    # arrive back where it started.
    excluded_tickers: List[str]
    guardrail_feedback: List[str]

    # Recommendations that cleared suitability but were withheld on tax
    # grounds. Recorded so the client report can explain what was withheld and
    # why -- the withholding is the point, and an unexplained absence is
    # indistinguishable from the system not having thought of it.
    tax_blocked_recommendations: List[str]

    # Set at load time and read by every node that prices anything.
    data_as_of: Optional[str]
    llm_enabled: bool
