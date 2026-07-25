import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class ClientProfile(TypedDict):
    client_id: int
    age: int
    risk_tolerance: str  # "Conservative" | "Moderate" | "Aggressive"
    time_horizon_years: int
    goals: List[str]
    holdings: Dict[str, float]  # symbol -> current market value
    net_worth: float


class PortfolioDiagnostics(TypedDict):
    concentration: Dict[str, float]  # symbol -> fraction of portfolio
    sector_exposure: Dict[str, float]
    sharpe_ratio: float
    annual_return: float
    annual_volatility: float
    diversification_score: float  # 0-100, higher is better diversified
    flaws: List[str]  # plain-language findings, e.g. "42% concentrated in AAPL"
    market_data_tickers: List[str]  # tickers that had usable price data


class MarketRegime(TypedDict):
    regime_label: str  # "Bull" | "Bear" | "Volatile" | "Late-cycle" | "Early-cycle"
    confidence: float  # 0-1
    supporting_signals: Dict[str, Any]  # ticker/ratio -> observed value
    narrative: str


class Candidate(TypedDict, total=False):
    """A proposed position.

    `total=False` because the Stock Research agent emits candidates without
    the allocation fields -- those are filled in downstream by the
    Suitability guardrail, which is the only component allowed to decide how
    many dollars a position gets. Marking them required would be a lie about
    the shape Stock Research actually produces.
    """

    ticker: str
    valuation_metrics: Dict[str, float]  # e.g. {"pe_ratio": 12.4, "peg_ratio": 0.9}
    addresses_flaw: str  # which PortfolioDiagnostics.flaws entry this fixes
    regime_fit_rationale: str
    confidence: float  # 0-1
    # Set by the Suitability/Compliance Guardrail agent: the actual dollar
    # amount recommended for this position, and that amount as a fraction of
    # the client's net worth. Sized from available CASH, weighted by
    # confidence, and capped by the client's risk-tier position limit --
    # see agents/suitability.py::_compute_allocations.
    allocation_amount: float
    allocation_pct: float


class SuitabilityResult(TypedDict):
    approved: bool
    violations: List[str]
    adjusted_recommendations: List[Candidate]


class TaxAssessment(TypedDict):
    wash_sale_flags: List[str]  # tickers blocked by the 30-day wash-sale rule
    tax_efficiency_notes: List[str]


class AgentRunRecord(TypedDict):
    node_name: str
    started_at: str  # ISO timestamp
    completed_at: Optional[str]
    status: str  # "success" | "error" | "interrupted"
    summary: str
    error_detail: Optional[str]


class AgentState(TypedDict):
    run_id: str
    client_profile: ClientProfile

    # Set by Portfolio Diagnostics Agent
    portfolio_diagnostics: Optional[PortfolioDiagnostics]

    # Set by Market Regime Agent
    market_regime: Optional[MarketRegime]

    # Set by Stock Research Agent
    candidate_stocks: Optional[List[Candidate]]

    # Set by Suitability/Compliance Guardrail
    suitability_result: Optional[SuitabilityResult]

    # Set by Tax-Awareness Agent
    tax_assessment: Optional[TaxAssessment]

    # Set by Finance Report Agent
    final_report: Optional[str]

    # Appended to by every node. Annotated with operator.add so LangGraph
    # concatenates each node's returned list instead of overwriting it --
    # required because Portfolio Diagnostics/Market Regime and
    # Suitability/Tax-Awareness write this key concurrently in parallel
    # branches. Each node should return {"audit_trail": [its_own_record]},
    # never the full accumulated list.
    audit_trail: Annotated[List[AgentRunRecord], operator.add]

    # Human-in-the-loop control flow
    requires_human_approval: bool
    human_approved: Optional[bool]

    # Control flow: Stock Research <-> Suitability/Tax-Awareness loop-back
    research_attempts: int
    needs_research_retry: bool

    # Retry feedback, written by the guardrail gate and read by Stock
    # Research on its next pass. Without these the retry loop is a no-op:
    # Stock Research would re-run against an identical universe with
    # identical inputs and deterministically reproduce the same rejected
    # picks, burning three rounds of API calls to arrive back where it
    # started.
    excluded_tickers: List[str]  # tickers already rejected this run -- never propose again
    guardrail_feedback: List[str]  # why they were rejected, fed into the research prompt

    # Set by the guardrail gate after cross-checking the Suitability result
    # against the Tax-Awareness assessment. Recommendations that survive
    # suitability but violate the wash-sale rule are removed here; this
    # records what was removed so the client report can explain it.
    tax_blocked_recommendations: List[str]
