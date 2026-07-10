from typing import Any, Dict, List, Optional, TypedDict


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


class Candidate(TypedDict):
    ticker: str
    valuation_metrics: Dict[str, float]  # e.g. {"pe_ratio": 12.4, "peg_ratio": 0.9}
    addresses_flaw: str  # which PortfolioDiagnostics.flaws entry this fixes
    regime_fit_rationale: str
    confidence: float  # 0-1


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

    # Appended to by every node
    audit_trail: List[AgentRunRecord]

    # Human-in-the-loop control flow
    requires_human_approval: bool
    human_approved: Optional[bool]

    # Control flow: Stock Research <-> Suitability/Tax-Awareness loop-back
    research_attempts: int
    needs_research_retry: bool
