export type Role = 'viewer' | 'advisor' | 'compliance' | 'admin';

export interface User {
  id?: number;
  email: string;
  role: Role;
  org_id?: number;
  org_slug?: string;
}

export interface Session {
  token: string;
  user: User;
}

export interface Holding {
  id?: number;
  symbol: string;
  quantity: number;
  cost_per_share: number;
  current_price?: number;
  market_value?: number;
}

export interface Account {
  id?: number;
  name: string;
  account_type: string;
  tax_treatment: string;
  cash_balance: number;
  holdings: Holding[];
}

export interface Client {
  id: number;
  org_id?: number;
  name: string;
  email?: string;
  age: number;
  time_horizon_years: number;
  risk_tolerance: string;
  goals: string[];
  net_worth: number;
  accounts?: Account[];
  created_at?: string;
}

export interface QuestionOption {
  id: string;
  label: string;
  score?: number;
}

export interface Question {
  id: string;
  prompt: string;
  help_text?: string;
  options: QuestionOption[];
}

export interface QuestionnaireSchema {
  version: string;
  valid_days: number;
  questions: Question[];
}

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'timed_out';

export interface JobState {
  job_id: string;
  status: JobStatus;
  current_step?: string;
  progress_pct?: number;
  result?: any;
  error?: string;
  correlation_id?: string;
}

export interface PortfolioDiagnostics {
  sharpe_ratio?: number;
  annual_return?: number;
  annual_volatility?: number;
  diversification_score?: number;
  concentration?: Record<string, number>;
  flaws?: string[];
}

export interface MarketRegime {
  regime_label: string;
  confidence: number;
  narrative: string;
  indicators?: Record<string, any>;
}

export interface Recommendation {
  ticker: string;
  allocation_amount: number;
  allocation_pct: number;
  addresses_flaw?: string;
  regime_fit_rationale?: string;
  confidence?: number;
}

export interface SuitabilityResult {
  approved: boolean;
  violations?: string[];
  adjusted_recommendations?: Recommendation[];
}

export interface TaxAssessment {
  wash_sale_flags?: string[];
  tax_efficiency_notes?: string[];
  estimated_realized_gain?: number;
}

export interface TradeProposal {
  sequence: number;
  side: 'BUY' | 'SELL' | string;
  symbol: string;
  quantity?: number;
  notional: number;
  estimated_tax_cost?: number;
  rationale?: string;
}

export interface RebalancePlan {
  gross_notional: number;
  net_cash_delta: number;
  estimated_tax_cost: number;
  proposals: TradeProposal[];
  notes?: string[];
}

export interface InterruptPayload {
  reason?: string;
  explanation?: string;
  rebalance_plan?: RebalancePlan;
  suitability_result?: SuitabilityResult;
  tax_assessment?: TaxAssessment;
  tax_blocked_recommendations?: string[];
  market_regime?: MarketRegime;
}

export interface ReportViewModel {
  status: 'completed' | 'pending_approval';
  run_id?: string;
  report_id?: number;
  llm_enabled?: boolean;
  final_report?: string;
  approval_state?: 'pending' | 'approved' | 'rejected';
  degraded?: boolean;
  portfolio_diagnostics?: PortfolioDiagnostics;
  market_regime?: MarketRegime;
  suitability_result?: SuitabilityResult;
  tax_assessment?: TaxAssessment;
  tax_blocked_recommendations?: string[];
  rebalance_plan?: RebalancePlan;
  interrupt?: InterruptPayload;
}
