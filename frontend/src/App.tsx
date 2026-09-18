import React, { useState, useEffect, useRef } from 'react';
import { Session, Client, ReportViewModel, JobState } from './types';
import {
  listClients,
  triggerRun,
  getJob,
  getReport,
  approveRun,
  formatReportViewModel,
  TERMINAL_JOB_STATES,
  ApiError,
} from './api/client';
import { LoginForm } from './components/LoginForm';
import { Header } from './components/Header';
import { Sidebar } from './components/Sidebar';
import { NewClientModal } from './components/NewClientModal';
import { PipelineProgress } from './components/PipelineProgress';
import { PortfolioDiagnostics } from './components/PortfolioDiagnostics';
import { MarketRegimeCard } from './components/MarketRegimeCard';
import { RecommendationsTable } from './components/RecommendationsTable';
import { RebalancePlan } from './components/RebalancePlan';
import { TaxAlerts } from './components/TaxAlerts';
import { ApprovalGate } from './components/ApprovalGate';
import { FullReport } from './components/FullReport';
import {
  Briefcase,
  Layers,
  Sparkles,
  TrendingUp,
  AlertCircle,
  Clock,
  ShieldCheck,
  CheckCircle2,
} from 'lucide-react';

const SESSION_STORAGE_KEY = 'ai_wealth_session';
const POLL_INTERVAL_MS = 2000;
const POLL_TIMEOUT_MS = 900000; // 15 minutes

export const App: React.FC = () => {
  const [session, setSession] = useState<Session | null>(() => {
    const saved = localStorage.getItem(SESSION_STORAGE_KEY);
    if (saved) {
      try {
        return JSON.parse(saved);
      } catch {
        return null;
      }
    }
    return null;
  });

  const [clients, setClients] = useState<Client[]>([]);
  const [selectedClientId, setSelectedClientId] = useState<number | null>(null);
  const [isNewClientOpen, setIsNewClientOpen] = useState(false);

  const [isBusy, setIsBusy] = useState(false);
  const [currentJob, setCurrentJob] = useState<JobState | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [reportResult, setReportResult] = useState<ReportViewModel | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pollTimerRef = useRef<number | null>(null);

  const saveSession = (newSession: Session | null) => {
    setSession(newSession);
    if (newSession) {
      localStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(newSession));
    } else {
      localStorage.removeItem(SESSION_STORAGE_KEY);
      setClients([]);
      setSelectedClientId(null);
      setReportResult(null);
    }
  };

  // Load clients when session changes
  useEffect(() => {
    if (!session) return;
    listClients(session.token)
      .then((data) => {
        setClients(data);
        if (data.length > 0 && !selectedClientId) {
          setSelectedClientId(data[0].id);
        }
      })
      .catch((err) => {
        console.error('Failed to load clients:', err);
      });
  }, [session]);

  // Clean up timer on unmount
  useEffect(() => {
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
      }
    };
  }, []);

  const handleRunAnalysis = async () => {
    if (!session || !selectedClientId || isBusy) return;

    setError(null);
    setReportResult(null);
    setActiveRunId(null);
    setIsBusy(true);

    try {
      const runJob = await triggerRun(session.token, selectedClientId);
      const jobId = runJob.job_id;

      setCurrentJob({
        job_id: jobId,
        status: 'queued',
        current_step: 'queued',
        progress_pct: 0,
      });

      const startTime = Date.now();

      pollTimerRef.current = window.setInterval(async () => {
        try {
          if (Date.now() - startTime > POLL_TIMEOUT_MS) {
            if (pollTimerRef.current) clearInterval(pollTimerRef.current);
            setIsBusy(false);
            setError('Analysis run exceeded timeout. Check worker logs.');
            return;
          }

          const job = await getJob(session.token, jobId);
          setCurrentJob(job);

          if (TERMINAL_JOB_STATES.includes(job.status)) {
            if (pollTimerRef.current) clearInterval(pollTimerRef.current);
            setIsBusy(false);

            if (job.status !== 'succeeded') {
              setError(job.error || `Run failed with status: ${job.status}`);
              return;
            }

            const payload = job.result || {};
            setActiveRunId(payload.run_id);

            if (payload.status === 'pending_approval') {
              setReportResult({
                status: 'pending_approval',
                run_id: payload.run_id,
                interrupt: payload.interrupt,
                rebalance_plan: payload.interrupt?.rebalance_plan,
                suitability_result: payload.interrupt?.suitability_result,
                tax_assessment: payload.interrupt?.tax_assessment,
                tax_blocked_recommendations: payload.interrupt?.tax_blocked_recommendations,
                market_regime: payload.interrupt?.market_regime,
              });
              return;
            }

            if (payload.report_id) {
              const fullReport = await getReport(session.token, payload.report_id);
              setReportResult(formatReportViewModel(fullReport));
            } else {
              setReportResult(payload);
            }
          }
        } catch (err: any) {
          if (pollTimerRef.current) clearInterval(pollTimerRef.current);
          setIsBusy(false);
          setError(err.message || 'Error polling analysis job');
        }
      }, POLL_INTERVAL_MS);
    } catch (err: any) {
      setIsBusy(false);
      setError(err.message || 'Failed to trigger analysis run');
    }
  };

  const handleApprovalDecide = async (approved: boolean, note?: string) => {
    if (!session || !activeRunId) return;
    setError(null);

    try {
      const outcome = await approveRun(session.token, activeRunId, approved, note);
      const reportId = outcome?.report_id;
      if (reportId) {
        const fullReport = await getReport(session.token, reportId);
        setReportResult(formatReportViewModel(fullReport));
      } else {
        setReportResult(outcome);
      }
    } catch (err: any) {
      setError(err.message || 'Failed to submit approval decision');
    }
  };

  const handleClientCreated = (newClient: Client) => {
    setClients((prev) => [...prev, newClient]);
    setSelectedClientId(newClient.id);
    setReportResult(null);
  };

  if (!session) {
    return <LoginForm onSession={saveSession} />;
  }

  const activeClient = clients.find((c) => c.id === selectedClientId);

  return (
    <div className="app-container">
      <Sidebar
        session={session}
        clients={clients}
        selectedClientId={selectedClientId}
        onSelectClient={(id) => {
          setSelectedClientId(id);
          setReportResult(null);
          setError(null);
        }}
        onOpenNewClient={() => setIsNewClientOpen(true)}
        onRunAnalysis={handleRunAnalysis}
        onSignOut={() => saveSession(null)}
        isBusy={isBusy}
      />

      <div className="main-content">
        <Header session={session} clientCount={clients.length} />

        <main className="content-body">
          {error && (
            <div className="alert-banner alert-danger">
              <AlertCircle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
              <div>
                <strong>Operational Error:</strong> {error}
              </div>
            </div>
          )}

          {/* Active Job Progress */}
          {isBusy && (
            <PipelineProgress
              currentStep={currentJob?.current_step || currentJob?.status}
              progressPct={currentJob?.progress_pct || 0}
            />
          )}

          {/* Result View */}
          {reportResult ? (
            <div>
              {/* Human in the loop gate banner */}
              {reportResult.status === 'pending_approval' && (
                <ApprovalGate
                  interrupt={reportResult.interrupt}
                  session={session}
                  onDecide={handleApprovalDecide}
                />
              )}

              {/* Status Header */}
              {reportResult.status === 'completed' && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.5rem', background: 'var(--bg-surface)', padding: '1rem 1.5rem', borderRadius: 'var(--radius-lg)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                    <CheckCircle2 size={24} color="var(--accent-primary)" />
                    <div>
                      <h2 style={{ fontSize: '1.15rem', fontWeight: 600 }}>Portfolio Analysis Complete</h2>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                        All 7 specialist agent nodes finished with deterministic compliance verification.
                      </p>
                    </div>
                  </div>
                  <span className="badge badge-success">COMPLETED</span>
                </div>
              )}

              {/* Portfolio Diagnostics */}
              <PortfolioDiagnostics diagnostics={reportResult.portfolio_diagnostics} />

              {/* Market Regime */}
              <MarketRegimeCard regime={reportResult.market_regime} />

              {/* Suitability Recommendations */}
              <RecommendationsTable suitability={reportResult.suitability_result} />

              {/* Tax & Wash-Sale Alerts */}
              <TaxAlerts
                taxAssessment={reportResult.tax_assessment}
                taxBlocked={reportResult.tax_blocked_recommendations}
              />

              {/* Rebalance Plan */}
              <RebalancePlan plan={reportResult.rebalance_plan} />

              {/* Client Memo Full Report */}
              <FullReport
                reportText={reportResult.final_report}
                reportId={reportResult.report_id}
                approvalState={reportResult.approval_state}
                llmEnabled={reportResult.llm_enabled}
              />
            </div>
          ) : activeClient ? (
            /* Selected Client Profile & Holdings */
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              <div className="card">
                <div className="card-header">
                  <div>
                    <h1 style={{ fontSize: '1.4rem', fontWeight: 700, color: '#FFFFFF' }}>
                      {activeClient.name}
                    </h1>
                    <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                      Client File #{activeClient.id} • Registered Account Directory
                    </p>
                  </div>
                  <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <span className="badge badge-info">{activeClient.risk_tolerance} RISK</span>
                    <span className="badge badge-success">{activeClient.time_horizon_years} YR HORIZON</span>
                  </div>
                </div>

                <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
                  <div className="stat-tile">
                    <span className="stat-label">Total Net Worth</span>
                    <span className="stat-value highlight">
                      ${(activeClient.net_worth || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </span>
                  </div>

                  <div className="stat-tile">
                    <span className="stat-label">Client Age</span>
                    <span className="stat-value">{activeClient.age} yrs</span>
                  </div>

                  <div className="stat-tile">
                    <span className="stat-label">Risk Tier</span>
                    <span className="stat-value" style={{ textTransform: 'capitalize', color: 'var(--accent-cyan)' }}>
                      {activeClient.risk_tolerance}
                    </span>
                  </div>

                  <div className="stat-tile">
                    <span className="stat-label">Investment Horizon</span>
                    <span className="stat-value">{activeClient.time_horizon_years} yrs</span>
                  </div>
                </div>

                {/* Stated Goals */}
                {activeClient.goals && activeClient.goals.length > 0 && (
                  <div style={{ marginBottom: '1.5rem', background: 'var(--bg-surface-elevated)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                    <span style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-muted)', fontWeight: 600 }}>
                      Stated Client Goals:
                    </span>
                    <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
                      {activeClient.goals.map((g, i) => (
                        <span key={i} className="badge badge-info" style={{ background: 'rgba(6, 182, 212, 0.1)' }}>
                          {g}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Accounts & Holdings Table */}
                <h3 style={{ fontSize: '1rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Briefcase size={18} color="var(--accent-primary)" />
                  Portfolio Accounts & Current Lots
                </h3>

                {activeClient.accounts && activeClient.accounts.length > 0 ? (
                  activeClient.accounts.map((acc, aIdx) => (
                    <div key={aIdx} style={{ marginBottom: '1.5rem', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', overflow: 'hidden' }}>
                      <div style={{ padding: '0.85rem 1rem', background: 'var(--bg-surface-elevated)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div>
                          <strong style={{ color: '#FFFFFF' }}>{acc.name}</strong>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginLeft: '0.75rem' }}>
                            Type: <span style={{ textTransform: 'capitalize' }}>{acc.account_type.replace('_', ' ')}</span> • Tax: {acc.tax_treatment}
                          </span>
                        </div>
                        <div style={{ fontSize: '0.85rem', color: 'var(--accent-primary)', fontWeight: 600 }}>
                          Cash: ${(acc.cash_balance || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                        </div>
                      </div>

                      {acc.holdings && acc.holdings.length > 0 ? (
                        <table className="data-table">
                          <thead>
                            <tr>
                              <th>Symbol</th>
                              <th style={{ textAlign: 'right' }}>Shares</th>
                              <th style={{ textAlign: 'right' }}>Cost Basis / Share</th>
                              <th style={{ textAlign: 'right' }}>Total Cost</th>
                            </tr>
                          </thead>
                          <tbody>
                            {acc.holdings.map((h, hIdx) => (
                              <tr key={hIdx}>
                                <td style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                                  {h.symbol}
                                </td>
                                <td className="num-cell">
                                  {h.quantity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 })}
                                </td>
                                <td className="num-cell">
                                  ${h.cost_per_share.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </td>
                                <td className="num-cell" style={{ fontWeight: 600 }}>
                                  ${(h.quantity * h.cost_per_share).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      ) : (
                        <div style={{ padding: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                          Account currently holds 100% uninvested cash.
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No accounts registered under this client.</p>
                )}

                <div style={{ marginTop: '1.5rem', display: 'flex', justifyContent: 'flex-end' }}>
                  <button
                    id="trigger-analysis-banner-btn"
                    type="button"
                    className="btn btn-primary"
                    onClick={handleRunAnalysis}
                    disabled={isBusy}
                  >
                    <TrendingUp size={16} />
                    <span>Run Multi-Agent Analysis for {activeClient.name}</span>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* No Client Selected Empty State */
            <div className="card" style={{ textAlign: 'center', padding: '4rem 2rem' }}>
              <div style={{ width: 64, height: 64, borderRadius: '50%', background: 'rgba(16, 185, 129, 0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem', color: 'var(--accent-primary)' }}>
                <Layers size={32} />
              </div>
              <h2 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.5rem' }}>
                Welcome to AI Wealth Manager
              </h2>
              <p style={{ color: 'var(--text-secondary)', maxWidth: '500px', margin: '0 auto 1.5rem', fontSize: '0.95rem' }}>
                Select a client from the sidebar or onboard a new client to initiate the 7-agent portfolio research and compliance pipeline.
              </p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => setIsNewClientOpen(true)}
              >
                Onboard New Client
              </button>
            </div>
          )}
        </main>
      </div>

      <NewClientModal
        session={session}
        isOpen={isNewClientOpen}
        onClose={() => setIsNewClientOpen(false)}
        onCreated={handleClientCreated}
      />
    </div>
  );
};
