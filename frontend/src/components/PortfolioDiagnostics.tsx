import React from 'react';
import { PortfolioDiagnostics as IDiagnostics } from '../types';
import { Activity, AlertCircle, TrendingUp, ShieldAlert, PieChart } from 'lucide-react';

interface Props {
  diagnostics?: IDiagnostics;
}

export const PortfolioDiagnostics: React.FC<Props> = ({ diagnostics }) => {
  if (!diagnostics) return null;

  const concentration = diagnostics.concentration || {};
  const concentrationEntries = Object.entries(concentration).sort((a, b) => b[1] - a[1]);
  const flaws = diagnostics.flaws || [];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem', marginBottom: '1.5rem' }}>
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">
            <Activity size={20} color="var(--accent-primary)" />
            Portfolio Health & Diagnostics
          </h2>
          <span className="badge badge-success">Deterministic Diagnostic Gate</span>
        </div>

        <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
          <div className="stat-tile">
            <span className="stat-label">Sharpe Ratio</span>
            <span className="stat-value highlight">
              {diagnostics.sharpe_ratio !== undefined ? Number(diagnostics.sharpe_ratio).toFixed(2) : '—'}
            </span>
          </div>

          <div className="stat-tile">
            <span className="stat-label">Annual Return</span>
            <span className="stat-value">
              {diagnostics.annual_return !== undefined ? `${(Number(diagnostics.annual_return) * 100).toFixed(1)}%` : '—'}
            </span>
          </div>

          <div className="stat-tile">
            <span className="stat-label">Annual Volatility</span>
            <span className="stat-value">
              {diagnostics.annual_volatility !== undefined ? `${(Number(diagnostics.annual_volatility) * 100).toFixed(1)}%` : '—'}
            </span>
          </div>

          <div className="stat-tile">
            <span className="stat-label">Diversification Score</span>
            <span className="stat-value highlight">
              {diagnostics.diversification_score !== undefined ? `${diagnostics.diversification_score}/100` : '—'}
            </span>
          </div>
        </div>

        <div className="grid-2">
          {/* Concentration breakdown */}
          <div style={{ background: 'var(--bg-surface-elevated)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <PieChart size={16} color="var(--accent-cyan)" />
              Asset Concentration Breakdown
            </h3>
            {concentrationEntries.length > 0 ? (
              <div className="bar-container">
                {concentrationEntries.map(([symbol, fraction]) => {
                  const pct = (fraction * 100).toFixed(1);
                  return (
                    <div key={symbol} className="bar-row">
                      <span className="bar-ticker">{symbol}</span>
                      <div className="bar-track">
                        <div className="bar-fill" style={{ width: `${pct}%` }} />
                      </div>
                      <span className="bar-pct">{pct}%</span>
                    </div>
                  );
                })}
              </div>
            ) : (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No concentration data available.</p>
            )}
          </div>

          {/* Diagnosed flaws */}
          <div style={{ background: 'var(--bg-surface-elevated)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <h3 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <ShieldAlert size={16} color="var(--accent-amber)" />
              Diagnosed Structural Flaws
            </h3>
            {flaws.length > 0 ? (
              <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                {flaws.map((flaw, idx) => (
                  <li key={idx} style={{ color: '#FCD34D', fontSize: '0.85rem' }}>
                    {flaw}
                  </li>
                ))}
              </ul>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-primary)', fontSize: '0.85rem' }}>
                <TrendingUp size={18} />
                <span>No portfolio flaws detected. Portfolio adheres to risk profile.</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
