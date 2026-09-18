import React from 'react';
import { RebalancePlan as IRebalancePlan } from '../types';
import { RefreshCw, DollarSign, ArrowUpRight, ArrowDownRight, Scale } from 'lucide-react';

interface Props {
  plan?: IRebalancePlan;
}

export const RebalancePlan: React.FC<Props> = ({ plan }) => {
  if (!plan) return null;

  const proposals = plan.proposals || [];
  const notes = plan.notes || [];

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-header">
        <h2 className="card-title">
          <Scale size={20} color="var(--accent-cyan)" />
          Tax-Aware Rebalance & Trade Plan
        </h2>
        <span className="badge badge-info">Deterministic Sizing</span>
      </div>

      <div className="grid-3" style={{ marginBottom: '1.5rem' }}>
        <div className="stat-tile">
          <span className="stat-label">Gross Notional</span>
          <span className="stat-value">
            ${(plan.gross_notional || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        <div className="stat-tile">
          <span className="stat-label">Net Cash Delta</span>
          <span className={`stat-value ${plan.net_cash_delta >= 0 ? 'highlight' : ''}`} style={{ color: plan.net_cash_delta < 0 ? 'var(--accent-danger)' : undefined }}>
            {plan.net_cash_delta >= 0 ? '+' : '-'}${Math.abs(plan.net_cash_delta || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>

        <div className="stat-tile">
          <span className="stat-label">Estimated Tax Cost</span>
          <span className="stat-value" style={{ color: (plan.estimated_tax_cost || 0) > 0 ? 'var(--accent-amber)' : 'var(--accent-primary)' }}>
            ${(plan.estimated_tax_cost || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
      </div>

      {notes.length > 0 && (
        <div style={{ marginBottom: '1rem', padding: '0.85rem', background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
          {notes.map((note, i) => (
            <p key={i} style={{ fontSize: '0.825rem', color: 'var(--text-secondary)', fontStyle: 'italic' }}>
              • {note}
            </p>
          ))}
        </div>
      )}

      {proposals.length > 0 ? (
        <div className="table-container">
          <table className="data-table" id="trade-proposals-table">
            <thead>
              <tr>
                <th style={{ width: '60px' }}>Seq</th>
                <th style={{ width: '90px' }}>Action</th>
                <th>Symbol</th>
                <th style={{ textAlign: 'right' }}>Shares</th>
                <th style={{ textAlign: 'right' }}>Notional</th>
                <th style={{ textAlign: 'right' }}>Est. Tax</th>
                <th>Trade Rationale</th>
              </tr>
            </thead>
            <tbody>
              {proposals.map((p, i) => {
                const isBuy = p.side.toUpperCase() === 'BUY';
                return (
                  <tr key={i}>
                    <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>
                      #{p.sequence || i + 1}
                    </td>
                    <td>
                      <span className={`badge ${isBuy ? 'badge-success' : 'badge-danger'}`}>
                        {isBuy ? <ArrowUpRight size={12} /> : <ArrowDownRight size={12} />}
                        {p.side}
                      </span>
                    </td>
                    <td style={{ fontWeight: 700, fontFamily: 'var(--font-mono)' }}>
                      {p.symbol}
                    </td>
                    <td className="num-cell">
                      {p.quantity !== undefined ? p.quantity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 }) : '—'}
                    </td>
                    <td className="num-cell" style={{ fontWeight: 600 }}>
                      ${(p.notional || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="num-cell" style={{ color: (p.estimated_tax_cost || 0) > 0 ? 'var(--accent-amber)' : 'var(--text-muted)' }}>
                      ${(p.estimated_tax_cost || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td style={{ fontSize: '0.825rem', color: 'var(--text-secondary)' }}>
                      {p.rationale || 'Portfolio drift and risk alignment'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          No rebalance trades required for current portfolio state.
        </p>
      )}
    </div>
  );
};
