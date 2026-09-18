import React from 'react';
import { SuitabilityResult } from '../types';
import { ShieldCheck, AlertTriangle, Target, Check } from 'lucide-react';

interface Props {
  suitability?: SuitabilityResult;
}

export const RecommendationsTable: React.FC<Props> = ({ suitability }) => {
  if (!suitability) return null;

  const recs = suitability.adjusted_recommendations || [];
  const totalAllocation = recs.reduce((acc, r) => acc + (r.allocation_amount || 0), 0);
  const violations = suitability.violations || [];

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-header">
        <h2 className="card-title">
          <Target size={20} color="var(--accent-primary)" />
          Suitability-Screened Security Recommendations
        </h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          {suitability.approved ? (
            <span className="badge badge-success">
              <Check size={14} /> IPS Approved
            </span>
          ) : (
            <span className="badge badge-warning">
              <AlertTriangle size={14} /> Review Required
            </span>
          )}
        </div>
      </div>

      {violations.length > 0 && (
        <div className="alert-banner alert-warning">
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>Suitability Caps & Policy Violations Filtered:</strong>
            <ul style={{ paddingLeft: '1.25rem', marginTop: '0.35rem' }}>
              {violations.map((v, i) => (
                <li key={i}>{v}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {recs.length > 0 ? (
        <>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
              Candidate Securities Cleared by Deterministic Risk Caps
            </span>
            <span style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              Total Sizing: <strong style={{ color: 'var(--accent-primary)', fontFamily: 'var(--font-mono)' }}>${totalAllocation.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}</strong>
            </span>
          </div>

          <div className="table-container">
            <table className="data-table" id="recommendations-table">
              <thead>
                <tr>
                  <th>Ticker</th>
                  <th style={{ textAlign: 'right' }}>Allocation ($)</th>
                  <th style={{ textAlign: 'right' }}>Allocation (%)</th>
                  <th>Addresses Flaw</th>
                  <th>Regime Fit Rationale</th>
                  <th style={{ textAlign: 'right' }}>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {recs.map((r, i) => (
                  <tr key={i}>
                    <td style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', color: 'var(--accent-cyan)' }}>
                      {r.ticker}
                    </td>
                    <td className="num-cell" style={{ fontWeight: 600 }}>
                      ${(r.allocation_amount || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                    </td>
                    <td className="num-cell">
                      {((r.allocation_pct || 0) * 100).toFixed(1)}%
                    </td>
                    <td style={{ color: 'var(--accent-amber)', fontSize: '0.85rem' }}>
                      {r.addresses_flaw || '—'}
                    </td>
                    <td style={{ fontSize: '0.85rem', maxWidth: '350px' }}>
                      {r.regime_fit_rationale || '—'}
                    </td>
                    <td className="num-cell" style={{ color: 'var(--text-secondary)' }}>
                      {r.confidence !== undefined ? `${Math.round(r.confidence * 100)}%` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
          No candidate securities were approved for allocation this cycle.
        </p>
      )}
    </div>
  );
};
