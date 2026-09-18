import React from 'react';
import { TaxAssessment } from '../types';
import { ShieldAlert, Info, AlertTriangle, CheckCircle } from 'lucide-react';

interface Props {
  taxAssessment?: TaxAssessment;
  taxBlocked?: string[];
}

export const TaxAlerts: React.FC<Props> = ({ taxAssessment, taxBlocked }) => {
  if (!taxAssessment && (!taxBlocked || taxBlocked.length === 0)) return null;

  const notes = taxAssessment?.tax_efficiency_notes || [];
  const washFlags = taxAssessment?.wash_sale_flags || [];

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-header">
        <h2 className="card-title">
          <ShieldAlert size={20} color="var(--accent-amber)" />
          Tax-Awareness & Wash-Sale Guardrails
        </h2>
        <span className="badge badge-warning">IRC § 1091 Wash-Sale Filter</span>
      </div>

      {taxBlocked && taxBlocked.length > 0 && (
        <div className="alert-banner alert-warning" style={{ marginBottom: '1rem' }}>
          <AlertTriangle size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <strong>Strictly Blocked by 30-Day IRS Wash-Sale Rule:</strong>
            <p style={{ marginTop: '0.25rem', fontSize: '0.85rem' }}>
              The following securities were excluded from recommendation allocations to protect client tax deductions:
            </p>
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem', flexWrap: 'wrap' }}>
              {taxBlocked.map((ticker) => (
                <span key={ticker} className="badge badge-danger" style={{ fontFamily: 'var(--font-mono)' }}>
                  {ticker}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {washFlags.length > 0 && (
        <div style={{ marginBottom: '1rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          <strong style={{ color: 'var(--text-primary)' }}>Active Wash-Sale Monitored Tickers:</strong>{' '}
          {washFlags.join(', ')}
        </div>
      )}

      {notes.length > 0 && (
        <div style={{ background: 'var(--bg-surface-elevated)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <h4 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Tax Optimization Notes
          </h4>
          <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
            {notes.map((note, i) => (
              <li key={i} style={{ fontSize: '0.85rem', color: 'var(--text-primary)' }}>
                {note}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
