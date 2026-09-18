import React from 'react';
import { MarketRegime as IMarketRegime } from '../types';
import { Brain, Compass, Award } from 'lucide-react';

interface Props {
  regime?: IMarketRegime;
}

export const MarketRegimeCard: React.FC<Props> = ({ regime }) => {
  if (!regime) return null;

  const label = regime.regime_label || 'Neutral';
  const confidence = regime.confidence !== undefined ? Math.round(regime.confidence * 100) : 0;

  let badgeClass = 'badge-info';
  if (label.toLowerCase().includes('bull')) badgeClass = 'badge-success';
  if (label.toLowerCase().includes('bear')) badgeClass = 'badge-danger';
  if (label.toLowerCase().includes('volat')) badgeClass = 'badge-warning';

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div className="card-header">
        <h2 className="card-title">
          <Brain size={20} color="var(--accent-cyan)" />
          Macroeconomic Market Regime
        </h2>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <span className={`badge ${badgeClass}`}>{label.toUpperCase()}</span>
          <span className="badge badge-info" style={{ fontFamily: 'var(--font-mono)' }}>
            Confidence: {confidence}%
          </span>
        </div>
      </div>

      <div style={{ background: 'var(--bg-surface-elevated)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
        <p style={{ color: 'var(--text-primary)', fontSize: '0.925rem', lineHeight: 1.6 }}>
          {regime.narrative || 'No macroeconomic regime narrative available for this cycle.'}
        </p>
      </div>
    </div>
  );
};
