import React from 'react';
import { Activity, Brain, Shield, FileText, CheckCircle2, RefreshCw } from 'lucide-react';

interface PipelineProgressProps {
  currentStep?: string;
  progressPct?: number;
}

const STEPS = [
  { id: 'diagnostics', label: 'Diagnostics', icon: Activity },
  { id: 'market_regime', label: 'Market Regime', icon: Brain },
  { id: 'stock_research', label: 'Stock Research', icon: RefreshCw },
  { id: 'suitability', label: 'Suitability Guard', icon: Shield },
  { id: 'tax_awareness', label: 'Tax & Wash-Sale', icon: Shield },
  { id: 'rebalance', label: 'Rebalance Engine', icon: RefreshCw },
  { id: 'finance_report', label: 'Client Report', icon: FileText },
];

export const PipelineProgress: React.FC<PipelineProgressProps> = ({
  currentStep = 'queued',
  progressPct = 0,
}) => {
  const normalizedStep = (currentStep || '').toLowerCase();
  const activeIndex = STEPS.findIndex((s) => normalizedStep.includes(s.id));

  return (
    <div className="card" style={{ marginBottom: '1.5rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="badge badge-info">LangGraph Pipeline</span>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
            Current node: <strong style={{ color: 'var(--accent-cyan)' }}>{currentStep}</strong>
          </span>
        </div>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem', color: 'var(--accent-primary)' }}>
          {Math.round(progressPct)}% Complete
        </span>
      </div>

      <div className="pipeline-track">
        {STEPS.map((step, idx) => {
          const isCompleted = activeIndex > idx || (progressPct >= 100 && activeIndex === -1);
          const isActive = activeIndex === idx;

          let stepClass = 'pipeline-step';
          if (isCompleted) stepClass += ' completed';
          if (isActive) stepClass += ' active';

          return (
            <div key={step.id} className={stepClass}>
              <div className="step-node">
                {isCompleted ? <CheckCircle2 size={16} /> : idx + 1}
              </div>
              <span className="step-label">{step.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
};
