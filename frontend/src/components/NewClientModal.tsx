import React, { useState, useEffect } from 'react';
import { QuestionnaireSchema, Session, Client } from '../types';
import { getQuestionnaire, createClient, submitRiskAssessment, getClient, ApiError } from '../api/client';
import { X, Plus, Trash2, UserPlus, HelpCircle } from 'lucide-react';

interface Props {
  session: Session;
  isOpen: boolean;
  onClose: () => void;
  onCreated: (client: Client) => void;
}

interface HoldingRow {
  symbol: string;
  quantity: number;
  cost_per_share: number;
}

const ACCOUNT_TYPES = [
  { value: 'individual', label: 'Individual Taxable', tax: 'taxable' },
  { value: 'joint', label: 'Joint Taxable', tax: 'taxable' },
  { value: 'traditional_ira', label: 'Traditional IRA (Tax-Deferred)', tax: 'tax_deferred' },
  { value: 'roth_ira', label: 'Roth IRA (Tax-Exempt)', tax: 'tax_exempt' },
  { value: '401k', label: '401(k) (Tax-Deferred)', tax: 'tax_deferred' },
  { value: 'trust', label: 'Trust Taxable', tax: 'taxable' },
  { value: 'custodial', label: 'Custodial (Taxable)', tax: 'taxable' },
];

export const NewClientModal: React.FC<Props> = ({
  session,
  isOpen,
  onClose,
  onCreated,
}) => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [age, setAge] = useState(45);
  const [horizon, setHorizon] = useState(15);
  const [goalsText, setGoalsText] = useState('retirement, wealth preservation');
  const [accountType, setAccountType] = useState('individual');
  const [cash, setCash] = useState(100000);
  const [holdings, setHoldings] = useState<HoldingRow[]>([
    { symbol: '', quantity: 0, cost_per_share: 0 },
  ]);

  const [schema, setSchema] = useState<QuestionnaireSchema | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [loadingSchema, setLoadingSchema] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedAccount = ACCOUNT_TYPES.find((a) => a.value === accountType) || ACCOUNT_TYPES[0];
  const taxTreatment = selectedAccount.tax;

  useEffect(() => {
    if (isOpen) {
      setLoadingSchema(true);
      setError(null);
      getQuestionnaire(session.token)
        .then((q) => {
          setSchema(q);
          // initialize default answers if questions exist
          const initialAnswers: Record<string, string> = {};
          q.questions?.forEach((qu) => {
            if (qu.options && qu.options.length > 0) {
              initialAnswers[qu.id] = qu.options[0].id;
            }
          });
          setAnswers(initialAnswers);
        })
        .catch((err) => {
          setError(err.message || 'Failed to load risk questionnaire');
        })
        .finally(() => setLoadingSchema(false));
    }
  }, [isOpen, session.token]);

  if (!isOpen) return null;

  const handleAddHolding = () => {
    setHoldings([...holdings, { symbol: '', quantity: 0, cost_per_share: 0 }]);
  };

  const handleRemoveHolding = (index: number) => {
    setHoldings(holdings.filter((_, i) => i !== index));
  };

  const handleHoldingChange = (index: number, field: keyof HoldingRow, value: any) => {
    const updated = [...holdings];
    updated[index] = { ...updated[index], [field]: value };
    setHoldings(updated);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setError('Client name is required.');
      return;
    }

    const expectedQids = (schema?.questions || []).map((q) => q.id);
    const missing = expectedQids.filter((qid) => !answers[qid]);
    if (missing.length > 0) {
      setError('Please answer all questions in the risk tolerance questionnaire.');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const validHoldings = holdings
        .filter((h) => h.symbol.trim() && h.quantity > 0)
        .map((h) => ({
          symbol: h.symbol.trim().toUpperCase(),
          quantity: Number(h.quantity),
          cost_per_share: Number(h.cost_per_share),
        }));

      const holdingsVal = validHoldings.reduce((sum, h) => sum + h.quantity * h.cost_per_share, 0);
      const totalNetWorth = Number(cash) + holdingsVal;

      const payload = {
        name: name.trim(),
        email: email.trim() || null,
        age: Number(age),
        time_horizon_years: Number(horizon),
        goals: goalsText.split(',').map((g) => g.trim()).filter(Boolean),
        net_worth: totalNetWorth,
        accounts: [
          {
            name: `${name.trim()} ${selectedAccount.label}`,
            account_type: accountType,
            tax_treatment: taxTreatment,
            cash_balance: Number(cash),
            holdings: validHoldings,
          },
        ],
      };

      const created = await createClient(session.token, payload);
      await submitRiskAssessment(session.token, created.id, answers);
      const refreshed = await getClient(session.token, created.id);

      onCreated(refreshed);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to create client.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem', paddingBottom: '0.75rem', borderBottom: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <UserPlus size={20} color="var(--accent-primary)" />
            <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Onboard New Client</h2>
          </div>
          <button type="button" className="btn btn-ghost" onClick={onClose} style={{ padding: '0.35rem' }}>
            <X size={18} />
          </button>
        </div>

        {error && (
          <div className="alert-banner alert-danger">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="grid-2">
            <div className="form-group">
              <label className="form-label" htmlFor="new-client-name">Full Name *</label>
              <input
                id="new-client-name"
                type="text"
                className="form-input"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Sarah Jenkins"
                required
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="new-client-email">Email Address</label>
              <input
                id="new-client-email"
                type="email"
                className="form-input"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="sarah@example.com"
              />
            </div>
          </div>

          <div className="grid-2">
            <div className="form-group">
              <label className="form-label" htmlFor="new-client-age">Age</label>
              <input
                id="new-client-age"
                type="number"
                min="18"
                max="120"
                className="form-input"
                value={age}
                onChange={(e) => setAge(Number(e.target.value))}
              />
            </div>

            <div className="form-group">
              <label className="form-label" htmlFor="new-client-horizon">Time Horizon (Years)</label>
              <input
                id="new-client-horizon"
                type="number"
                min="1"
                max="60"
                className="form-input"
                value={horizon}
                onChange={(e) => setHorizon(Number(e.target.value))}
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="new-client-goals">Financial Goals (Comma-separated)</label>
            <input
              id="new-client-goals"
              type="text"
              className="form-input"
              value={goalsText}
              onChange={(e) => setGoalsText(e.target.value)}
              placeholder="Retirement, capital preservation, tax reduction"
            />
          </div>

          {/* Dynamic Risk Questionnaire */}
          <div style={{ margin: '1.5rem 0', padding: '1.25rem', background: 'var(--bg-surface-elevated)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 600, color: 'var(--accent-cyan)' }}>
                Suitability Risk Questionnaire
              </h3>
              {schema && (
                <span className="badge badge-info" style={{ fontSize: '0.7rem' }}>
                  v{schema.version} (Valid {schema.valid_days}d)
                </span>
              )}
            </div>

            {loadingSchema ? (
              <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>Loading versioned questionnaire...</p>
            ) : schema?.questions ? (
              schema.questions.map((q) => (
                <div key={q.id} className="form-group" style={{ marginBottom: '1rem' }}>
                  <label className="form-label" style={{ color: 'var(--text-primary)', fontWeight: 500 }}>
                    {q.prompt}
                  </label>
                  <select
                    className="form-select"
                    value={answers[q.id] || ''}
                    onChange={(e) => setAnswers({ ...answers, [q.id]: e.target.value })}
                  >
                    {q.options.map((opt) => (
                      <option key={opt.id} value={opt.id}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                  {q.help_text && (
                    <small style={{ color: 'var(--text-muted)', marginTop: '2px', display: 'block' }}>
                      {q.help_text}
                    </small>
                  )}
                </div>
              ))
            ) : null}
          </div>

          {/* Account Configuration */}
          <div style={{ margin: '1.5rem 0' }}>
            <h3 style={{ fontSize: '0.95rem', fontWeight: 600, marginBottom: '0.75rem' }}>
              Primary Account Setup
            </h3>
            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">Account Type</label>
                <select
                  className="form-select"
                  value={accountType}
                  onChange={(e) => setAccountType(e.target.value)}
                >
                  {ACCOUNT_TYPES.map((t) => (
                    <option key={t.value} value={t.value}>
                      {t.label}
                    </option>
                  ))}
                </select>
                <small style={{ color: 'var(--text-secondary)', marginTop: '2px', display: 'block' }}>
                  Tax treatment: <strong style={{ color: 'var(--accent-primary)' }}>{taxTreatment}</strong> (automatic)
                </small>
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="new-client-cash">Uninvested Cash ($)</label>
                <input
                  id="new-client-cash"
                  type="number"
                  min="0"
                  step="1000"
                  className="form-input"
                  value={cash}
                  onChange={(e) => setCash(Number(e.target.value))}
                />
              </div>
            </div>
          </div>

          {/* Initial Holdings */}
          <div style={{ margin: '1.5rem 0' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <h3 style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                Initial Holdings (Optional)
              </h3>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={handleAddHolding}
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.75rem' }}
              >
                <Plus size={14} /> Add Holding
              </button>
            </div>

            {holdings.map((h, idx) => (
              <div key={idx} style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '0.5rem' }}>
                <input
                  type="text"
                  placeholder="Ticker (e.g. AAPL)"
                  className="form-input"
                  style={{ textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}
                  value={h.symbol}
                  onChange={(e) => handleHoldingChange(idx, 'symbol', e.target.value)}
                />
                <input
                  type="number"
                  placeholder="Shares"
                  className="form-input"
                  min="0"
                  step="0.01"
                  value={h.quantity || ''}
                  onChange={(e) => handleHoldingChange(idx, 'quantity', Number(e.target.value))}
                />
                <input
                  type="number"
                  placeholder="Cost Basis ($)"
                  className="form-input"
                  min="0"
                  step="0.01"
                  value={h.cost_per_share || ''}
                  onChange={(e) => handleHoldingChange(idx, 'cost_per_share', Number(e.target.value))}
                />
                {holdings.length > 1 && (
                  <button
                    type="button"
                    className="btn btn-ghost"
                    onClick={() => handleRemoveHolding(idx)}
                    style={{ padding: '0.5rem', color: 'var(--accent-danger)' }}
                  >
                    <Trash2 size={16} />
                  </button>
                )}
              </div>
            ))}
          </div>

          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '1rem', marginTop: '1.5rem' }}>
            <button type="button" className="btn btn-secondary" onClick={onClose} disabled={submitting}>
              Cancel
            </button>
            <button id="create-client-btn" type="submit" className="btn btn-primary" disabled={submitting}>
              {submitting ? 'Creating Client...' : 'Create Client File'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
