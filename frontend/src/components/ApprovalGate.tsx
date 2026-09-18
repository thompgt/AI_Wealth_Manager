import React, { useState } from 'react';
import { InterruptPayload, Session } from '../types';
import { AlertCircle, CheckCircle, XCircle, ShieldCheck, UserCheck } from 'lucide-react';

interface Props {
  interrupt?: InterruptPayload;
  session: Session;
  onDecide: (approved: boolean, note?: string) => Promise<void>;
  disabled?: boolean;
}

export const ApprovalGate: React.FC<Props> = ({
  interrupt,
  session,
  onDecide,
  disabled = false,
}) => {
  const [note, setNote] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const mayApprove = session.user.role === 'compliance' || session.user.role === 'admin';
  const explanation = interrupt?.explanation || interrupt?.reason || 'Human-in-the-loop review required before executing trade plan.';

  const handleAction = async (approved: boolean) => {
    setSubmitting(true);
    try {
      await onDecide(approved, note.trim() || undefined);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="card" style={{ borderColor: 'var(--accent-amber)', marginBottom: '2rem', boxShadow: '0 0 30px rgba(245, 158, 11, 0.15)' }}>
      <div className="card-header" style={{ borderColor: 'rgba(245, 158, 11, 0.3)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <div style={{ width: 36, height: 36, borderRadius: '50%', background: 'rgba(245, 158, 11, 0.2)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--accent-amber)' }}>
            <AlertCircle size={22} />
          </div>
          <div>
            <h2 className="card-title" style={{ color: 'var(--accent-amber)', fontSize: '1.15rem' }}>
              Human-in-the-Loop Review Required
            </h2>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
              Graph execution paused at approval_gate node
            </p>
          </div>
        </div>
        <span className="badge badge-warning">PENDING APPROVAL</span>
      </div>

      <div style={{ background: 'var(--bg-surface-elevated)', padding: '1.25rem', borderRadius: 'var(--radius-md)', marginBottom: '1.5rem', border: '1px solid var(--border-subtle)' }}>
        <p style={{ fontSize: '0.95rem', color: '#FCD34D', lineHeight: 1.6 }}>
          {explanation}
        </p>
      </div>

      {mayApprove ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label className="form-label" htmlFor="approval-note">
              Compliance Review Note (Optional audit comment)
            </label>
            <input
              id="approval-note"
              type="text"
              className="form-input"
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="e.g. Cleared after client risk policy verification"
              disabled={submitting || disabled}
            />
          </div>

          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <button
              id="approve-run-btn"
              type="button"
              className="btn btn-primary"
              onClick={() => handleAction(true)}
              disabled={submitting || disabled}
              style={{ padding: '0.75rem 1.5rem' }}
            >
              <CheckCircle size={18} />
              <span>Approve Recommendation</span>
            </button>

            <button
              id="reject-run-btn"
              type="button"
              className="btn btn-outline-danger"
              onClick={() => handleAction(false)}
              disabled={submitting || disabled}
              style={{ padding: '0.75rem 1.5rem' }}
            >
              <XCircle size={18} />
              <span>Reject Proposal</span>
            </button>

            {submitting && (
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Resuming LangGraph execution...
              </span>
            )}
          </div>
        </div>
      ) : (
        <div className="alert-banner alert-info" style={{ marginBottom: 0 }}>
          <UserCheck size={20} style={{ flexShrink: 0 }} />
          <div>
            <strong>Compliance Officer Approval Required</strong>
            <p style={{ marginTop: '0.25rem', fontSize: '0.85rem' }}>
              Signed in as <strong>{session.user.role}</strong> ({session.user.email}). The advisor who requested this analysis cannot approve their own recommendations. A compliance officer or administrator must review and clear this run.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
