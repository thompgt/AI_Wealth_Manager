import React from 'react';
import { Client, Session } from '../types';
import { ShieldCheck, UserPlus, Play, LogOut, ChevronRight, Users, User, DollarSign } from 'lucide-react';

interface Props {
  session: Session;
  clients: Client[];
  selectedClientId: number | null;
  onSelectClient: (id: number) => void;
  onOpenNewClient: () => void;
  onRunAnalysis: () => void;
  onSignOut: () => void;
  isBusy: boolean;
}

export const Sidebar: React.FC<Props> = ({
  session,
  clients,
  selectedClientId,
  onSelectClient,
  onOpenNewClient,
  onRunAnalysis,
  onSignOut,
  isBusy,
}) => {
  const role = session.user.role;
  const canCreateClient = role === 'advisor' || role === 'compliance' || role === 'admin';
  const canRun = role === 'advisor' || role === 'compliance' || role === 'admin';

  const selectedClient = clients.find((c) => c.id === selectedClientId);

  return (
    <aside className="sidebar">
      {/* Brand */}
      <div className="sidebar-header">
        <div className="brand-icon">
          <ShieldCheck size={22} />
        </div>
        <div>
          <div className="brand-title">AI Wealth Manager</div>
          <div className="brand-subtitle">Autonomous Advisory</div>
        </div>
      </div>

      {/* Operator Session */}
      <div className="user-badge">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            Operator Session
          </span>
          <span className={`user-role-badge role-${role}`}>{role}</span>
        </div>
        <span className="user-email" title={session.user.email}>
          {session.user.email}
        </span>
      </div>

      {/* Client Selection */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <label className="form-label" style={{ marginBottom: 0, display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <Users size={14} /> Active Client
          </label>
          {canCreateClient && (
            <button
              id="new-client-trigger-btn"
              type="button"
              className="btn btn-ghost"
              onClick={onOpenNewClient}
              style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem', color: 'var(--accent-primary)' }}
            >
              <UserPlus size={14} /> New
            </button>
          )}
        </div>

        {clients.length > 0 ? (
          <select
            id="client-select-dropdown"
            className="form-select"
            value={selectedClientId || ''}
            onChange={(e) => onSelectClient(Number(e.target.value))}
            disabled={isBusy}
          >
            <option value="" disabled>-- Select a client --</option>
            {clients.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name} (#{c.id})
              </option>
            ))}
          </select>
        ) : (
          <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', padding: '0.5rem 0' }}>
            No client records found.
          </div>
        )}
      </div>

      {/* Selected Client Quick Summary Card */}
      {selectedClient && (
        <div style={{ background: 'var(--bg-surface-elevated)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', padding: '1rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{selectedClient.name}</span>
            <span className="badge badge-info" style={{ fontSize: '0.7rem' }}>
              {selectedClient.risk_tolerance}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <span>Net Worth:</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: 'var(--accent-primary)' }}>
              ${(selectedClient.net_worth || 0).toLocaleString()}
            </span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <span>Time Horizon:</span>
            <span>{selectedClient.time_horizon_years} yrs</span>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            <span>Age:</span>
            <span>{selectedClient.age}</span>
          </div>
        </div>
      )}

      {/* Actions */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: 'auto' }}>
        {selectedClientId && canRun && (
          <button
            id="run-analysis-btn"
            type="button"
            className="btn btn-primary"
            onClick={onRunAnalysis}
            disabled={isBusy}
            style={{ width: '100%', padding: '0.85rem' }}
          >
            <Play size={16} fill="currentColor" />
            <span>{isBusy ? 'Pipeline Active...' : 'Run Portfolio Analysis'}</span>
          </button>
        )}

        <button
          id="sign-out-btn"
          type="button"
          className="btn btn-ghost"
          onClick={onSignOut}
          disabled={isBusy}
          style={{ width: '100%', justifyContent: 'flex-start' }}
        >
          <LogOut size={16} />
          <span>Sign Out</span>
        </button>
      </div>
    </aside>
  );
};
