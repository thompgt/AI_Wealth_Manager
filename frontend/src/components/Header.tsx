import React from 'react';
import { Shield, Sparkles, Database, CheckCircle2 } from 'lucide-react';
import { Session } from '../types';

interface Props {
  session: Session;
  clientCount: number;
}

export const Header: React.FC<Props> = ({ session, clientCount }) => {
  return (
    <header style={{ height: '64px', borderBottom: '1px solid var(--border-subtle)', background: 'var(--bg-surface-glass)', backdropFilter: 'blur(10px)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 2rem', flexShrink: 0 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Organization:</span>
          <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-primary)' }}>
            {session.user.org_slug || 'Acme Wealth Advisory'}
          </span>
        </div>

        <div style={{ height: '16px', width: '1px', background: 'var(--border-subtle)' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
          <span>{clientCount} Managed Clients</span>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', color: 'var(--accent-primary)', background: 'rgba(16, 185, 129, 0.1)', padding: '0.3rem 0.75rem', borderRadius: 'var(--radius-full)', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-primary)', display: 'inline-block' }} />
          <span>FastAPI Engine Online</span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', color: 'var(--accent-cyan)', background: 'rgba(6, 182, 212, 0.1)', padding: '0.3rem 0.75rem', borderRadius: 'var(--radius-full)', border: '1px solid rgba(6, 182, 212, 0.2)' }}>
          <Database size={12} />
          <span>PostgreSQL + MySQL Analytics</span>
        </div>
      </div>
    </header>
  );
};
