import React, { useState } from 'react';
import { login, ApiError } from '../api/client';
import { Session } from '../types';
import { Lock, Mail, Building2, ArrowRight, ShieldCheck } from 'lucide-react';

interface LoginFormProps {
  onSession: (session: Session) => void;
}

export const LoginForm: React.FC<LoginFormProps> = ({ onSession }) => {
  const [orgSlug, setOrgSlug] = useState('acme-wealth');
  const [email, setEmail] = useState('advisor@acme.example');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!orgSlug.trim() || !email.trim() || !password) {
      setError('Please fill in all fields.');
      return;
    }

    setError(null);
    setLoading(true);

    try {
      const session = await login(orgSlug, email, password);
      onSession(session);
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError('Authentication failed. Check your credentials.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-wrapper">
      <div className="auth-card">
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center', marginBottom: '2rem' }}>
          <div className="brand-icon" style={{ width: 48, height: 48, marginBottom: '1rem', fontSize: '1.25rem' }}>
            <ShieldCheck size={28} />
          </div>
          <h1 className="brand-title" style={{ fontSize: '1.5rem' }}>AI Wealth Manager</h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginTop: '0.25rem' }}>
            Autonomous Multi-Agent Portfolio Research
          </p>
        </div>

        {error && (
          <div className="alert-banner alert-danger" id="login-error-banner">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label className="form-label" htmlFor="login-org">Organisation Slug</label>
            <div style={{ position: 'relative' }}>
              <input
                id="login-org"
                type="text"
                className="form-input"
                value={orgSlug}
                onChange={(e) => setOrgSlug(e.target.value)}
                placeholder="e.g. acme-wealth"
                required
              />
            </div>
          </div>

          <div className="form-group">
            <label className="form-label" htmlFor="login-email">Email Address</label>
            <input
              id="login-email"
              type="email"
              className="form-input"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="advisor@acme.example"
              required
            />
          </div>

          <div className="form-group" style={{ marginBottom: '1.75rem' }}>
            <label className="form-label" htmlFor="login-password">Password</label>
            <input
              id="login-password"
              type="password"
              className="form-input"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••••••"
              required
            />
          </div>

          <button
            id="login-submit-btn"
            type="submit"
            className="btn btn-primary"
            style={{ width: '100%', padding: '0.85rem' }}
            disabled={loading}
          >
            {loading ? (
              <span>Authenticating...</span>
            ) : (
              <>
                <span>Sign in to Platform</span>
                <ArrowRight size={16} />
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
};
