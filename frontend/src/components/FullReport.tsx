import React from 'react';
import { FileText, Download, CheckCircle2 } from 'lucide-react';

interface Props {
  reportText?: string;
  reportId?: number;
  approvalState?: string;
  llmEnabled?: boolean;
}

export const FullReport: React.FC<Props> = ({
  reportText,
  reportId,
  approvalState,
  llmEnabled = true,
}) => {
  if (!reportText) return null;

  // Simple markdown renderer for headers, bold, bullets, paragraphs
  const renderMarkdown = (content: string) => {
    const lines = content.split('\n');
    const elements: React.ReactNode[] = [];
    let listItems: string[] = [];

    const flushList = () => {
      if (listItems.length > 0) {
        elements.push(
          <ul key={`list-${elements.length}`} style={{ paddingLeft: '1.5rem', marginBottom: '1rem' }}>
            {listItems.map((item, idx) => (
              <li key={idx} style={{ marginBottom: '0.4rem', color: '#D1D5DB' }}>
                {renderInline(item)}
              </li>
            ))}
          </ul>
        );
        listItems = [];
      }
    };

    const renderInline = (text: string) => {
      // replace **bold**
      const parts = text.split(/(\*\*.*?\*\*)/g);
      return parts.map((part, i) => {
        if (part.startsWith('**') && part.endsWith('**')) {
          return <strong key={i} style={{ color: '#FFFFFF' }}>{part.slice(2, -2)}</strong>;
        }
        return part;
      });
    };

    lines.forEach((line, index) => {
      const trimmed = line.trim();
      if (trimmed.startsWith('# ')) {
        flushList();
        elements.push(<h1 key={index} style={{ fontSize: '1.4rem', fontWeight: 700, margin: '1.5rem 0 0.75rem', color: '#FFFFFF' }}>{trimmed.slice(2)}</h1>);
      } else if (trimmed.startsWith('## ')) {
        flushList();
        elements.push(<h2 key={index} style={{ fontSize: '1.2rem', fontWeight: 600, margin: '1.25rem 0 0.5rem', color: 'var(--accent-cyan)' }}>{trimmed.slice(3)}</h2>);
      } else if (trimmed.startsWith('### ')) {
        flushList();
        elements.push(<h3 key={index} style={{ fontSize: '1.05rem', fontWeight: 600, margin: '1rem 0 0.5rem', color: '#E5E7EB' }}>{trimmed.slice(4)}</h3>);
      } else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
        listItems.push(trimmed.slice(2));
      } else if (trimmed === '---') {
        flushList();
        elements.push(<hr key={index} style={{ border: 0, borderTop: '1px solid var(--border-subtle)', margin: '1.5rem 0' }} />);
      } else if (trimmed.length > 0) {
        flushList();
        elements.push(
          <p key={index} style={{ marginBottom: '1rem', color: '#D1D5DB', lineHeight: 1.7 }}>
            {renderInline(trimmed)}
          </p>
        );
      }
    });

    flushList();
    return elements;
  };

  const handleExport = () => {
    const blob = new Blob([reportText], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `investment-memo-report-${reportId || 'latest'}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="card" style={{ marginBottom: '2rem' }}>
      <div className="card-header">
        <h2 className="card-title">
          <FileText size={20} color="var(--accent-primary)" />
          Client-Facing Investment Research Memo
        </h2>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          {approvalState === 'approved' && (
            <span className="badge badge-success">
              <CheckCircle2 size={14} /> COMPLIANCE APPROVED
            </span>
          )}
          <button
            id="download-report-btn"
            type="button"
            className="btn btn-secondary"
            onClick={handleExport}
            style={{ padding: '0.4rem 0.85rem', fontSize: '0.8rem' }}
          >
            <Download size={14} />
            <span>Export Markdown</span>
          </button>
        </div>
      </div>

      {!llmEnabled && (
        <div className="alert-banner alert-warning" style={{ marginBottom: '1.5rem' }}>
          <span>
            <strong>Deterministic Mode:</strong> No GEMINI_API_KEY configured; synthesis generated via verified template fallback.
          </span>
        </div>
      )}

      <div style={{ background: 'var(--bg-surface-elevated)', padding: '2rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }} className="prose">
        {renderMarkdown(reportText)}
      </div>
    </div>
  );
};
