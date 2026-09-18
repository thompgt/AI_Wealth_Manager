import {
  Client,
  JobState,
  QuestionnaireSchema,
  ReportViewModel,
  Session,
  User,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || '';

export class ApiError extends Error {
  statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
  token?: string
): Promise<T> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const url = `${API_BASE}${path}`;

  let response: Response;
  try {
    response = await fetch(url, { ...options, headers });
  } catch (err: any) {
    throw new ApiError(`Could not reach the API at ${url}: ${err.message || err}`);
  }

  if (response.status === 204 || response.headers.get('content-length') === '0') {
    return null as any;
  }

  let data: any;
  const contentType = response.headers.get('content-type') || '';
  if (contentType.includes('application/json')) {
    try {
      data = await response.json();
    } catch {
      data = null;
    }
  } else {
    data = await response.text();
  }

  if (!response.ok) {
    let message = 'Request failed';
    if (data && typeof data === 'object') {
      const detail = data.detail;
      if (Array.isArray(detail)) {
        message = detail
          .map((item: any) => {
            if (typeof item === 'object' && item.loc) {
              const loc = item.loc.slice(1).join('.');
              return loc ? `${loc}: ${item.msg}` : item.msg;
            }
            return String(item);
          })
          .join('; ');
      } else if (typeof detail === 'string') {
        message = detail;
      } else if (data.message) {
        message = String(data.message);
      }
    } else if (typeof data === 'string' && data.length > 0) {
      message = data;
    } else {
      message = `${response.status} ${response.statusText}`;
    }
    throw new ApiError(message, response.status);
  }

  return data as T;
}

export async function login(orgSlug: string, email: string, password: string): Promise<Session> {
  const authPayload = await request<{ access_token: string }>('/api/v1/auth/login', {
    method: 'POST',
    body: JSON.stringify({
      org_slug: orgSlug.trim(),
      email: email.trim(),
      password,
    }),
  });

  const token = authPayload.access_token;
  const user = await request<User>('/api/v1/auth/me', { method: 'GET' }, token);

  return { token, user };
}

export async function listClients(token: string): Promise<Client[]> {
  return (await request<Client[]>('/api/v1/clients', { method: 'GET' }, token)) || [];
}

export async function getClient(token: string, id: number): Promise<Client> {
  return request<Client>(`/api/v1/clients/${id}`, { method: 'GET' }, token);
}

export async function createClient(token: string, payload: any): Promise<Client> {
  return request<Client>('/api/v1/clients', {
    method: 'POST',
    body: JSON.stringify(payload),
  }, token);
}

export async function getQuestionnaire(token: string): Promise<QuestionnaireSchema> {
  return request<QuestionnaireSchema>('/api/v1/questionnaire', { method: 'GET' }, token);
}

export async function submitRiskAssessment(
  token: string,
  clientId: number,
  answers: Record<string, string>
): Promise<any> {
  return request<any>(`/api/v1/clients/${clientId}/risk-assessment`, {
    method: 'POST',
    body: JSON.stringify({ answers }),
  }, token);
}

export async function triggerRun(token: string, clientId: number): Promise<{ job_id: string; message?: string }> {
  return request<{ job_id: string; message?: string }>(`/api/v1/clients/${clientId}/runs`, {
    method: 'POST',
  }, token);
}

export async function getJob(token: string, jobId: string): Promise<JobState> {
  return request<JobState>(`/api/v1/jobs/${jobId}`, { method: 'GET' }, token);
}

export async function getReport(token: string, reportId: number): Promise<any> {
  return request<any>(`/api/v1/reports/${reportId}`, { method: 'GET' }, token);
}

export async function approveRun(
  token: string,
  runId: string,
  approved: boolean,
  note?: string
): Promise<any> {
  return request<any>(`/api/v1/runs/${runId}/approve`, {
    method: 'POST',
    body: JSON.stringify({ approved, note: note || null }),
  }, token);
}

export function formatReportViewModel(report: any): ReportViewModel {
  const payload = report.structured_payload || {};
  return {
    status: 'completed',
    run_id: report.run_id,
    report_id: report.id,
    llm_enabled: report.llm_enabled,
    final_report: report.report_text,
    approval_state: report.approval_state,
    degraded: report.degraded,
    portfolio_diagnostics: payload.portfolio_diagnostics,
    market_regime: payload.market_regime,
    suitability_result: payload.suitability_result,
    tax_assessment: payload.tax_assessment,
    tax_blocked_recommendations: payload.tax_blocked_recommendations,
    rebalance_plan: payload.rebalance_plan,
  };
}

export const TERMINAL_JOB_STATES: string[] = ['succeeded', 'failed', 'cancelled', 'timed_out'];
