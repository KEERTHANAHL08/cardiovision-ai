/**
 * CardioVision AI — API Service
 * All communication with the FastAPI backend.
 */
const API_BASE = (typeof window !== 'undefined' && window.location.hostname === 'localhost' && window.location.port === '5173')
  ? 'http://localhost:8000'
  : '';

async function request(path, options = {}) {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Request failed: ${res.status}`);
    }
    return await res.json();
  } catch (e) {
    if (e.message.includes('Failed to fetch') || e.message.includes('NetworkError')) {
      throw new Error('Unable to connect to the AI backend. Please make sure the FastAPI server is running on port 8000.');
    }
    throw e;
  }
}

export const api = {
  health:          ()     => request('/health'),
  datasetSummary:  ()     => request('/dataset-summary'),
  modelPerformance:()     => request('/model-performance'),
  demoPatients:    ()     => request('/demo-patients'),
  inputSchema:     ()     => request('/input-schema'),
  predict:         (data) => request('/predict', { method: 'POST', body: JSON.stringify(data) }),
  explain:         (data) => request('/explain', { method: 'POST', body: JSON.stringify(data) }),
};
