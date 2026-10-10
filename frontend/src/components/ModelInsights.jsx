import { useEffect, useState } from 'react';
import { BarChart3, Award, AlertCircle } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis
} from 'recharts';
import { api } from '../api';

const METRIC_LABELS = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  f1_score: 'F1 Score',
  roc_auc: 'ROC-AUC',
};

const MODEL_COLORS = {
  'Logistic Regression': '#4F8CDB',
  'K-Nearest Neighbors': '#8B7CC8',
  'Decision Tree': '#34D399',
  'Random Forest': '#F59E0B',
  'SVM': '#E8788A',
  'HistGradientBoosting': '#EC6B3F',
};

export default function ModelInsights({ modelPerf: initialPerf }) {
  const [perf, setPerf] = useState(initialPerf);
  const [loading, setLoading] = useState(!initialPerf);

  useEffect(() => {
    if (!initialPerf) {
      api.modelPerformance().then(setPerf).catch(() => {}).finally(() => setLoading(false));
    }
  }, [initialPerf]);

  if (loading) {
    return (
      <div className="loading-overlay">
        <div className="loading-spinner" />
        <div className="loading-text">Loading model performance data...</div>
      </div>
    );
  }

  if (!perf) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Model Insights</h1>
        </div>
        <div className="error-state">
          <AlertCircle size={48} />
          <h3>Performance Data Unavailable</h3>
          <p>Could not load model performance metrics. Make sure the backend is running.</p>
        </div>
      </div>
    );
  }

  const comparison = perf.model_comparison || {};
  const selected = perf.selected_model;
  const finalMetrics = perf.final_model_metrics || {};
  const cm = finalMetrics.confusion_matrix || [[0,0],[0,0]];
  const rankings = perf.rankings || {};

  // Comparison bar chart data
  const comparisonData = Object.entries(comparison).map(([name, m]) => ({
    name: name.length > 15 ? name.substring(0, 15) + '…' : name,
    fullName: name,
    accuracy: Number((m.accuracy * 100).toFixed(1)),
    recall: Number((m.recall * 100).toFixed(1)),
    f1_score: Number((m.f1_score * 100).toFixed(1)),
    roc_auc: Number((m.roc_auc * 100).toFixed(1)),
    isSelected: name === selected,
  }));

  // Radar chart for selected model
  const radarData = finalMetrics ? [
    { metric: 'Accuracy', value: Number((finalMetrics.accuracy * 100).toFixed(1)) },
    { metric: 'Precision', value: Number((finalMetrics.precision * 100).toFixed(1)) },
    { metric: 'Recall', value: Number((finalMetrics.recall * 100).toFixed(1)) },
    { metric: 'F1 Score', value: Number((finalMetrics.f1_score * 100).toFixed(1)) },
    { metric: 'ROC-AUC', value: Number((finalMetrics.roc_auc * 100).toFixed(1)) },
  ] : [];

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div style={{
        background: 'white', padding: '12px 16px', borderRadius: 12,
        boxShadow: '0 4px 16px rgba(0,0,0,0.1)', border: '1px solid var(--border-color)',
        fontSize: 13,
      }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>
          {d.fullName || d.name}
          {d.isSelected && <span className="best-badge" style={{ marginLeft: 6 }}>Selected</span>}
        </div>
        {payload.map((p, i) => (
          <div key={i} style={{ color: 'var(--text-secondary)', marginBottom: 2 }}>
            {p.name}: <strong>{p.value}%</strong>
          </div>
        ))}
      </div>
    );
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Model Insights</h1>
        <p className="page-description">
          Actual model performance metrics from the trained pipeline — all values from metrics.json
        </p>
      </div>

      {/* Selected Model */}
      <div className="card" style={{ marginBottom: 'var(--space-xl)', borderColor: 'var(--accent-primary)', borderWidth: 2 }}>
        <div className="card-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <Award size={22} color="var(--accent-primary)" />
            <div>
              <h2 className="card-title">Selected Model: {selected}</h2>
              <p className="card-subtitle">{perf.selection_reason}</p>
            </div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-xl)' }}>
          {/* Metrics */}
          <div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-sm)' }}>
              {Object.entries(METRIC_LABELS).map(([key, label]) => (
                <div className="stat-card" key={key} style={{ padding: 'var(--space-md)' }}>
                  <div className="stat-value" style={{ fontSize: 20 }}>
                    {(finalMetrics[key] * 100).toFixed(1)}%
                  </div>
                  <div className="stat-label">{label}</div>
                </div>
              ))}
              {comparison[selected] && (
                <div className="stat-card" style={{ padding: 'var(--space-md)' }}>
                  <div className="stat-value" style={{ fontSize: 16 }}>
                    {(comparison[selected].cv_accuracy_mean * 100).toFixed(1)}% ± {(comparison[selected].cv_accuracy_std * 100).toFixed(1)}%
                  </div>
                  <div className="stat-label">CV Accuracy</div>
                </div>
              )}
            </div>

            {/* Confusion Matrix */}
            <div style={{ marginTop: 'var(--space-xl)' }}>
              <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 'var(--space-md)', textAlign: 'center' }}>
                Confusion Matrix
              </h3>
              <div className="confusion-matrix">
                <div className="cm-label"></div>
                <div className="cm-label">Pred: No CVD</div>
                <div className="cm-label">Pred: CVD</div>
                <div className="cm-label" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual: No CVD</div>
                <div className="cm-cell cm-tn">{cm[0][0]}</div>
                <div className="cm-cell cm-fp">{cm[0][1]}</div>
                <div className="cm-label" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>Actual: CVD</div>
                <div className="cm-cell cm-fn">{cm[1][0]}</div>
                <div className="cm-cell cm-tp">{cm[1][1]}</div>
              </div>
            </div>
          </div>

          {/* Radar Chart */}
          <div>
            <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 'var(--space-sm)', textAlign: 'center' }}>
              Performance Profile
            </h3>
            <div style={{ height: 280 }}>
              <ResponsiveContainer>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="var(--border-color)" />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
                  <PolarRadiusAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                  <Radar
                    name="Score"
                    dataKey="value"
                    stroke="var(--accent-primary)"
                    fill="var(--accent-primary)"
                    fillOpacity={0.15}
                    strokeWidth={2}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>

      {/* Model Comparison */}
      <div className="card" style={{ marginBottom: 'var(--space-xl)' }}>
        <div className="card-header">
          <div>
            <h2 className="card-title">Model Comparison</h2>
            <p className="card-subtitle">6 models compared with stratified 5-fold cross-validation</p>
          </div>
        </div>

        <div style={{ height: 320, marginBottom: 'var(--space-xl)' }}>
          <ResponsiveContainer>
            <BarChart data={comparisonData} margin={{ left: 10, right: 20, top: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
              <YAxis domain={[60, 85]} tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Bar dataKey="recall" name="Recall" radius={[4, 4, 0, 0]} maxBarSize={18}>
                {comparisonData.map((d, i) => (
                  <Cell key={i} fill={d.isSelected ? '#4F8CDB' : '#A8D0F0'} />
                ))}
              </Bar>
              <Bar dataKey="roc_auc" name="ROC-AUC" radius={[4, 4, 0, 0]} maxBarSize={18}>
                {comparisonData.map((d, i) => (
                  <Cell key={i} fill={d.isSelected ? '#8B7CC8' : '#D4CCEF'} />
                ))}
              </Bar>
              <Bar dataKey="f1_score" name="F1 Score" radius={[4, 4, 0, 0]} maxBarSize={18}>
                {comparisonData.map((d, i) => (
                  <Cell key={i} fill={d.isSelected ? '#34D399' : '#A7F3D0'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Comparison Table */}
        <div style={{ overflowX: 'auto' }}>
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
                <th>ROC-AUC</th>
                <th>CV Mean ± Std</th>
                <th>Composite</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison).map(([name, m]) => (
                <tr key={name} className={name === selected ? 'selected' : ''}>
                  <td>
                    <span className="model-name">{name}</span>
                    {name === selected && <span className="best-badge">Selected</span>}
                  </td>
                  <td>{(m.accuracy * 100).toFixed(1)}%</td>
                  <td>{(m.precision * 100).toFixed(1)}%</td>
                  <td>{(m.recall * 100).toFixed(1)}%</td>
                  <td>{(m.f1_score * 100).toFixed(1)}%</td>
                  <td>{(m.roc_auc * 100).toFixed(1)}%</td>
                  <td>{(m.cv_accuracy_mean * 100).toFixed(1)}% ± {(m.cv_accuracy_std * 100).toFixed(1)}%</td>
                  <td style={{ fontWeight: 600, color: name === selected ? 'var(--accent-primary)' : 'var(--text-primary)' }}>
                    {rankings[name] ? (rankings[name] * 100).toFixed(1) : '–'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Why This Model */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Why {selected}?</h2>
        </div>
        <div style={{ fontSize: 14, lineHeight: 1.8, color: 'var(--text-secondary)' }}>
          <p>{perf.selection_reason}</p>
          <p style={{ marginTop: 12 }}>
            The model was selected using a weighted composite score that prioritizes
            recall (30%), ROC-AUC (30%), cross-validation stability (20%), and F1 score (20%).
            In a healthcare context, recall is weighted higher because missing a positive
            cardiovascular disease case (false negative) is generally more consequential
            than a false positive, which can be resolved through further clinical testing.
          </p>
          {perf.calibrated && (
            <p style={{ marginTop: 12 }}>
              <strong>Calibration:</strong> The model's predicted probabilities have been
              calibrated using isotonic regression to better reflect actual risk proportions.
              The AI Risk Index (0–100) is derived from calibrated probability × 100.
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
