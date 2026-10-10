import { Brain, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from 'recharts';

export default function AIInsights({ prediction, explanation }) {
  if (!prediction || !explanation) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">AI Insights</h1>
          <p className="page-description">Run an assessment to view AI model explanations</p>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 'var(--space-3xl)' }}>
          <Brain size={48} color="var(--text-tertiary)" />
          <p style={{ marginTop: 16, color: 'var(--text-secondary)' }}>
            No prediction available. Go to Patient Assessment to run an AI assessment.
          </p>
        </div>
      </div>
    );
  }

  const allContribs = explanation.all_contributors || [];
  const riskClass = prediction.risk_category.toLowerCase();

  // Prepare chart data — sort by contribution value
  const chartData = allContribs
    .map(c => ({
      name: c.display_name,
      value: Number((c.contribution * 100).toFixed(2)),
      direction: c.contribution > 0 ? 'increasing' : 'decreasing',
      rawValue: c.value,
      feature: c.feature,
    }))
    .sort((a, b) => b.value - a.value);

  const formatValue = (feature, value) => {
    if (feature === 'gender') return value === 1 ? 'Female' : 'Male';
    if (feature === 'cholesterol' || feature === 'gluc') {
      return value === 1 ? 'Normal' : value === 2 ? 'Above Normal' : 'Well Above Normal';
    }
    if (feature === 'smoke') return value ? 'Smoker' : 'Non-smoker';
    if (feature === 'alco') return value ? 'Yes' : 'No';
    if (feature === 'active') return value ? 'Active' : 'Not Active';
    return value;
  };

  const CustomTooltip = ({ active, payload }) => {
    if (!active || !payload?.length) return null;
    const d = payload[0].payload;
    return (
      <div style={{
        background: 'white', padding: '12px 16px', borderRadius: 12,
        boxShadow: '0 4px 16px rgba(0,0,0,0.1)', border: '1px solid var(--border-color)',
        fontSize: 13,
      }}>
        <div style={{ fontWeight: 600, marginBottom: 4 }}>{d.name}</div>
        <div style={{ color: 'var(--text-secondary)' }}>
          Value: <strong>{formatValue(d.feature, d.rawValue)}</strong>
        </div>
        <div style={{ color: d.value > 0 ? 'var(--risk-high-text)' : 'var(--risk-low-text)', fontWeight: 600 }}>
          Contribution: {d.value > 0 ? '+' : ''}{d.value}%
        </div>
        <div style={{ fontSize: 11, color: 'var(--text-tertiary)', marginTop: 4 }}>
          {d.value > 0 ? 'Increases model risk' : 'Decreases model risk'}
        </div>
      </div>
    );
  };

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">AI Insights</h1>
        <p className="page-description">
          Model feature contribution analysis — {explanation.explanation_method}
        </p>
      </div>

      {/* Summary Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 'var(--space-md)', marginBottom: 'var(--space-xl)' }}>
        <div className="stat-card">
          <div className="stat-value">{prediction.ai_risk_index}</div>
          <div className="stat-label">AI Risk Index</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{(prediction.model_probability * 100).toFixed(1)}%</div>
          <div className="stat-label">Model Probability</div>
        </div>
        <div className="stat-card">
          <span className={`risk-badge ${riskClass}`}>
            <span className="risk-badge-dot" />
            {prediction.risk_category}
          </span>
          <div className="stat-label" style={{ marginTop: 8 }}>Risk Category</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ fontSize: 18 }}>
            {prediction.prediction === 1 ? 'Detected' : 'Lower Risk'}
          </div>
          <div className="stat-label">Prediction</div>
        </div>
      </div>

      {/* Feature Contribution Chart */}
      <div className="card">
        <div className="card-header">
          <div>
            <h2 className="card-title">Model Feature Contribution</h2>
            <p className="card-subtitle">
              How each feature shifts the model prediction relative to baseline values
            </p>
          </div>
          <Info size={18} color="var(--text-tertiary)" />
        </div>
        <div style={{ height: 400 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 120, right: 40, top: 10, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border-light)" />
              <XAxis type="number" tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} />
              <YAxis
                type="category"
                dataKey="name"
                tick={{ fontSize: 12, fill: 'var(--text-primary)' }}
                width={110}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine x={0} stroke="var(--border-color)" />
              <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={24}>
                {chartData.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={entry.value > 0 ? '#E8788A' : '#34D399'}
                    opacity={0.85}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Explanation */}
      <div className="card" style={{ marginTop: 'var(--space-lg)' }}>
        <div className="card-header">
          <h2 className="card-title">Understanding This Assessment</h2>
        </div>
        <div style={{ fontSize: 14, lineHeight: 1.8, color: 'var(--text-secondary)' }}>
          <p style={{ marginBottom: 12 }}>
            The chart above shows how each patient feature contributes to the model's prediction.
            <strong style={{ color: '#E8788A' }}> Red bars</strong> indicate features that increase
            the model's estimated cardiovascular disease risk, while
            <strong style={{ color: '#34D399' }}> green bars</strong> indicate protective features
            that decrease the estimated risk.
          </p>
          <p style={{ marginBottom: 12 }}>
            The contribution values are calculated by comparing the prediction with the patient's actual
            values versus baseline (median) values from the dataset. Larger bars indicate stronger
            influence on the model's output.
          </p>
          <p>
            <strong>Important:</strong> These contributions represent the model's mathematical sensitivity
            to each feature — they do not represent physical damage to organs, arteries, or any
            anatomical structure.
          </p>
        </div>
      </div>

      <div className="disclaimer-banner" style={{ marginTop: 'var(--space-lg)' }}>
        <strong>⚕️ Disclaimer</strong>
        {explanation.disclaimer}
      </div>
    </div>
  );
}
