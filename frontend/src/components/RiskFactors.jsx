import { Activity, TrendingUp, TrendingDown, Minus } from 'lucide-react';

export default function RiskFactors({ prediction, explanation, patientData }) {
  if (!prediction || !explanation) {
    return (
      <div>
        <div className="page-header">
          <h1 className="page-title">Risk Factors</h1>
          <p className="page-description">Run an assessment to view risk factor analysis</p>
        </div>
        <div className="card" style={{ textAlign: 'center', padding: 'var(--space-3xl)' }}>
          <Activity size={48} color="var(--text-tertiary)" />
          <p style={{ marginTop: 16, color: 'var(--text-secondary)' }}>
            No prediction available. Go to Patient Assessment to run an AI assessment.
          </p>
        </div>
      </div>
    );
  }

  const riskClass = prediction.risk_category.toLowerCase();
  const increasing = explanation.risk_increasing_factors || [];
  const decreasing = explanation.risk_decreasing_factors || [];
  const neutral = explanation.neutral_factors || [];

  const allContribs = explanation.all_contributors || [];
  const maxContrib = Math.max(...allContribs.map(c => Math.abs(c.contribution)), 0.001);

  const featureLabels = {
    age: 'Age', gender: 'Gender', height: 'Height', weight: 'Weight',
    ap_hi: 'Systolic BP', ap_lo: 'Diastolic BP', cholesterol: 'Cholesterol',
    gluc: 'Glucose', smoke: 'Smoking', alco: 'Alcohol', active: 'Physical Activity'
  };

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

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Risk Factor Analysis</h1>
        <p className="page-description">
          Model feature contributions — these represent MODEL CONTRIBUTION, not physical damage
        </p>
      </div>

      {/* Summary */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 'var(--space-md)', marginBottom: 'var(--space-xl)' }}>
        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--risk-high)' }}>{increasing.length}</div>
          <div className="stat-label">Risk-Increasing</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--risk-low)' }}>{decreasing.length}</div>
          <div className="stat-label">Protective</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--text-tertiary)' }}>{neutral.length}</div>
          <div className="stat-label">Neutral</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-xl)' }}>
        {/* Risk Increasing */}
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <TrendingUp size={18} color="var(--risk-high)" />
              <h2 className="card-title" style={{ fontSize: 16 }}>Risk-Increasing Factors</h2>
            </div>
          </div>
          {increasing.length === 0 ? (
            <p style={{ color: 'var(--text-tertiary)', fontSize: 13 }}>No significant risk-increasing factors detected.</p>
          ) : (
            increasing.map((c, i) => {
              const pct = Math.abs(c.contribution) / maxContrib * 100;
              return (
                <div className="contribution-bar-container" key={i}>
                  <div className="contribution-bar-label">
                    <span className="contribution-bar-name">
                      {c.display_name}: <strong>{formatValue(c.feature, c.value)}</strong>
                    </span>
                    <span className="contribution-bar-value" style={{ color: 'var(--risk-high-text)' }}>
                      +{(c.contribution * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="contribution-bar-track">
                    <div className="contribution-bar-fill positive" style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Protective */}
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <TrendingDown size={18} color="var(--risk-low)" />
              <h2 className="card-title" style={{ fontSize: 16 }}>Protective Factors</h2>
            </div>
          </div>
          {decreasing.length === 0 ? (
            <p style={{ color: 'var(--text-tertiary)', fontSize: 13 }}>No significant protective factors detected.</p>
          ) : (
            decreasing.map((c, i) => {
              const pct = Math.abs(c.contribution) / maxContrib * 100;
              return (
                <div className="contribution-bar-container" key={i}>
                  <div className="contribution-bar-label">
                    <span className="contribution-bar-name">
                      {c.display_name}: <strong>{formatValue(c.feature, c.value)}</strong>
                    </span>
                    <span className="contribution-bar-value" style={{ color: 'var(--risk-low-text)' }}>
                      {(c.contribution * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="contribution-bar-track">
                    <div className="contribution-bar-fill negative" style={{ width: `${pct}%` }} />
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      <div className="disclaimer-banner" style={{ marginTop: 'var(--space-xl)' }}>
        <strong>⚕️ Interpretation Note</strong>
        Feature contributions represent how each feature value shifts the model's prediction relative to a baseline.
        They indicate MODEL CONTRIBUTION, not physical anatomical damage. This is an educational prototype.
      </div>
    </div>
  );
}
