import { useState, useEffect } from 'react';
import { Stethoscope, Play, Users, Loader2 } from 'lucide-react';

const FORM_FIELDS = [
  { group: 'Demographics', fields: [
    { name: 'age', label: 'Age', unit: 'years', type: 'number', min: 18, max: 100, step: 1 },
    { name: 'gender', label: 'Gender', type: 'select', options: [
      { value: 1, label: 'Female' }, { value: 2, label: 'Male' }
    ]},
    { name: 'height', label: 'Height', unit: 'cm', type: 'number', min: 100, max: 250, step: 1 },
    { name: 'weight', label: 'Weight', unit: 'kg', type: 'number', min: 20, max: 300, step: 0.5 },
  ]},
  { group: 'Vitals', fields: [
    { name: 'ap_hi', label: 'Systolic Blood Pressure', unit: 'mmHg', type: 'number', min: 60, max: 300, step: 1 },
    { name: 'ap_lo', label: 'Diastolic Blood Pressure', unit: 'mmHg', type: 'number', min: 30, max: 200, step: 1 },
  ]},
  { group: 'Blood Indicators', fields: [
    { name: 'cholesterol', label: 'Cholesterol Level', type: 'select', options: [
      { value: 1, label: 'Normal' }, { value: 2, label: 'Above Normal' }, { value: 3, label: 'Well Above Normal' }
    ]},
    { name: 'gluc', label: 'Glucose Level', type: 'select', options: [
      { value: 1, label: 'Normal' }, { value: 2, label: 'Above Normal' }, { value: 3, label: 'Well Above Normal' }
    ]},
  ]},
  { group: 'Lifestyle', fields: [
    { name: 'smoke', label: 'Smoking', type: 'select', options: [
      { value: 0, label: 'Non-smoker' }, { value: 1, label: 'Smoker' }
    ]},
    { name: 'alco', label: 'Alcohol Intake', type: 'select', options: [
      { value: 0, label: 'No' }, { value: 1, label: 'Yes' }
    ]},
    { name: 'active', label: 'Physical Activity', type: 'select', options: [
      { value: 0, label: 'Not Active' }, { value: 1, label: 'Active' }
    ]},
  ]},
];

const DEFAULT_VALUES = {
  age: '', gender: '', height: '', weight: '',
  ap_hi: '', ap_lo: '', cholesterol: '', gluc: '',
  smoke: '', alco: '', active: '',
};

export default function PatientAssessment({ onPredict, prediction, loading, demoPatients, onDemoSelect, patientData }) {
  const [form, setForm] = useState(DEFAULT_VALUES);
  const [errors, setErrors] = useState({});

  // Sync form when patientData changes (from demo)
  useEffect(() => {
    if (patientData) {
      const newForm = {};
      Object.keys(DEFAULT_VALUES).forEach(k => {
        newForm[k] = patientData[k] !== undefined ? patientData[k] : '';
      });
      setForm(newForm);
    }
  }, [patientData]);

  const handleChange = (name, value) => {
    setForm(prev => ({ ...prev, [name]: value }));
    setErrors(prev => ({ ...prev, [name]: null }));
  };

  const validate = () => {
    const errs = {};
    const allFields = FORM_FIELDS.flatMap(g => g.fields);
    for (const f of allFields) {
      const v = form[f.name];
      if (v === '' || v === null || v === undefined) {
        errs[f.name] = 'Required';
        continue;
      }
      if (f.type === 'number') {
        const num = Number(v);
        if (isNaN(num)) { errs[f.name] = 'Must be a number'; continue; }
        if (f.min !== undefined && num < f.min) errs[f.name] = `Min ${f.min}`;
        if (f.max !== undefined && num > f.max) errs[f.name] = `Max ${f.max}`;
      }
    }
    // Diastolic < systolic
    if (Number(form.ap_lo) >= Number(form.ap_hi)) {
      errs.ap_lo = 'Must be less than systolic';
    }
    setErrors(errs);
    return Object.keys(errs).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!validate()) return;
    const data = {};
    Object.keys(form).forEach(k => { data[k] = Number(form[k]); });
    onPredict(data);
  };

  const riskClass = prediction ? prediction.risk_category.toLowerCase() : '';

  return (
    <div>
      <div className="page-header">
        <h1 className="page-title">Patient Assessment</h1>
        <p className="page-description">Enter patient data to generate an AI cardiovascular risk assessment</p>
      </div>

      {/* Demo Patients */}
      {demoPatients && demoPatients.length > 0 && (
        <div className="section">
          <div className="section-header">
            <div className="section-icon" style={{ background: 'var(--bg-pastel-lavender)', color: 'var(--accent-secondary)' }}>
              <Users size={16} />
            </div>
            <h2 className="section-title">Demo Profiles</h2>
            <span style={{ fontSize: 12, color: 'var(--text-tertiary)', marginLeft: 8 }}>Derived from actual dataset rows</span>
          </div>
          <div className="demo-patients-grid">
            {demoPatients.map((demo, i) => (
              <div
                key={i}
                className="demo-patient-card"
                onClick={() => onDemoSelect(demo)}
                role="button"
                tabIndex={0}
                onKeyDown={e => e.key === 'Enter' && onDemoSelect(demo)}
              >
                <div className="demo-patient-label">{demo.label}</div>
                <div className="demo-patient-source">{demo.source}</div>
                <div className="demo-patient-stats">
                  <div className="demo-patient-stat">
                    <div className="demo-patient-stat-label">Age</div>
                    <div className="demo-patient-stat-value">{demo.features.age} yrs</div>
                  </div>
                  <div className="demo-patient-stat">
                    <div className="demo-patient-stat-label">BP</div>
                    <div className="demo-patient-stat-value">{demo.features.ap_hi}/{demo.features.ap_lo}</div>
                  </div>
                  <div className="demo-patient-stat">
                    <div className="demo-patient-stat-label">BMI</div>
                    <div className="demo-patient-stat-value">
                      {(demo.features.weight / Math.pow(demo.features.height / 100, 2)).toFixed(1)}
                    </div>
                  </div>
                  <div className="demo-patient-stat">
                    <div className="demo-patient-stat-label">Expected</div>
                    <div className="demo-patient-stat-value">
                      <span className={`risk-badge ${demo.expected_category.toLowerCase()}`} style={{ fontSize: 10, padding: '2px 8px' }}>
                        <span className="risk-badge-dot" />
                        {demo.expected_category}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: prediction ? '1fr 1fr' : '1fr', gap: 'var(--space-xl)' }}>
        {/* Form */}
        <div className="card">
          <div className="card-header">
            <div>
              <h2 className="card-title">Clinical Parameters</h2>
              <p className="card-subtitle">All fields are required • Values based on model training features</p>
            </div>
            <Stethoscope size={20} color="var(--accent-primary)" />
          </div>

          <form onSubmit={handleSubmit}>
            {FORM_FIELDS.map(group => (
              <div key={group.group}>
                <div className="form-section-title">{group.group}</div>
                <div className="form-grid">
                  {group.fields.map(field => (
                    <div className="form-group" key={field.name}>
                      <label className="form-label" htmlFor={field.name}>
                        {field.label}
                        {field.unit && <span className="unit"> ({field.unit})</span>}
                      </label>
                      {field.type === 'number' ? (
                        <input
                          id={field.name}
                          type="number"
                          className={`form-input ${errors[field.name] ? 'error' : ''}`}
                          value={form[field.name]}
                          onChange={e => handleChange(field.name, e.target.value)}
                          min={field.min}
                          max={field.max}
                          step={field.step}
                          placeholder={`${field.min}–${field.max}`}
                        />
                      ) : (
                        <select
                          id={field.name}
                          className={`form-select ${errors[field.name] ? 'error' : ''}`}
                          value={form[field.name]}
                          onChange={e => handleChange(field.name, e.target.value)}
                        >
                          <option value="">Select...</option>
                          {field.options.map(opt => (
                            <option key={opt.value} value={opt.value}>{opt.label}</option>
                          ))}
                        </select>
                      )}
                      {errors[field.name] && <div className="form-error">{errors[field.name]}</div>}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            <div style={{ marginTop: 'var(--space-xl)', display: 'flex', gap: 'var(--space-md)' }}>
              <button
                type="submit"
                className="btn btn-primary btn-lg"
                disabled={loading}
                style={{ flex: 1 }}
              >
                {loading ? (
                  <>
                    <Loader2 size={18} className="loading-spinner" style={{ width: 18, height: 18, border: 'none', borderTop: 'none' }} />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Play size={18} />
                    Run AI Assessment
                  </>
                )}
              </button>
            </div>
          </form>
        </div>

        {/* Result Panel */}
        {prediction && (
          <div className="result-panel">
            <div className="card" style={{
              borderColor: riskClass === 'low' ? 'var(--risk-low)' :
                          riskClass === 'moderate' ? 'var(--risk-moderate)' : 'var(--risk-high)',
              borderWidth: 2
            }}>
              <div className="card-header">
                <h2 className="card-title">Assessment Result</h2>
                <span className={`risk-badge ${riskClass}`}>
                  <span className="risk-badge-dot" />
                  {prediction.risk_category} RISK
                </span>
              </div>

              {/* Risk Meter */}
              <div className="risk-meter">
                <div className="risk-meter-circle">
                  <svg viewBox="0 0 160 160" style={{ transform: 'rotate(-90deg)' }}>
                    <circle cx="80" cy="80" r="68" fill="none" stroke="var(--border-light)" strokeWidth="10" />
                    <circle
                      cx="80" cy="80" r="68" fill="none"
                      stroke={riskClass === 'low' ? 'var(--risk-low)' :
                             riskClass === 'moderate' ? 'var(--risk-moderate)' : 'var(--risk-high)'}
                      strokeWidth="10"
                      strokeDasharray={`${prediction.ai_risk_index / 100 * 427} 427`}
                      strokeLinecap="round"
                      style={{ transition: 'stroke-dasharray 1s ease' }}
                    />
                  </svg>
                  <div className="risk-meter-value">
                    <div className={`risk-meter-number ${riskClass}`}>{prediction.ai_risk_index}</div>
                    <div className="risk-meter-label">AI Risk Index</div>
                  </div>
                </div>
              </div>

              {/* Stats */}
              <div className="result-grid" style={{ marginTop: 'var(--space-lg)' }}>
                <div className="stat-card">
                  <div className="stat-value" style={{
                    fontSize: 20,
                    color: prediction.prediction === 1 ? 'var(--risk-high)' : 'var(--risk-low)'
                  }}>
                    {prediction.prediction_label}
                  </div>
                  <div className="stat-label">Prediction</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value" style={{ fontSize: 20 }}>
                    {(prediction.model_probability * 100).toFixed(1)}%
                  </div>
                  <div className="stat-label">Model Probability</div>
                </div>
              </div>

              {/* Top Contributors */}
              {prediction.top_contributors && prediction.top_contributors.length > 0 && (
                <div style={{ marginTop: 'var(--space-xl)' }}>
                  <h3 style={{ fontSize: 14, fontWeight: 600, marginBottom: 'var(--space-md)' }}>
                    Top Contributing Factors
                  </h3>
                  {prediction.top_contributors.slice(0, 5).map((c, i) => {
                    const maxContrib = Math.max(...prediction.top_contributors.map(x => Math.abs(x.contribution)));
                    const pct = maxContrib > 0 ? Math.abs(c.contribution) / maxContrib * 100 : 0;
                    return (
                      <div className="contribution-bar-container" key={i}>
                        <div className="contribution-bar-label">
                          <span className="contribution-bar-name">{c.display_name}</span>
                          <span className="contribution-bar-value" style={{
                            color: c.contribution > 0 ? 'var(--risk-high-text)' : 'var(--risk-low-text)'
                          }}>
                            {c.direction}
                          </span>
                        </div>
                        <div className="contribution-bar-track">
                          <div
                            className={`contribution-bar-fill ${c.contribution > 0 ? 'positive' : 'negative'}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}

              <div className="disclaimer-banner" style={{ marginTop: 'var(--space-xl)' }}>
                <strong>⚕️ Important Disclaimer</strong>
                {prediction.disclaimer}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
