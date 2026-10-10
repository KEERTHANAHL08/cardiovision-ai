import { useState, useEffect, useCallback } from 'react';
import { api } from './api';
import PatientAssessment from './components/PatientAssessment';
import AnatomyViewer from './components/AnatomyViewer';
import RiskFactors from './components/RiskFactors';
import AIInsights from './components/AIInsights';
import ModelInsights from './components/ModelInsights';
import Limitations from './components/Limitations';
import {
  Heart, Activity, Brain, BarChart3, Shield, AlertTriangle,
  Users, Stethoscope
} from 'lucide-react';
import './App.css';

const TABS = [
  { id: 'assessment', label: 'Patient Assessment', icon: Stethoscope },
  { id: 'anatomy',    label: '3D Anatomy',         icon: Heart },
  { id: 'risk',       label: 'Risk Factors',       icon: Activity },
  { id: 'insights',   label: 'AI Insights',        icon: Brain },
  { id: 'model',      label: 'Model Insights',     icon: BarChart3 },
  { id: 'limits',     label: 'Limitations',        icon: Shield },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('assessment');
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [patientData, setPatientData] = useState(null);
  const [modelPerf, setModelPerf] = useState(null);
  const [demoPatients, setDemoPatients] = useState(null);
  const [backendOk, setBackendOk] = useState(null);
  const [loading, setLoading] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    api.health()
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false));

    api.modelPerformance()
      .then(setModelPerf)
      .catch(() => {});

    api.demoPatients()
      .then(setDemoPatients)
      .catch(() => {});
  }, []);

  const handlePredict = useCallback(async (data) => {
    setLoading(true);
    setPatientData(data);
    try {
      const [pred, expl] = await Promise.all([
        api.predict(data),
        api.explain(data),
      ]);
      setPrediction(pred);
      setExplanation(expl);
    } catch (e) {
      alert(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handleDemoSelect = useCallback(async (demo) => {
    const data = demo.features;
    setActiveTab('assessment');
    await handlePredict(data);
    // After prediction is set, switch to anatomy view
    setTimeout(() => setActiveTab('anatomy'), 300);
  }, [handlePredict]);

  if (backendOk === false) {
    return (
      <div className="app-layout">
        <nav className="top-nav">
          <div className="nav-brand">
            <div className="nav-brand-icon">♥</div>
            <div className="nav-brand-text">Cardio<span>Vision</span> AI</div>
          </div>
        </nav>
        <div className="error-state" style={{ flex: 1 }}>
          <AlertTriangle size={48} />
          <h3>Backend Unavailable</h3>
          <p>Unable to connect to the AI backend. Please make sure the FastAPI server is running on port 8000.</p>
          <button className="btn btn-primary" onClick={() => window.location.reload()}>
            Retry Connection
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-layout">
      {/* Navigation */}
      <nav className="top-nav">
        <div className="nav-brand">
          <div className="nav-brand-icon">♥</div>
          <div className="nav-brand-text">Cardio<span>Vision</span> AI</div>
        </div>
        <div className="nav-tabs">
          {TABS.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`nav-tab ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon size={16} />
                {tab.label}
                {tab.id === 'anatomy' && prediction && (
                  <span className="nav-badge">{prediction.risk_category}</span>
                )}
              </button>
            );
          })}
        </div>
      </nav>

      {/* Content */}
      <main className="page-container">
        {backendOk === null ? (
          <div className="loading-overlay">
            <div className="loading-spinner" />
            <div className="loading-text">Connecting to AI backend...</div>
          </div>
        ) : (
          <div className="tab-content" key={activeTab}>
            {activeTab === 'assessment' && (
              <PatientAssessment
                onPredict={handlePredict}
                prediction={prediction}
                loading={loading}
                demoPatients={demoPatients}
                onDemoSelect={handleDemoSelect}
                patientData={patientData}
              />
            )}
            {activeTab === 'anatomy' && (
              <AnatomyViewer
                prediction={prediction}
                explanation={explanation}
                patientData={patientData}
                demoPatients={demoPatients}
                onDemoSelect={handleDemoSelect}
              />
            )}
            {activeTab === 'risk' && (
              <RiskFactors
                prediction={prediction}
                explanation={explanation}
                patientData={patientData}
              />
            )}
            {activeTab === 'insights' && (
              <AIInsights
                prediction={prediction}
                explanation={explanation}
              />
            )}
            {activeTab === 'model' && (
              <ModelInsights modelPerf={modelPerf} />
            )}
            {activeTab === 'limits' && (
              <Limitations />
            )}
          </div>
        )}
      </main>

      {/* Fixed Disclaimer */}
      <footer style={{
        textAlign: 'center',
        padding: '12px',
        fontSize: '11px',
        color: 'var(--text-tertiary)',
        borderTop: '1px solid var(--border-light)',
        background: 'var(--bg-white)',
      }}>
        ⚕️ This system is an AI-based educational/research prototype and is not a medical diagnostic tool.
        Predictions should not be used to make medical decisions. Consult a qualified healthcare professional for clinical assessment.
      </footer>
    </div>
  );
}
