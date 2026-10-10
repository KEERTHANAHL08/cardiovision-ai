import { useState, useRef, useMemo, useEffect, Suspense, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, useGLTF } from '@react-three/drei';
import {
  RotateCcw, Eye, EyeOff, Activity, Heart, Zap,
  RefreshCw, AlertTriangle, ShieldAlert, CheckCircle2,
  Stethoscope, Play, Pause, ChevronRight, Info
} from 'lucide-react';
import * as THREE from 'three';

// Preload the real anatomical human heart 3D model
useGLTF.preload('/models/heart.glb');

// ─── Light Pastel Theme System ───────────────────────────────────────
const PASTEL_THEMES = {
  LOW: {
    category: 'LOW',
    label: 'Low Risk',
    primary: '#10B981',
    glow: '#6EE7B7',
    badgeBg: '#E6F7F0',
    badgeBorder: '#A7F3D0',
    badgeText: '#065F46',
    bpm: 64,
    statusText: 'Optimal Myocardial Compliance & Perfusion'
  },
  MODERATE: {
    category: 'MODERATE',
    label: 'Moderate Risk',
    primary: '#F59E0B',
    glow: '#FCD34D',
    badgeBg: '#FFFBEB',
    badgeBorder: '#FDE68A',
    badgeText: '#92400E',
    bpm: 78,
    statusText: 'Elevated Hemodynamic Afterload & Shear'
  },
  HIGH: {
    category: 'HIGH',
    label: 'High Risk',
    primary: '#E11D48',
    glow: '#FDA4AF',
    badgeBg: '#FFF1F2',
    badgeBorder: '#FECDD3',
    badgeText: '#9F1239',
    bpm: 94,
    statusText: 'Critical Ventricular Afterload & Coronary Risk'
  },
  DEFAULT: {
    category: 'BASELINE',
    label: 'Physiological Baseline',
    primary: '#3B82F6',
    glow: '#93C5FD',
    badgeBg: '#EFF6FF',
    badgeBorder: '#BFDBFE',
    badgeText: '#1E40AF',
    bpm: 70,
    statusText: 'Normative Resting Hemodynamics'
  }
};

function getTheme(category) {
  if (!category) return PASTEL_THEMES.DEFAULT;
  return PASTEL_THEMES[category.toUpperCase()] || PASTEL_THEMES.DEFAULT;
}

// ─── Anatomical Structures (Encyclopaedia Britannica Format) ────────
// Divided into LEFT margin labels and RIGHT margin labels
const LEFT_STRUCTURES = [
  {
    id: 'SVC',
    name: 'superior vena cava',
    subtext: '(carries blood to heart)',
    surfacePoint: [0.48, 1.15, -0.1],
    color: '#2563EB',
    role: 'Large vein carrying deoxygenated systemic return from the head, neck, and upper limbs to the right atrium.'
  },
  {
    id: 'RIGHT_AURICLE',
    name: 'right auricle',
    subtext: '(muscular pouch of right atrium)',
    surfacePoint: [0.42, 0.48, 0.28],
    color: '#3B82F6',
    role: 'Small muscular appendage of the right atrium increasing holding capacity.'
  },
  {
    id: 'RIGHT_ATRIUM',
    name: 'right atrium',
    subtext: '(receives deoxygenated blood)',
    surfacePoint: [0.55, 0.22, 0.12],
    color: '#3B82F6',
    role: 'Receives deoxygenated blood from the SVC and IVC before sending it through the tricuspid valve to the right ventricle.'
  },
  {
    id: 'RCA',
    name: 'right coronary artery (RCA)',
    subtext: '(perfuses RV & nodal tissue)',
    surfacePoint: [0.45, -0.08, 0.42],
    color: '#E11D48',
    role: 'Courses through the right atrioventricular groove supplying the right ventricle, SA node, and AV nodal conduction tissue.'
  },
  {
    id: 'RIGHT_VENTRICLE',
    name: 'right ventricle',
    subtext: '(pumps blood to lungs)',
    surfacePoint: [0.22, -0.38, 0.45],
    color: '#6366F1',
    role: 'Pumps blood under lower pressure through the pulmonary valve into the pulmonary trunk.'
  },
  {
    id: 'IVC',
    name: 'inferior vena cava',
    subtext: '(carries blood to heart)',
    surfacePoint: [0.42, -0.92, -0.1],
    color: '#2563EB',
    role: 'Large vein returning deoxygenated blood from the abdomen and lower extremities to the right atrium.'
  }
];

const RIGHT_STRUCTURES = [
  {
    id: 'AORTA',
    name: 'aorta',
    subtext: '(carries blood to body)',
    surfacePoint: [-0.08, 1.25, 0.18],
    color: '#DC2626',
    role: 'Primary systemic artery distributing oxygenated blood from the left ventricle to the entire body.'
  },
  {
    id: 'PULMONARY_TRUNK',
    name: 'pulmonary trunk',
    subtext: '(carries blood to lungs)',
    surfacePoint: [0.22, 0.76, 0.38],
    color: '#2563EB',
    role: 'Arises from the right ventricle and bifurcates into the left and right pulmonary arteries to deliver blood to the lungs.'
  },
  {
    id: 'LEFT_AURICLE',
    name: 'left auricle',
    subtext: '(muscular pouch of left atrium)',
    surfacePoint: [-0.42, 0.45, 0.22],
    color: '#9333EA',
    role: 'Trabeculated appendage of the left atrium.'
  },
  {
    id: 'PULMONARY_VEINS',
    name: 'pulmonary veins',
    subtext: '(carry blood from lungs)',
    surfacePoint: [-0.55, 0.48, -0.25],
    color: '#E11D48',
    role: 'Deliver oxygen-rich blood from both lungs into the posterior left atrium.'
  },
  {
    id: 'LAD',
    name: 'LAD coronary artery',
    subtext: '(perfuses anterior LV & septum)',
    surfacePoint: [-0.1, -0.12, 0.65],
    color: '#E11D48',
    role: 'Left Anterior Descending artery courses along the anterior interventricular sulcus supplying ~50% of the left ventricular myocardium.'
  },
  {
    id: 'LEFT_ATRIUM',
    name: 'left atrium',
    subtext: '(receives oxygenated blood)',
    surfacePoint: [-0.45, 0.35, -0.12],
    color: '#9333EA',
    role: 'Chamber receiving oxygenated pulmonary venous return before pumping it through the mitral valve.'
  },
  {
    id: 'LEFT_VENTRICLE',
    name: 'left ventricle',
    subtext: '(pumps oxygenated blood to body)',
    surfacePoint: [-0.36, -0.52, 0.38],
    color: '#9333EA',
    role: 'Thick muscular chamber that pumps blood against systemic vascular resistance at high systolic pressures.'
  }
];

const ALL_STRUCTURES = [...LEFT_STRUCTURES, ...RIGHT_STRUCTURES];

// ─── Patient-Specific Clinical Problem Analyzer ──────────────────────
function analyzePatientProblems(patientData, prediction) {
  if (!patientData) return [];

  const problems = [];
  const sbp = Number(patientData.ap_hi || 120);
  const dbp = Number(patientData.ap_lo || 80);
  const chol = Number(patientData.cholesterol || 1);
  const smoke = Number(patientData.smoke || 0);
  const gluc = Number(patientData.gluc || 1);
  const pulsePressure = sbp - dbp;

  // 1. Systolic Blood Pressure Impact
  if (sbp >= 140) {
    problems.push({
      id: 'BP_HIGH',
      structures: ['AORTA', 'LEFT_VENTRICLE'],
      severity: 'CRITICAL',
      title: 'High Afterload & Ventricular Wall Tension',
      trigger: `Systolic BP: ${sbp} mmHg (Stage 2 Hypertension)`,
      affectedNames: 'Aorta & Left Ventricle',
      mechanism: `Systolic pressure of ${sbp} mmHg forces the left ventricle to pump against high vascular resistance, causing chronic left ventricular strain, hypertrophy, and aortic root shear stress.`,
      clinicalAction: 'Echocardiographic assessment of LV mass index and aggressive antihypertensive management.'
    });
  } else if (sbp >= 130) {
    problems.push({
      id: 'BP_ELEVATED',
      structures: ['AORTA', 'LEFT_VENTRICLE'],
      severity: 'MODERATE',
      title: 'Elevated Systemic Vascular Resistance',
      trigger: `Systolic BP: ${sbp} mmHg (Elevated)`,
      affectedNames: 'Aorta & Left Ventricle',
      mechanism: `Elevated systolic pressure increases pulsatile wall stress on the ascending aorta and raises left ventricular myocardial workload.`,
      clinicalAction: 'Lifestyle modifications (DASH diet, sodium reduction) and blood pressure monitoring.'
    });
  }

  // 2. Cholesterol Impact on Coronary Arteries
  if (chol === 3) {
    problems.push({
      id: 'CHOL_CRITICAL',
      structures: ['LAD', 'RCA'],
      severity: 'CRITICAL',
      title: 'Coronary Atheroma & Plaque Vulnerability',
      trigger: 'Cholesterol: Well Above Normal (Level 3)',
      affectedNames: 'LAD & RCA Coronary Arteries',
      mechanism: `Severely elevated circulating lipids accelerate subendothelial LDL accumulation and atheroma growth, especially in the high-flow Left Anterior Descending (LAD) and Right Coronary (RCA) arteries.`,
      clinicalAction: 'Lipid panel quantification, statin therapy evaluation, and coronary calcium scoring.'
    });
  } else if (chol === 2) {
    problems.push({
      id: 'CHOL_MODERATE',
      structures: ['LAD'],
      severity: 'MODERATE',
      title: 'Early Coronary Endothelial Infiltration',
      trigger: 'Cholesterol: Above Normal (Level 2)',
      affectedNames: 'LAD Coronary Artery',
      mechanism: `Elevated cholesterol contributes to early coronary intimal-medial thickening and oxidative microvascular stress.`,
      clinicalAction: 'Dietary counseling and baseline lipid monitoring.'
    });
  }

  // 3. Widened Pulse Pressure / Aortic Compliance
  if (pulsePressure >= 60) {
    problems.push({
      id: 'AORTIC_STIFFNESS',
      structures: ['AORTA'],
      severity: 'MODERATE',
      title: 'Aortic Elastic Stiffening (Reduced Compliance)',
      trigger: `Pulse Pressure: ${pulsePressure} mmHg (Widened)`,
      affectedNames: 'Aorta & Aortic Arch',
      mechanism: `A widened pulse pressure (${pulsePressure} mmHg) indicates loss of aortic elastic cushioning, exposing coronary vessels to abrupt pulsatile peaks.`,
      clinicalAction: 'Vascular compliance assessment and arterial stiffness evaluation.'
    });
  }

  // 4. Smoking Vasoconstriction
  if (smoke === 1) {
    problems.push({
      id: 'SMOKING_ENDOTHELIAL',
      structures: ['LAD', 'RCA'],
      severity: 'MODERATE',
      title: 'Coronary Endothelial Dysfunction & Vasospasm',
      trigger: 'Active Smoker',
      affectedNames: 'Coronary Microvasculature',
      mechanism: `Nicotine and combustion oxidants reduce nitric oxide bioavailability, provoking coronary artery vasospasm and promoting thrombotic platelet aggregation.`,
      clinicalAction: 'Immediate smoking cessation counseling and microvascular risk evaluation.'
    });
  }

  return problems;
}

// ─── Real 3D Human Heart Scene with Smooth Rotation ──────────────────
function Heart3DScene({
  theme,
  riskLevel,
  autoRotate,
  rotationSpeed,
  activeAngle,
  highlightedStructure,
  problemStructures,
  onUpdateProjections
}) {
  const { scene } = useGLTF('/models/heart.glb');
  const groupRef = useRef();
  const heartMeshRef = useRef();
  const { camera, size } = useThree();

  // Clone scene so multiple mounts remain pristine
  const clonedScene = useMemo(() => scene.clone(true), [scene]);

  // Adjust materials for authentic medical appearance
  useEffect(() => {
    clonedScene.traverse((child) => {
      if (child.isMesh && child.material) {
        child.castShadow = true;
        child.receiveShadow = true;
        child.material.roughness = 0.44;
        child.material.metalness = 0.08;

        // Subtle realistic organic cardiac tissue with delicate perfusion glow
        if (riskLevel > 0.6) {
          child.material.emissive = new THREE.Color(0x350409);
          child.material.emissiveIntensity = 0.22;
        } else {
          child.material.emissive = new THREE.Color(0x100204);
          child.material.emissiveIntensity = 0.06;
        }
        child.material.needsUpdate = true;
      }
    });
  }, [clonedScene, riskLevel]);

  // Handle angle presets smoothly
  useEffect(() => {
    if (!autoRotate && groupRef.current && activeAngle !== null) {
      groupRef.current.rotation.y = activeAngle;
    }
  }, [activeAngle, autoRotate]);

  // Pumping animation + 360 Turntable Continuous Rotation + Coordinate Projection
  const freq = (theme.bpm / 60) * Math.PI * 2;
  useFrame(({ clock }, delta) => {
    // 1. Continuous 360° Smooth Turntable Rotation
    if (autoRotate && groupRef.current) {
      groupRef.current.rotation.y += delta * rotationSpeed;
    }

    // 2. Anatomical Heartbeat Pulsing
    if (heartMeshRef.current) {
      const t = clock.getElapsedTime() * freq;
      const beat =
        Math.sin(t) * 0.025 +
        Math.pow(Math.max(0, Math.sin(t)), 4) * 0.04 * (0.8 + riskLevel * 0.6);
      const s = 1.0 + beat;
      heartMeshRef.current.scale.set(s, s * 0.98, s * 1.02);
    }

    // 3. Project 3D Coordinates to 2D Screen Space for Britannica-Style Leader Lines
    if (groupRef.current && onUpdateProjections) {
      const projectedMap = {};
      const matrixWorld = groupRef.current.matrixWorld;

      ALL_STRUCTURES.forEach((struct) => {
        const v = new THREE.Vector3(...struct.surfacePoint);
        v.applyMatrix4(matrixWorld);

        // Calculate surface normal to see if the structure is facing the camera
        const camDir = new THREE.Vector3().subVectors(camera.position, v).normalize();
        const localNormal = new THREE.Vector3(...struct.surfacePoint).normalize();
        localNormal.applyMatrix4(matrixWorld).sub(groupRef.current.position).normalize();
        const dot = localNormal.dot(camDir);

        // Project to normalized device coordinates (-1 to +1)
        v.project(camera);

        // Convert to canvas pixel coordinates
        const x = ((v.x + 1) / 2) * size.width;
        const y = ((-v.y + 1) / 2) * size.height;

        projectedMap[struct.id] = {
          x,
          y,
          // Structure is visible if facing generally toward camera and not behind camera
          isFacingFront: dot > -0.25,
          inZ: v.z < 1.0
        };
      });

      onUpdateProjections(projectedMap);
    }
  });

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      {/* Real Textured Human Heart Model scaled up prominently */}
      <group ref={heartMeshRef}>
        <primitive
          object={clonedScene}
          scale={1.35}
          position={[0, -0.1, 0]}
          rotation={[0.1, -0.25, -0.05]}
        />
      </group>
    </group>
  );
}

// ─── Loading State ───────────────────────────────────────────────────
function HeartModelLoader() {
  return null;
}

// ─── Camera Controller ───────────────────────────────────────────────
function CameraController({ resetTrigger }) {
  const { camera } = useThree();
  useEffect(() => {
    if (resetTrigger > 0) {
      camera.position.set(0, 0, 4.3);
      camera.lookAt(0, 0, 0);
    }
  }, [resetTrigger, camera]);
  return null;
}

// ─── Master Anatomy Viewer Component ─────────────────────────────────
export default function AnatomyViewer({
  prediction,
  explanation,
  patientData,
  demoPatients,
  onDemoSelect
}) {
  const [autoRotate, setAutoRotate] = useState(true);
  const [rotationSpeed, setRotationSpeed] = useState(0.45);
  const [activeAngle, setActiveAngle] = useState(0);
  const [resetTrigger, setResetTrigger] = useState(0);
  const [highlightedId, setHighlightedId] = useState('AORTA');
  const [showLabels, setShowLabels] = useState(true);

  // 2D screen positions of 3D landmarks updated every frame
  const [screenCoords, setScreenCoords] = useState({});

  const containerRef = useRef();

  const theme = getTheme(prediction?.risk_category);
  const riskIndex = prediction ? prediction.ai_risk_index : 25;
  const probability = prediction ? (prediction.model_probability * 100).toFixed(1) : '25.0';

  // Analyze patient-specific problems from actual clinical data
  const patientProblems = useMemo(() => {
    return analyzePatientProblems(patientData, prediction);
  }, [patientData, prediction]);

  // Set of structure IDs that have identified patient problems
  const problemStructureIds = useMemo(() => {
    const ids = new Set();
    patientProblems.forEach(p => p.structures.forEach(s => ids.add(s)));
    return ids;
  }, [patientProblems]);

  // Callback to receive updated screen coordinates from 3D canvas
  const handleUpdateProjections = useCallback((coords) => {
    setScreenCoords(coords);
  }, []);

  // Preset angle views
  const handleAngleSelect = (angleRads) => {
    setAutoRotate(false);
    setActiveAngle(angleRads);
  };

  // Jump to specific problem and focus on it
  const handleFocusProblem = (problem) => {
    setAutoRotate(false);
    const firstStruct = problem.structures[0];
    if (firstStruct) {
      setHighlightedId(firstStruct);
      // Turn heart toward front or optimal angle
      setActiveAngle(0);
    }
  };

  const selectedStructureData = useMemo(() => {
    return ALL_STRUCTURES.find(s => s.id === highlightedId) || ALL_STRUCTURES[0];
  }, [highlightedId]);

  const activeProblemForSelected = useMemo(() => {
    return patientProblems.find(p => p.structures.includes(highlightedId)) || null;
  }, [patientProblems, highlightedId]);

  return (
    <div className="tab-pane" style={{ animation: 'fadeIn 0.3s ease' }}>
      {/* Header with status pill */}
      <div
        className="page-header"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          gap: '16px',
          marginBottom: '16px'
        }}
      >
        <div>
          <h2 className="page-title" style={{ color: 'var(--text-primary)' }}>
            3D Cardiovascular Anatomy & Clinical Problem Map
          </h2>
          <p className="page-subtitle" style={{ color: 'var(--text-secondary)' }}>
            Anatomical structure diagram with leader-line pointers and patient-specific cardiovascular stress detection.
          </p>
        </div>

        {/* Status Badge in Light Pastel */}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '10px',
            background: theme.badgeBg,
            border: `1.5px solid ${theme.badgeBorder}`,
            padding: '8px 18px',
            borderRadius: '24px',
            boxShadow: '0 2px 8px rgba(0,0,0,0.03)'
          }}
        >
          <span
            style={{
              width: '10px',
              height: '10px',
              borderRadius: '50%',
              background: theme.primary,
              boxShadow: `0 0 8px ${theme.primary}`
            }}
          />
          <span style={{ fontSize: '13px', fontWeight: '700', color: theme.badgeText }}>
            {prediction ? `${prediction.risk_category} RISK PROFILE (${riskIndex}/100)` : 'PHYSIOLOGICAL BASELINE'}
          </span>
        </div>
      </div>

      {/* Patient Profile Quick-Loader Pills */}
      {demoPatients && demoPatients.length > 0 && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            padding: '10px 18px',
            background: '#FFFFFF',
            borderRadius: '16px',
            border: '1px solid var(--border-color)',
            marginBottom: '16px',
            flexWrap: 'wrap',
            boxShadow: '0 2px 10px rgba(79, 140, 219, 0.05)'
          }}
        >
          <span
            style={{
              fontSize: '13px',
              fontWeight: '700',
              color: 'var(--text-secondary)',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Activity size={16} color="var(--accent-primary)" />
            Load Clinical Profile:
          </span>
          {demoPatients.map((demo, idx) => {
            const isSelected =
              prediction && Math.abs(prediction.model_probability - demo.expected_probability) < 0.05;
            const catTheme = getTheme(demo.expected_category);

            return (
              <button
                key={idx}
                onClick={() => onDemoSelect && onDemoSelect(demo)}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '6px 14px',
                  borderRadius: '20px',
                  border: isSelected
                    ? `2px solid ${catTheme.primary}`
                    : '1px solid var(--border-color)',
                  background: isSelected ? catTheme.badgeBg : '#FFFFFF',
                  cursor: 'pointer',
                  fontSize: '12px',
                  fontWeight: isSelected ? '700' : '500',
                  color: isSelected ? catTheme.badgeText : 'var(--text-primary)',
                  boxShadow: isSelected
                    ? `0 2px 8px ${catTheme.primary}25`
                    : '0 1px 3px rgba(0,0,0,0.04)',
                  transition: 'all 0.2s ease'
                }}
              >
                <span
                  style={{
                    width: '8px',
                    height: '8px',
                    borderRadius: '50%',
                    background: catTheme.primary
                  }}
                />
                {demo.label} ({demo.expected_risk_index}/100)
              </button>
            );
          })}
        </div>
      )}

      {/* ─── "WHERE IS THE PROBLEM FOR PATIENT?" CLINICAL BANNER ─── */}
      <div
        style={{
          background: patientProblems.length > 0 ? '#FFF8F8' : '#F6FAF7',
          border: patientProblems.length > 0 ? '1.5px solid #FECDD3' : '1.5px solid #A7F3D0',
          borderRadius: '16px',
          padding: '14px 20px',
          marginBottom: '18px',
          boxShadow: '0 2px 12px rgba(225, 29, 72, 0.05)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px', marginBottom: '8px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            {patientProblems.length > 0 ? (
              <ShieldAlert size={20} color="#E11D48" />
            ) : (
              <CheckCircle2 size={20} color="#10B981" />
            )}
            <span style={{ fontSize: '14px', fontWeight: '800', color: patientProblems.length > 0 ? '#9F1239' : '#065F46' }}>
              {patientProblems.length > 0
                ? `Patient-Specific Problem Zones (${patientProblems.length} Identified)`
                : 'No Severe Anatomical Stress Hotspots Detected'}
            </span>
          </div>

          <span style={{ fontSize: '11.5px', color: '#64748B', fontWeight: '500' }}>
            Click any problem to view affected structure & clinical pathology
          </span>
        </div>

        {patientProblems.length > 0 ? (
          <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', marginTop: '6px' }}>
            {patientProblems.map((problem) => {
              const isActive = activeProblemForSelected && activeProblemForSelected.id === problem.id;
              return (
                <button
                  key={problem.id}
                  onClick={() => handleFocusProblem(problem)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 14px',
                    borderRadius: '12px',
                    border: isActive ? '2px solid #E11D48' : '1px solid #FDA4AF',
                    background: isActive ? '#FFE4E6' : '#FFFFFF',
                    color: '#9F1239',
                    fontSize: '12.5px',
                    fontWeight: '700',
                    cursor: 'pointer',
                    boxShadow: isActive ? '0 2px 10px rgba(225, 29, 72, 0.2)' : '0 1px 4px rgba(0,0,0,0.03)',
                    transition: 'all 0.2s'
                  }}
                >
                  <AlertTriangle size={15} color="#E11D48" />
                  <span><strong>{problem.affectedNames}</strong>: {problem.title}</span>
                  <span style={{ fontSize: '11px', background: '#E11D48', color: '#fff', padding: '2px 6px', borderRadius: '6px' }}>
                    {problem.trigger}
                  </span>
                </button>
              );
            })}
          </div>
        ) : (
          <p style={{ margin: 0, fontSize: '13px', color: '#065F46', lineHeight: '1.5' }}>
            Patient vital indicators (blood pressure, cholesterol, glucose) are within normative physiological ranges. Coronary perfusion resistance and ventricular afterload are at baseline.
          </p>
        )}
      </div>

      {/* ─── 3D VIEWPORT WITH ENCYCLOPAEDIA BRITANNICA STYLE LABELS & LINES ─── */}
      <div
        ref={containerRef}
        style={{
          position: 'relative',
          width: '100%',
          height: '660px',
          borderRadius: '24px',
          overflow: 'hidden',
          // Pristine light pastel studio background
          background: 'radial-gradient(circle at 50% 35%, #FFFFFF 0%, #F5F9FC 55%, #EAF1F8 100%)',
          border: '1.5px solid #DCE6F1',
          boxShadow: '0 12px 36px rgba(79, 140, 219, 0.10)',
          marginBottom: '20px'
        }}
      >
        {/* Three.js Canvas */}
        <Canvas camera={{ position: [0, 0, 4.3], fov: 40 }}>
          <CameraController resetTrigger={resetTrigger} />

          {/* Pristine Medical Studio Lighting */}
          <ambientLight intensity={0.95} color="#FFFFFF" />
          <directionalLight position={[6, 8, 7]} intensity={1.3} color="#FFFFFF" castShadow />
          <directionalLight position={[-6, 4, -4]} intensity={0.6} color="#DDEFF7" />
          <directionalLight position={[0, -6, 5]} intensity={0.4} color="#FFF0E5" />
          <pointLight position={[0, 0, 5]} intensity={0.8} color="#FFFFFF" />

          <Suspense fallback={<HeartModelLoader />}>
            <Heart3DScene
              theme={theme}
              riskLevel={riskIndex / 100}
              autoRotate={autoRotate}
              rotationSpeed={rotationSpeed}
              activeAngle={activeAngle}
              highlightedStructure={highlightedId}
              problemStructures={problemStructureIds}
              onUpdateProjections={handleUpdateProjections}
            />
          </Suspense>

          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
            minDistance={2.2}
            maxDistance={7.0}
            target={[0, 0, 0]}
            dampingFactor={0.05}
          />
        </Canvas>

        {/* ─── SVG LEADER LINES OVERLAY (BRITANNICA DIAGRAM STYLE) ─── */}
        {showLabels && (
          <svg
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
              zIndex: 5
            }}
          >
            {/* Draw lines from Left Margin Labels to 3D Heart Surface */}
            {LEFT_STRUCTURES.map((item, idx) => {
              const target = screenCoords[item.id];
              if (!target || !target.inZ) return null;

              // Left margin label line start point:
              // Fixed Y slots on left margin
              const startX = 265;
              const startY = 95 + idx * 75;
              const isFacing = target.isFacingFront;
              const isProblem = problemStructureIds.has(item.id);
              const isSelected = highlightedId === item.id;
              const lineColor = isProblem ? '#E11D48' : isSelected ? '#2563EB' : '#64748B';

              return (
                <g key={`left-line-${item.id}`} opacity={isFacing ? (isProblem || isSelected ? 1 : 0.75) : 0.2}>
                  {/* Clean angled leader line */}
                  <polyline
                    points={`${startX},${startY} ${startX + (target.x - startX) * 0.45},${startY} ${target.x},${target.y}`}
                    fill="none"
                    stroke={lineColor}
                    strokeWidth={isProblem ? 2 : isSelected ? 2 : 1.2}
                    strokeDasharray={isFacing ? 'none' : '3 3'}
                  />

                  {/* Surface anchor dot on the heart */}
                  <circle
                    cx={target.x}
                    cy={target.y}
                    r={isProblem ? 5 : isSelected ? 4.5 : 3}
                    fill={lineColor}
                  />

                  {/* Pulsing ring on problem structures */}
                  {isProblem && (
                    <circle
                      cx={target.x}
                      cy={target.y}
                      r={10}
                      fill="none"
                      stroke="#E11D48"
                      strokeWidth={1.8}
                      opacity={0.8}
                    />
                  )}
                </g>
              );
            })}

            {/* Draw lines from Right Margin Labels to 3D Heart Surface */}
            {RIGHT_STRUCTURES.map((item, idx) => {
              const target = screenCoords[item.id];
              if (!target || !target.inZ) return null;

              // Right margin label line start point:
              const containerWidth = containerRef.current?.clientWidth || 1000;
              const startX = containerWidth - 265;
              const startY = 85 + idx * 72;
              const isFacing = target.isFacingFront;
              const isProblem = problemStructureIds.has(item.id);
              const isSelected = highlightedId === item.id;
              const lineColor = isProblem ? '#E11D48' : isSelected ? '#2563EB' : '#64748B';

              return (
                <g key={`right-line-${item.id}`} opacity={isFacing ? (isProblem || isSelected ? 1 : 0.75) : 0.2}>
                  {/* Clean angled leader line */}
                  <polyline
                    points={`${startX},${startY} ${startX + (target.x - startX) * 0.45},${startY} ${target.x},${target.y}`}
                    fill="none"
                    stroke={lineColor}
                    strokeWidth={isProblem ? 2 : isSelected ? 2 : 1.2}
                    strokeDasharray={isFacing ? 'none' : '3 3'}
                  />

                  {/* Surface anchor dot on the heart */}
                  <circle
                    cx={target.x}
                    cy={target.y}
                    r={isProblem ? 5 : isSelected ? 4.5 : 3}
                    fill={lineColor}
                  />

                  {/* Pulsing ring on problem structures */}
                  {isProblem && (
                    <circle
                      cx={target.x}
                      cy={target.y}
                      r={10}
                      fill="none"
                      stroke="#E11D48"
                      strokeWidth={1.8}
                      opacity={0.8}
                    />
                  )}
                </g>
              );
            })}
          </svg>
        )}

        {/* ─── LEFT MARGIN ANATOMICAL LABELS (ENCYCLOPAEDIA BRITANNICA FORMAT) ─── */}
        {showLabels && (
          <div
            style={{
              position: 'absolute',
              top: '55px',
              left: '20px',
              width: '240px',
              display: 'flex',
              flexDirection: 'column',
              gap: '24px',
              zIndex: 6,
              pointerEvents: 'auto'
            }}
          >
            {LEFT_STRUCTURES.map((item) => {
              const isProblem = problemStructureIds.has(item.id);
              const isSelected = highlightedId === item.id;

              return (
                <div
                  key={item.id}
                  onClick={() => setHighlightedId(item.id)}
                  style={{
                    cursor: 'pointer',
                    textAlign: 'right',
                    padding: '4px 8px',
                    borderRadius: '8px',
                    background: isSelected ? 'rgba(238, 246, 255, 0.92)' : isProblem ? 'rgba(255, 241, 242, 0.85)' : 'transparent',
                    borderRight: isProblem ? '3px solid #E11D48' : isSelected ? '3px solid #2563EB' : '3px solid transparent',
                    transition: 'all 0.15s'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '6px' }}>
                    {isProblem && (
                      <span style={{ fontSize: '9px', fontWeight: '800', background: '#E11D48', color: '#fff', padding: '1px 5px', borderRadius: '4px' }}>
                        PROBLEM
                      </span>
                    )}
                    <span
                      style={{
                        fontSize: '12.5px',
                        fontWeight: isSelected || isProblem ? '800' : '600',
                        color: isProblem ? '#9F1239' : isSelected ? '#1E40AF' : '#1E293B',
                        lineHeight: '1.2'
                      }}
                    >
                      {item.name}
                    </span>
                  </div>
                  <div style={{ fontSize: '10.5px', color: '#64748B', fontStyle: 'italic', marginTop: '2px' }}>
                    {item.subtext}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ─── RIGHT MARGIN ANATOMICAL LABELS (ENCYCLOPAEDIA BRITANNICA FORMAT) ─── */}
        {showLabels && (
          <div
            style={{
              position: 'absolute',
              top: '45px',
              right: '20px',
              width: '240px',
              display: 'flex',
              flexDirection: 'column',
              gap: '22px',
              zIndex: 6,
              pointerEvents: 'auto'
            }}
          >
            {RIGHT_STRUCTURES.map((item) => {
              const isProblem = problemStructureIds.has(item.id);
              const isSelected = highlightedId === item.id;

              return (
                <div
                  key={item.id}
                  onClick={() => setHighlightedId(item.id)}
                  style={{
                    cursor: 'pointer',
                    textAlign: 'left',
                    padding: '4px 8px',
                    borderRadius: '8px',
                    background: isSelected ? 'rgba(238, 246, 255, 0.92)' : isProblem ? 'rgba(255, 241, 242, 0.85)' : 'transparent',
                    borderLeft: isProblem ? '3px solid #E11D48' : isSelected ? '3px solid #2563EB' : '3px solid transparent',
                    transition: 'all 0.15s'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <span
                      style={{
                        fontSize: '12.5px',
                        fontWeight: isSelected || isProblem ? '800' : '600',
                        color: isProblem ? '#9F1239' : isSelected ? '#1E40AF' : '#1E293B',
                        lineHeight: '1.2'
                      }}
                    >
                      {item.name}
                    </span>
                    {isProblem && (
                      <span style={{ fontSize: '9px', fontWeight: '800', background: '#E11D48', color: '#fff', padding: '1px 5px', borderRadius: '4px' }}>
                        PROBLEM
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: '10.5px', color: '#64748B', fontStyle: 'italic', marginTop: '2px' }}>
                    {item.subtext}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ─── TOP TOOLBAR: 360° ROTATION, PLAY/PAUSE, AND ANGLE PRESETS ─── */}
        <div
          style={{
            position: 'absolute',
            top: '16px',
            left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            background: 'rgba(255, 255, 255, 0.94)',
            backdropFilter: 'blur(12px)',
            padding: '6px 14px',
            borderRadius: '24px',
            border: '1px solid #DCE6F1',
            boxShadow: '0 4px 16px rgba(0,0,0,0.06)',
            zIndex: 10
          }}
        >
          {/* Continuous Rotation Toggle Button */}
          <button
            onClick={() => setAutoRotate(!autoRotate)}
            style={{
              background: autoRotate ? '#EFF6FF' : '#FFFFFF',
              border: autoRotate ? '1.5px solid #3B82F6' : '1px solid #CBD5E1',
              borderRadius: '16px',
              padding: '6px 12px',
              color: autoRotate ? '#1D4ED8' : '#334155',
              fontSize: '11.5px',
              fontWeight: '700',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              transition: 'all 0.15s'
            }}
          >
            {autoRotate ? <Pause size={13} /> : <Play size={13} />}
            {autoRotate ? '360° Rotating (Active)' : 'Spin 360° (Paused)'}
          </button>

          {/* Preset Angle Buttons to inspect each corner */}
          <div style={{ display: 'flex', gap: '4px', borderLeft: '1px solid #E2E8F0', paddingLeft: '8px' }}>
            {[
              { label: 'Front (Anterior)', angle: 0 },
              { label: 'Left Side (LCx)', angle: Math.PI / 2 },
              { label: 'Back (Posterior)', angle: Math.PI },
              { label: 'Right Side (RCA)', angle: -Math.PI / 2 }
            ].map((p) => (
              <button
                key={p.label}
                onClick={() => handleAngleSelect(p.angle)}
                style={{
                  background: !autoRotate && Math.abs(activeAngle - p.angle) < 0.2 ? '#E8F3FA' : 'transparent',
                  border: !autoRotate && Math.abs(activeAngle - p.angle) < 0.2 ? '1px solid #4F8CDB' : 'none',
                  borderRadius: '12px',
                  padding: '4px 8px',
                  fontSize: '11px',
                  fontWeight: '600',
                  color: !autoRotate && Math.abs(activeAngle - p.angle) < 0.2 ? '#1E40AF' : '#475569',
                  cursor: 'pointer'
                }}
              >
                {p.label}
              </button>
            ))}
          </div>

          {/* Toggle Labels */}
          <button
            onClick={() => setShowLabels(!showLabels)}
            style={{
              background: showLabels ? '#FFFFFF' : '#F1F5F9',
              border: '1px solid #CBD5E1',
              borderRadius: '16px',
              padding: '5px 10px',
              fontSize: '11px',
              fontWeight: '600',
              color: '#475569',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            {showLabels ? <Eye size={12} /> : <EyeOff size={12} />}
            {showLabels ? 'Labels ON' : 'Labels OFF'}
          </button>

          {/* Reset Camera */}
          <button
            onClick={() => {
              setResetTrigger(c => c + 1);
              setAutoRotate(true);
            }}
            style={{
              background: '#FFFFFF',
              border: '1px solid #CBD5E1',
              borderRadius: '16px',
              padding: '5px 10px',
              fontSize: '11px',
              fontWeight: '600',
              color: '#475569',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '4px'
            }}
          >
            <RotateCcw size={12} />
            Reset
          </button>
        </div>

        {/* ─── BOTTOM-LEFT TELEMETRY STATS ─── */}
        <div
          style={{
            position: 'absolute',
            bottom: '16px',
            left: '20px',
            background: 'rgba(255, 255, 255, 0.94)',
            backdropFilter: 'blur(10px)',
            padding: '10px 16px',
            borderRadius: '14px',
            border: '1px solid #E2E8F0',
            boxShadow: '0 2px 8px rgba(0,0,0,0.04)',
            display: 'flex',
            alignItems: 'center',
            gap: '16px',
            zIndex: 10
          }}
        >
          <div>
            <div style={{ fontSize: '10px', fontWeight: '700', textTransform: 'uppercase', color: '#64748B' }}>
              Simulated Pulse Rate
            </div>
            <div style={{ fontSize: '15px', fontWeight: '800', color: theme.badgeText, display: 'flex', alignItems: 'center', gap: '5px' }}>
              <Heart size={14} color={theme.primary} />
              {theme.bpm} BPM
            </div>
          </div>
          <div style={{ borderLeft: '1px solid #E2E8F0', paddingLeft: '14px' }}>
            <div style={{ fontSize: '10px', fontWeight: '700', textTransform: 'uppercase', color: '#64748B' }}>
              AI Risk Index
            </div>
            <div style={{ fontSize: '15px', fontWeight: '800', color: '#0F172A' }}>
              {riskIndex}/100 ({prediction?.risk_category || 'BASELINE'})
            </div>
          </div>
        </div>

        {/* ─── BOTTOM-RIGHT INSTRUCTION ─── */}
        <div
          style={{
            position: 'absolute',
            bottom: '16px',
            right: '20px',
            background: 'rgba(255, 255, 255, 0.92)',
            backdropFilter: 'blur(8px)',
            border: '1px solid #E2E8F0',
            borderRadius: '10px',
            padding: '6px 12px',
            fontSize: '11px',
            color: '#64748B',
            fontWeight: '500',
            zIndex: 10
          }}
        >
          ⚕️ Click any label or drag to rotate 3D heart freely
        </div>
      </div>

      {/* ─── DETAILED CLINICAL INSPECTOR CARD ─── */}
      <div
        className="card"
        style={{
          background: '#FFFFFF',
          border: '1px solid var(--border-color)',
          borderLeft: activeProblemForSelected ? '5px solid #E11D48' : `5px solid ${selectedStructureData.color}`,
          borderRadius: '18px',
          padding: '20px 24px',
          marginBottom: '20px',
          boxShadow: '0 4px 16px rgba(79, 140, 219, 0.06)'
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={{
                background: activeProblemForSelected ? '#FFF1F2' : theme.badgeBg,
                color: activeProblemForSelected ? '#E11D48' : selectedStructureData.color,
                padding: '10px',
                borderRadius: '12px',
                border: activeProblemForSelected ? '1px solid #FECDD3' : `1px solid ${theme.badgeBorder}`
              }}
            >
              <Heart size={22} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <h3 style={{ fontSize: '18px', fontWeight: '800', margin: 0, color: 'var(--text-primary)', textTransform: 'capitalize' }}>
                  {selectedStructureData.name}
                </h3>
                {activeProblemForSelected && (
                  <span
                    style={{
                      fontSize: '10.5px',
                      background: '#E11D48',
                      color: '#fff',
                      padding: '2px 8px',
                      borderRadius: '8px',
                      fontWeight: '800',
                      letterSpacing: '0.04em'
                    }}
                  >
                    IDENTIFIED PATIENT STRESS ZONE
                  </span>
                )}
              </div>
              <span style={{ fontSize: '12px', color: 'var(--text-secondary)', fontWeight: '500' }}>
                {selectedStructureData.subtext}
              </span>
            </div>
          </div>
        </div>

        <p style={{ fontSize: '14px', color: 'var(--text-secondary)', lineHeight: '1.6', margin: '0 0 12px 0' }}>
          <strong style={{ color: 'var(--text-primary)' }}>Anatomical Structure & Function: </strong>
          {selectedStructureData.role}
        </p>

        {/* Clinical Patient Vulnerability Details */}
        <div
          style={{
            fontSize: '13.5px',
            color: '#1E293B',
            background: activeProblemForSelected ? '#FFF8F8' : 'var(--bg-primary)',
            border: activeProblemForSelected ? '1px solid #FECDD3' : '1px solid var(--border-light)',
            padding: '14px 18px',
            borderRadius: '12px',
            lineHeight: '1.6'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
            <Stethoscope size={16} color={activeProblemForSelected ? '#E11D48' : theme.primary} />
            <strong style={{ color: activeProblemForSelected ? '#9F1239' : theme.badgeText }}>
              {activeProblemForSelected ? 'Pathophysiological Impact for this Patient:' : 'Clinical Relevance:'}
            </strong>
          </div>
          <div>
            {activeProblemForSelected ? (
              <div>
                <p style={{ margin: '0 0 6px 0' }}>
                  <strong>{activeProblemForSelected.title} ({activeProblemForSelected.trigger}): </strong>
                  {activeProblemForSelected.mechanism}
                </p>
                <div style={{ fontSize: '12.5px', color: '#9F1239', fontWeight: '600' }}>
                  Recommended Clinical Follow-up: {activeProblemForSelected.clinicalAction}
                </div>
              </div>
            ) : (
              <span>
                Under normative physiological loads, this structure operates within standard compliance boundaries with resting hemodynamic pacing.
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ─── EDUCATIONAL COMPARISON REFERENCE ─── */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '16px'
        }}
      >
        <div
          className="card"
          style={{
            padding: '18px',
            background: '#FFFFFF',
            border: '1px solid var(--border-color)',
            borderRadius: '16px'
          }}
        >
          <h4 style={{ fontSize: '14px', fontWeight: '700', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
            <RefreshCw size={16} color="var(--accent-primary)" />
            360° Real Heart Inspection
          </h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
            • <strong>Automatic 360° Turntable:</strong> Continuous rotation displays anterior ventricles, lateral coronary grooves, and posterior venous inflows.<br />
            • <strong>Corner Presets:</strong> Quickly snap between Front (LAD & ventricles), Left side (LCx), Back (pulmonary veins), and Right side (RCA).<br />
            • <strong>Mouse Drag:</strong> Left-click and drag anywhere to inspect custom angles.
          </p>
        </div>

        <div
          className="card"
          style={{
            padding: '18px',
            background: '#FFFFFF',
            border: '1px solid var(--border-color)',
            borderRadius: '16px'
          }}
        >
          <h4 style={{ fontSize: '14px', fontWeight: '700', marginBottom: '8px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-primary)' }}>
            <AlertTriangle size={16} color="#E11D48" />
            Patient Vulnerability Mapping
          </h4>
          <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: '1.6', margin: 0 }}>
            • <strong>Blood Pressure (&ge; 140 mmHg):</strong> Targets Left Ventricle & Aorta (Hypertrophy & high afterload).<br />
            • <strong>Cholesterol (Level 2 or 3):</strong> Targets LAD & RCA coronary arteries (Accelerated atheroma progression).<br />
            • <strong>Widened Pulse Pressure (&ge; 60 mmHg):</strong> Targets Aortic Arch (Arterial stiffness & compliance loss).
          </p>
        </div>
      </div>
    </div>
  );
}
