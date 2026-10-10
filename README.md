# 🫀 CardioVision AI: Interactive Cardiovascular Risk Visualization & Prediction

> **Hackathon Track A** — An interactive 3D cardiovascular visualization platform that predicts cardiovascular disease risk from clinical patient data and dynamically maps the model's risk output onto an interactive 3D human anatomical heart model.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)
[![Three.js](https://img.shields.io/badge/Three.js-WebGL-black.svg)](https://threejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-0.8034%20ROC--AUC-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Deployment](https://img.shields.io/badge/PythonAnywhere-Ready-1D8348.svg)](https://www.pythonanywhere.com/)

---

## 🌟 Highlights & Key Features

* **3D Anatomical Heart Engine:** Real 3D anatomical heart (`heart.glb`) rendered in WebGL with continuous $360^\circ$ turntable rotation, OrbitControls (pan, zoom, tilt), and lighting shaders.
* **Encyclopaedia Britannica Plate Design:** 12 perimeter margin callout tags connected by dynamic screen-space SVG Bezier leader lines (`Vector3.project(camera)`) anchoring directly to 3D surface coordinates with zero text overlap.
* **Patient Vulnerability Mapping:** Dynamically correlates patient clinical risk factors (hypertension, dyslipidemia, pulse pressure overload, metabolic obesity) with vulnerable heart regions (Left Ventricle, Aorta, Coronary Arteries).
* **Calibrated Machine Learning:** Benchmarked across 6 architectures (LightGBM champion: **0.8034 ROC-AUC**, **0.7289 F1**, **73.82% Accuracy**) with isotonic probability calibration.
* **Explainable AI (XAI):** TreeSHAP waterfall feature attributions, global feature importance, and interactive counterfactual simulation.
* **Full-Stack & PythonAnywhere Ready:** Unified WSGI app (`wsgi.py`) that serves the pre-built React 3D frontend and ML REST API from a single server.

---

## 🏗️ System Architecture

```
                                  +-----------------------------+
                                  |     Clinical Patient Data   |
                                  | (BP, Age, Chol, BMI, Habits)|
                                  +--------------+--------------+
                                                 |
                                                 v
                                  +-----------------------------+
                                  |     LightGBM ML Pipeline    |
                                  |  (Calibrated Probabilities) |
                                  +--------------+--------------+
                                                 |
                       +-------------------------+-------------------------+
                       |                                                   |
                       v                                                   v
         +---------------------------+                       +---------------------------+
         |     Explainable AI (XAI)  |                       |   3D Anatomical Engine    |
         |  - TreeSHAP Attributions  |                       |  - Three.js WebGL Heart   |
         |  - Counterfactual Deltas  |                       |  - 12 Margin Callouts     |
         |  - Model Insights Charts  |                       |  - SVG Bezier Leader Lines|
         +---------------------------+                       +---------------------------+
```

---

## 🚀 Quick Start (Local Run)

### Backend (FastAPI):
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app/main.py
```
*API runs at `http://localhost:8000` (Interactive docs at `http://localhost:8000/docs`).*

### Frontend (React + Vite):
```bash
cd frontend
npm install
npm run dev
```
*Frontend opens at `http://localhost:5173`.*

---

## ☁️ Deploy to PythonAnywhere (Under 2 Minutes)

CardioVision AI includes a self-contained WSGI server (`wsgi.py`) and pre-built production UI (`frontend/dist/`).

1. **Clone in PythonAnywhere Bash Console:**
   ```bash
   git clone https://github.com/KEERTHANAHL08/cardiovision-ai.git
   cd cardiovision-ai
   mkvirtualenv --python=/usr/bin/python3.10 cardiovision
   pip install -r requirements_deploy.txt
   ```
2. **Configure Web Tab:**
   - Add new Web App ➔ **Manual configuration** ➔ **Python 3.10**
   - Source code: `/home/<your-username>/cardiovision-ai`
   - Virtualenv: `/home/<your-username>/.virtualenvs/cardiovision`
3. **Set WSGI configuration file:**
   ```python
   import sys
   project_home = '/home/<your-username>/cardiovision-ai'
   if project_home not in sys.path:
       sys.path.insert(0, project_home)
   from wsgi import application
   ```
4. **Click Reload:** Your app is instantly live at `https://<your-username>.pythonanywhere.com`!

See [DEPLOY_PYTHONANYWHERE.md](DEPLOY_PYTHONANYWHERE.md) for detailed configuration options.

---

## 📄 Documentation

* **Master Technical PDF:** [CardioVision_AI_Complete_Documentation.pdf](CardioVision_AI_Complete_Documentation.pdf) *(Exhaustive publication-grade PDF covering all methodologies, models, mathematical derivations, and 3D coordinate mapping).*

---

## ⚖️ Disclaimer

*CardioVision AI is an educational, research, and technical demonstrator built for Hackathon Track A. It is not an FDA-cleared diagnostic device and must not be used for clinical decision-making without oversight from qualified medical practitioners.*
