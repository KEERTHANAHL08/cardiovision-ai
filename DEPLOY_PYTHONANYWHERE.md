# 🚀 Deploying CardioVision AI to PythonAnywhere

This guide provides the exact, copy-paste instructions to deploy **CardioVision AI** to **PythonAnywhere** so anyone on the internet can view and interact with the 3D heart and real-time AI risk assessment engine.

---

## Architecture of the Deployment
* **Frontend:** React + Three.js 3D Heart + Tailwind CSS pre-compiled into `frontend/dist/`.
* **Backend & WSGI Engine:** A high-performance, unified Flask WSGI application (`wsgi.py`) that serves the compiled 3D frontend single-page app and handles all REST API prediction requests (`/predict`, `/explain`, `/health`, etc.).
* **Model:** LightGBM pre-calibrated model (`backend/models/final_model.joblib`) pre-loaded at startup.

---

## Step 1: Push Code to GitHub

In your local terminal (or inside VS Code):

1. Go to [github.com/new](https://github.com/new) and create a new repository named `cardiovision-ai` (leave it public or private).
2. In your terminal, run:
```bash
cd /Users/keerthanahl/Desktop/cardiovision-ai
git remote add origin git@github.com:KEERTHANAHL08/cardiovision-ai.git
git push -u origin main
```

---

## Step 2: Set Up on PythonAnywhere

1. Log in to [PythonAnywhere](https://www.pythonanywhere.com).
2. Open a **Bash Console** from the **Consoles** tab.
3. Clone your repository:
```bash
git clone https://github.com/KEERTHANAHL08/cardiovision-ai.git
cd cardiovision-ai
```
4. Create a virtual environment with Python 3.10 and install the deployment requirements:
```bash
mkvirtualenv --python=/usr/bin/python3.10 cardiovision
pip install -r requirements_deploy.txt
```

---

## Step 3: Configure the Web App in PythonAnywhere

1. Go to the **Web** tab in PythonAnywhere.
2. Click **Add a new web app**.
3. Choose **Manual configuration** (do NOT choose Django or Flask quickstarts).
4. Select **Python 3.10**.
5. Once created, configure the paths on the Web page:
   - **Source code:** `/home/<your-username>/cardiovision-ai`
   - **Working directory:** `/home/<your-username>/cardiovision-ai`
   - **Virtualenv:** `/home/<your-username>/.virtualenvs/cardiovision`
6. Click on the **WSGI configuration file** link (e.g. `/var/www/<your-username>_pythonanywhere_com_wsgi.py`).
7. Replace the entire contents of that file with the following:

```python
import sys
import os

# Path to your project
project_home = '/home/<your-username>/cardiovision-ai'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Import the WSGI application
from wsgi import application
```
*(Make sure to replace `<your-username>` with your actual PythonAnywhere account username!)*

8. Save the file.

---

## Step 4: Reload and View Live!

1. Back on the **Web** tab, click the big green **Reload <your-username>.pythonanywhere.com** button.
2. Open your live URL:
   `https://<your-username>.pythonanywhere.com`
3. Anyone across the world can now:
   - View and rotate the **3D anatomical heart in 360°**.
   - Inspect margin leader lines and callouts.
   - Adjust patient clinical parameters (age, systolic/diastolic blood pressure, cholesterol, BMI, smoking, etc.).
   - Receive instant **calibrated cardiovascular disease risk predictions** and local **SHAP waterfall explanations**.
