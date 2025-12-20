# Project Setup Guide

This file explains how to set up the project from scratch after cloning, create and activate a virtual environment, install dependencies from `requirements.txt`, run the project, update dependencies, and access the documentation (PDF). Copy this directly into `README.md`.

---

## 📘 Important: Read Our Three Step Process

This repository includes a document named:




# Project Setup Guide

This file explains how to set up the project from scratch after cloning, create and activate a virtual environment, install dependencies from `requirements.txt`, run the project, update dependencies, and common troubleshooting tips. Copy this into `README.md` at the root of your repo.

---

## Quick-start (all steps in one sequence)

```bash
# 1. Clone the repository and enter the folder
git clone <your-repository-url>
cd <project-folder>

# 2. Create a virtual environment (single command that works on most systems)
python -m venv venv

# 3. Activate the virtual environment
# - Windows (PowerShell):
#   venv\Scripts\Activate
# - Windows (cmd.exe):
#   venv\Scripts\activate.bat
# - macOS / Linux:
#   source venv/bin/activate

# Example (macOS/Linux)
source venv/bin/activate

# 4. Upgrade pip (recommended)
pip install --upgrade pip

# 5. Install dependencies from requirements.txt
pip install -r requirements.txt

# 6. Run the project (adjust entrypoint as needed)
python main.py

# Example for FastAPI apps:
# uvicorn main:app --reload

# 7. When finished, deactivate the environment
deactivate
