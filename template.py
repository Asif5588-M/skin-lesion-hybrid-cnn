"""
template.py
------------
Run this once to auto-generate the full project folder/file structure
for the Hybrid Skin Lesion Classification project (Project 1).

Usage:
    python template.py

This follows the Cookiecutter Data Science (v2) convention, adapted
for a medical-imaging deep learning project with Colab+Drive training.
"""

import os
from pathlib import Path

# ---- Project name (change per project) ----
PROJECT_NAME = "skin-lesion-hybrid-cnn"

# ---- List of files/folders to create ----
list_of_files = [
    ".github/workflows/.gitkeep",          # placeholder for future CI/CD
    f"data/raw/.gitkeep",                  # original downloaded dataset (NOT pushed to GitHub, see .gitignore)
    f"data/processed/.gitkeep",            # cleaned/resized images, train-val-test split
    f"data/external/.gitkeep",             # any third-party reference data
    "notebooks/01_eda.ipynb",              # exploratory data analysis
    "notebooks/02_colab_training.ipynb",   # the notebook you'll actually run on Google Colab
    "src/__init__.py",
    "src/config.py",                       # paths, hyperparameters, constants
    "src/data/__init__.py",
    "src/data/make_dataset.py",            # download/organize raw data
    "src/data/preprocessing.py",           # resizing, normalization, augmentation
    "src/features/__init__.py",
    "src/features/build_features.py",      # feature engineering if needed
    "src/models/__init__.py",
    "src/models/architectures.py",         # your hybrid CNN architecture definitions
    "src/models/train.py",                 # training loop
    "src/models/predict.py",               # inference script
    "src/models/optimize_nsga2.py",        # NSGA-II architecture optimization (Phase 3)
    "src/explainability/__init__.py",
    "src/explainability/gradcam.py",
    "src/explainability/shap_explain.py",
    "src/visualization/__init__.py",
    "src/visualization/plots.py",
    "models/.gitkeep",                     # trained model checkpoints (small ones only; large -> Drive)
    "reports/figures/.gitkeep",
    "reports/results.md",
    "deployment/app.py",                   # Streamlit/FastAPI demo app
    "deployment/requirements.txt",
    ".gitignore",
    "requirements.txt",
    "environment.yml",
    "README.md",
    "params.yaml",                         # hyperparameters as config (clean, reproducible)
]


def create_project_structure():
    project_root = Path(PROJECT_NAME)

    for filepath in list_of_files:
        filepath = project_root / filepath
        filedir = filepath.parent

        # create directories
        filedir.mkdir(parents=True, exist_ok=True)

        # create empty file if it doesn't already exist
        if not filepath.exists() or filepath.stat().st_size == 0:
            filepath.touch()
            print(f"Created: {filepath}")
        else:
            print(f"Already exists, skipped: {filepath}")

    print(f"\nProject '{PROJECT_NAME}' structure created successfully at: {project_root.resolve()}")


if __name__ == "__main__":
    create_project_structure()