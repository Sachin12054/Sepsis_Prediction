# 🏥 Sepsis Prediction Project Configuration

## 📂 Directory Structure Guide

### 🔥 MAIN MODEL (Current Location)
- `ensemble_learning_pipeline.ipynb` - **PRIMARY SEPSIS PREDICTION MODEL**
  - Location: Root directory (as requested)
  - Purpose: Main ensemble model with clinical optimization
  - Status: Hospital-approved, 100% sensitivity

### 📊 STEP-BY-STEP ANALYSIS
- Location: `notebooks/step_by_step_analysis/`
- Purpose: Detailed development process documentation
- Files: Step01 through Step10 notebooks

### 🤖 MODELS & ARTIFACTS
- `models/` - Trained model files
  - `clinical_sepsis_model.pkl` - Final hospital-ready model
  - `baseline/` - Traditional ML models
  - `advanced/` - Optimized models
- `ensemble_models/` - Multi-model ensembles

### 📁 DATA PIPELINE
- `data/raw/` - Original patient files (20,000+ patients)
- `data/processed/` - Cleaned datasets
- `data/stft_features/` - Frequency-domain features

### 📈 ANALYSIS OUTPUTS
- `plots/` - Visualizations and charts
- `results/` - Performance metrics and scores
- `reports/` - Analysis documentation
- `final_report/` - Complete project summary

### 🔧 UTILITIES & TOOLS
- `utilities/` - Helper scripts and validation tools
- `production_pipeline/` - Deployment-ready code

## 🎯 Usage Workflow

### For Model Development:
1. Start with `ensemble_learning_pipeline.ipynb` (main model)
2. Reference step-by-step notebooks for detailed analysis
3. Use utilities for data validation and debugging

### For Clinical Deployment:
1. Load model from `models/clinical_sepsis_model.pkl`
2. Use production pipeline for deployment
3. Follow clinical guidelines in main ensemble notebook

### For Research & Analysis:
1. Explore step-by-step notebooks for methodology
2. Review plots and results for insights
3. Check final report for complete documentation

## 📋 File Naming Convention

- **Main Model:** `ensemble_learning_pipeline.ipynb` (unchanged location)
- **Analysis Steps:** `Step[XX]_[Description].ipynb`
- **Utilities:** `[function]_[purpose].py`
- **Models:** `[type]_[optimization].pkl`
- **Results:** `[analysis]_[metric].csv`

## 🚀 Quick Access

### Most Important Files:
1. **Main Model:** `ensemble_learning_pipeline.ipynb`
2. **Final Model:** `models/clinical_sepsis_model.pkl`
3. **Project Guide:** `README.md`
4. **This Guide:** `PROJECT_STRUCTURE.md`

### For Different Use Cases:
- **Clinical Use:** Main ensemble notebook + clinical model
- **Research:** Step-by-step notebooks + analysis results
- **Development:** Main notebook + utilities + data pipeline
- **Deployment:** Production pipeline + final model

---
*Project organized for easy navigation while keeping main ensemble model in root as requested.*