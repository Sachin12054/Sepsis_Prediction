# 🔧 Utilities & Helper Functions

## 📋 Available Utilities

### 📊 Data Validation
- **`check_data.py`** - Validate dataset integrity and structure
- **`check_stft_data.py`** - Verify STFT feature quality
- **`extract_labels.py`** - Extract and validate sepsis labels
- **`debug_features.py`** - Debug feature engineering issues

### 🤖 Model Validation  
- **`check_models.py`** - Validate trained model performance
- **`simple_validation.py`** - Basic model testing
- **`corrected_validation.py`** - Enhanced validation with corrections
- **`final_validation.py`** - Complete model validation suite

### 📈 Analysis & Reporting
- **`production_summary.py`** - Generate production reports
- **`production_validation.py`** - Validate production readiness
- **`final_project_summary.py`** - Complete project documentation
- **`final_summary_clean.py`** - Clean summary generation

## 🚀 Usage Examples

### Quick Data Check
```python
# Run data validation
python utilities/check_data.py

# Check STFT features
python utilities/check_stft_data.py
```

### Model Validation
```python
# Basic model check
python utilities/check_models.py

# Complete validation
python utilities/final_validation.py
```

### Generate Reports
```python
# Production summary
python utilities/production_summary.py

# Project documentation
python utilities/final_project_summary.py
```

## 📁 Integration with Main Model

These utilities support the main ensemble model (`ensemble_learning_pipeline.ipynb`) by providing:
- **Data quality assurance** before training
- **Model performance validation** after training  
- **Production readiness checks** before deployment
- **Comprehensive reporting** for documentation

## 🔄 Workflow Integration

1. **Before Training:** Use data validation utilities
2. **During Development:** Use debug utilities for troubleshooting
3. **After Training:** Use model validation utilities
4. **Before Deployment:** Use production validation utilities
5. **For Documentation:** Use summary generation utilities

## 🖥️ Integration with Launcher System

These utilities are now integrated with the organized launcher system:
- **Main Backend:** `../launchers/main.bat` (Option 5: System Diagnostics)
- **Python Interface:** `../launchers/sepsis_backend.py --mode validate`
- **Root Access:** `../start.bat` for quick navigation

---
*Helper functions to support the main ensemble learning pipeline*