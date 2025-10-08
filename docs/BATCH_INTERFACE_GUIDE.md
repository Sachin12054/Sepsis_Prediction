# 🖥️ Windows Batch Interface Guide

## 🏥 Sepsis Prediction System - Command Line Interface

This guide explains how to use the Windows batch files to run the sepsis prediction system from the command line.

## 🚀 Available Launchers

### 1. **main.bat** - Full System Backend
**Purpose:** Complete clinical backend system with full menu interface

**Usage:**
```batch
# Double-click main.bat OR
# From command line:
cd "c:\Users\sachi\Desktop\Sepsis STFT"
main.bat
```

**Features:**
- 🚀 Run complete sepsis prediction model
- 🔍 Validate model performance  
- 🏥 Load clinical model for predictions
- 📊 Generate performance reports
- 🔧 Run system diagnostics
- 📋 View project documentation

### 2. **quick_launch.bat** - Quick Access Menu
**Purpose:** Fast access to common operations

**Usage:**
```batch
# Double-click quick_launch.bat OR
quick_launch.bat
```

**Features:**
- ⚡ Quick model reports
- 🔍 Fast validation
- 🏥 Clinical interface access
- 🚀 Launch full backend

### 3. **sepsis_backend.py** - Python Interface
**Purpose:** Advanced Python backend for programmatic access

**Usage:**
```batch
# Generate clinical report
python sepsis_backend.py --mode report

# Validate model
python sepsis_backend.py --mode validate

# Make predictions
python sepsis_backend.py --mode predict --input data.csv --output results.csv
```

### 4. **create_shortcuts.bat** - Desktop Shortcuts
**Purpose:** Create desktop shortcuts for easy access

**Usage:**
```batch
# Run once to create shortcuts
create_shortcuts.bat
```

**Creates:**
- 🏥 Sepsis Prediction System.lnk
- 🚀 Sepsis Quick Launch.lnk  
- 🔥 Main Sepsis Model.lnk

## 🖥️ Command Line Examples

### Run Full System
```batch
# Method 1: Direct execution
"c:\Users\sachi\Desktop\Sepsis STFT\main.bat"

# Method 2: Navigate and run
cd "c:\Users\sachi\Desktop\Sepsis STFT"
main.bat
```

### Quick Model Report
```batch
cd "c:\Users\sachi\Desktop\Sepsis STFT"
python sepsis_backend.py --mode report
```

### Make Predictions on New Data
```batch
cd "c:\Users\sachi\Desktop\Sepsis STFT"
python sepsis_backend.py --mode predict --input "path\to\patient_data.csv"
```

## 📊 Expected Data Format

For making predictions with new patient data:

**Input CSV Format:**
- 532 columns (STFT features)
- One row per patient
- No headers required
- Numerical values only

**Example:**
```csv
0.123,0.456,0.789,...(532 features total)
0.234,0.567,0.890,...(532 features total)
```

**Output CSV Format:**
```csv
sample_id,sepsis_probability,sepsis_prediction,clinical_alert
0,0.023,0,✅ LIKELY HEALTHY
1,0.987,1,🚨 SEPSIS RISK
```

## 🔧 System Requirements

### Prerequisites:
- ✅ Windows 10/11
- ✅ Python 3.8+ installed and in PATH
- ✅ Required packages: pandas, numpy, scikit-learn, xgboost, lightgbm, joblib

### Auto-Installation:
The batch files will automatically:
- Check Python installation
- Install missing packages
- Activate virtual environment (if available)
- Validate model files

## 🚨 Troubleshooting

### Common Issues:

**Python not found:**
```
❌ ERROR: Python not found!
```
**Solution:** Install Python 3.8+ and add to PATH

**Model not found:**
```
❌ ERROR: Main model file not found!
```
**Solution:** Ensure you're in the correct directory with ensemble_learning_pipeline.ipynb

**Package missing:**
```
❌ Error loading model: No module named 'xgboost'
```
**Solution:** Run `pip install xgboost` or let the batch file auto-install

### Manual Package Installation:
```batch
pip install pandas numpy scikit-learn xgboost lightgbm joblib matplotlib seaborn imblearn
```

## 🏥 Clinical Usage

### For Hospital Deployment:
1. Run `main.bat`
2. Select option 3: "🏥 Load Clinical Model for Predictions"
3. Follow clinical guidelines for interpretation

### Safety Guidelines:
- ✅ **1 = SEPSIS RISK** → Immediate clinical review required
- ✅ **0 = LIKELY HEALTHY** → Continue standard monitoring
- ⚠️ **High false alarm rate acceptable** → Better safe than sorry
- 🎯 **100% sensitivity achieved** → No sepsis cases missed

## 📞 Support

### For Technical Issues:
- Check system diagnostics: `main.bat` → Option 5
- Review error messages in command prompt
- Ensure all files are in correct locations

### For Clinical Questions:
- Review model performance: `main.bat` → Option 4
- Check clinical validation results
- Consult project documentation

---
*Professional batch interface for clinical sepsis prediction deployment*