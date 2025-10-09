# 🏗️ Project Reorganization Plan

## Current Issues
- Too many files in root directory (31+ files)
- Dashboard servers scattered with unclear naming
- Test files mixed with main code
- Batch files not grouped
- HTML files in root directory

## Proposed Structure

```
📁 Sepsis_Prediction/
├── 🔥 ensemble_learning_pipeline.ipynb    # MAIN MODEL (stays in root)
├── 📋 README.md                          # Project overview (stays in root)
├── ⚡ start.bat                          # Quick launcher (stays in root)
│
├── 📱 src/                               # SOURCE CODE
│   ├── dashboard/                        # Dashboard servers
│   │   ├── dashboard_server.py           # Main dashboard server
│   │   ├── dashboard_server_enhanced.py  # Enhanced version
│   │   ├── dashboard_server_icu.py       # ICU-compatible
│   │   └── dashboard_server_icu_fixed.py # Fixed ICU version
│   ├── models/                           # Model creation & management
│   │   ├── create_model.py               # Model creation
│   │   ├── create_real_models.py         # Real model creation
│   │   ├── export_production_models.py   # Production export
│   │   └── extract_real_models.py        # Model extraction
│   ├── data_processing/                  # Data handling
│   │   ├── icu_data_converter.py         # ICU data conversion
│   │   └── generate_test_data.py         # Test data generation
│   └── utils/                            # Utility scripts
│       ├── check_server.py               # Server health check
│       └── restart_dashboard.py          # Server restart
│
├── 🌐 web/                               # WEB INTERFACE
│   ├── enhanced_dashboard.html           # Enhanced dashboard
│   ├── sepsis_dashboard.html             # Original dashboard  
│   └── sepsis_dashboard_live.html        # Live dashboard
│
├── 🚀 scripts/                           # AUTOMATION SCRIPTS
│   ├── main.bat                          # Main launcher
│   ├── start_system.bat                  # System launcher
│   └── launchers/                        # Additional launchers
│
├── 🧪 tests/                             # TEST FILES
│   ├── test_dashboard.py                 # Dashboard tests
│   ├── test_feature_fix.py               # Feature fix tests
│   ├── test_integration.py               # Integration tests
│   ├── test_butterworth.py               # Butterworth tests
│   ├── test_real_patient.py              # Real patient tests
│   ├── butterworth_demo.py               # Butterworth demo
│   ├── demo_icu_analysis.py              # ICU demo
│   ├── quick_api_test.py                 # API tests
│   └── run_sepsis_model.py               # Model run tests
│
├── 📊 data/                              # DATA (existing structure)
│   ├── raw/
│   ├── processed/
│   ├── stft_features/
│   └── test_patients/
│       ├── patients_with_sepsis.csv      # Moved from root
│       ├── sepsis_patients.csv           # Moved from root
│       └── test_patient_data.csv         # Moved from root
│
├── 🤖 models/                            # MODELS (existing structure)
├── 📈 analysis/                          # ANALYSIS (existing structure)
├── 📚 notebooks/                         # NOTEBOOKS (existing structure)
├── 🔧 utilities/                         # UTILITIES (existing structure)
├── 🏭 production_pipeline/               # PRODUCTION (existing structure)
├── 📋 docs/                              # DOCUMENTATION (existing structure)
├── 🔍 monitoring/                        # MONITORING (existing structure)
├── 📊 results/                           # RESULTS (existing structure)
├── 📡 signal_processing/                 # SIGNAL PROCESSING (existing structure)
├── 🗃️ ensemble_models/                   # ENSEMBLE MODELS (existing structure)
├── 🏗️ deployment/                        # DEPLOYMENT (existing structure)
├── 🌐 api/                               # API (existing structure)
├── 📝 logs/                              # LOGS (existing structure)
└── 🗂️ catboost_info/                     # CATBOOST INFO (existing structure)
```

## Benefits of New Organization

### 🎯 **Clear Separation of Concerns**
- **Source Code**: All Python scripts organized by function
- **Web Interface**: All HTML files together
- **Tests**: All test files in dedicated directory
- **Scripts**: All batch files and launchers organized

### 🔍 **Easy Navigation**
- **Dashboard Servers**: All variations in one place
- **Model Scripts**: All model-related code together  
- **Data Files**: Test data organized in data/test_patients/
- **Documentation**: Better organization in docs/

### 🚀 **Professional Structure**
- **Clean Root**: Only essential files in root
- **Logical Grouping**: Related files together
- **Scalable**: Easy to add new components
- **Maintainable**: Clear structure for future development

## Implementation Strategy

1. **Create new directory structure**
2. **Move files to appropriate locations**
3. **Update import statements and paths**
4. **Update batch files with new paths**
5. **Test all functionality**
6. **Update documentation**