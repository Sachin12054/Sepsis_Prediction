#!/usr/bin/env python3
"""
Sepsis Patient Data Generator
=============================

Creates realistic CSV files with sepsis and non-sepsis patients
using 536 STFT features based on real physiological patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

class SepsisDataGenerator:
    def __init__(self):
        self.n_features = 536
        self.feature_names = self._generate_feature_names()
        
    def _generate_feature_names(self):
        """Generate realistic STFT feature names"""
        features = []
        
        # Vital sign STFT features (20 base signals)
        vital_signals = [
            'heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 'respiratory_rate',
            'oxygen_saturation', 'temperature', 'cardiac_output', 'stroke_volume',
            'systemic_vascular_resistance', 'central_venous_pressure', 'mean_arterial_pressure',
            'pulse_pressure', 'heart_rate_variability', 'respiratory_effort',
            'perfusion_index', 'pleth_variability_index', 'shock_index',
            'modified_early_warning_score', 'lactate_trend', 'base_excess'
        ]
        
        # STFT frequency bands (26 frequency bins, 20 time windows)
        for signal in vital_signals:
            for freq_bin in range(26):
                for time_win in range(20):
                    features.append(f'{signal}_stft_freq_{freq_bin}_time_{time_win}')
                    if len(features) >= 536:
                        break
                if len(features) >= 536:
                    break
            if len(features) >= 536:
                break
        
        # Pad if needed
        while len(features) < 536:
            features.append(f'derived_feature_{len(features)}')
        
        return features[:536]
    
    def _generate_sepsis_patient(self, patient_id, severity='moderate'):
        """Generate a sepsis patient with realistic physiological patterns"""
        np.random.seed(patient_id + 1000)  # Ensure reproducibility
        
        # Base sepsis patterns - altered physiological signals
        base_patterns = {
            'severe': {
                'heart_rate': (110, 140, 15),  # Tachycardia
                'blood_pressure': (70, 90, 10),  # Hypotension
                'respiratory_rate': (22, 35, 8),  # Tachypnea
                'temperature': (38.5, 40.5, 1.0),  # Fever
                'lactate': (4.0, 8.0, 2.0),  # High lactate
                'oxygen_sat': (88, 94, 3),  # Decreased O2
                'variability': 0.8  # High variability
            },
            'moderate': {
                'heart_rate': (95, 120, 10),
                'blood_pressure': (85, 100, 8),
                'respiratory_rate': (20, 28, 5),
                'temperature': (37.8, 39.5, 0.8),
                'lactate': (2.5, 4.5, 1.0),
                'oxygen_sat': (90, 96, 2),
                'variability': 0.6
            },
            'mild': {
                'heart_rate': (85, 105, 8),
                'blood_pressure': (95, 110, 6),
                'respiratory_rate': (18, 24, 4),
                'temperature': (37.2, 38.5, 0.5),
                'lactate': (1.8, 3.0, 0.5),
                'oxygen_sat': (92, 97, 2),
                'variability': 0.4
            }
        }
        
        patterns = base_patterns[severity]
        
        # Generate STFT features
        features = []
        
        # Generate features for each vital sign
        for i, signal in enumerate(['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 
                                   'respiratory_rate', 'oxygen_saturation', 'temperature'] + 
                                  [f'signal_{j}' for j in range(14)]):  # 20 total signals
            
            if i < 6:  # Known vital signs
                if 'heart_rate' in signal:
                    base_freq = np.random.normal(patterns['heart_rate'][1], patterns['heart_rate'][2])
                elif 'blood_pressure' in signal:
                    base_freq = np.random.normal(patterns['blood_pressure'][1], patterns['blood_pressure'][2])
                elif 'respiratory' in signal:
                    base_freq = np.random.normal(patterns['respiratory_rate'][1], patterns['respiratory_rate'][2])
                elif 'temperature' in signal:
                    base_freq = np.random.normal(patterns['temperature'][1], patterns['temperature'][2])
                elif 'oxygen' in signal:
                    base_freq = np.random.normal(patterns['oxygen_sat'][1], patterns['oxygen_sat'][2])
                else:
                    base_freq = np.random.normal(100, 20)
            else:
                base_freq = np.random.normal(80, 30)
            
            # Generate STFT spectrum (26 freq bins x 20 time windows)
            for freq_bin in range(26):
                for time_win in range(20):
                    # Sepsis creates specific frequency patterns
                    freq_factor = freq_bin / 25.0  # 0 to 1
                    time_factor = time_win / 19.0  # 0 to 1
                    
                    # Sepsis typically shows:
                    # - High energy in low frequencies (autonomic dysfunction)
                    # - Irregular patterns in mid frequencies
                    # - Reduced high-frequency variability
                    
                    if freq_factor < 0.3:  # Low frequency - high energy in sepsis
                        energy = np.abs(np.random.normal(base_freq * 0.8, base_freq * 0.4))
                    elif freq_factor < 0.7:  # Mid frequency - irregular
                        energy = np.abs(np.random.normal(base_freq * 0.4, base_freq * 0.6))
                    else:  # High frequency - reduced in sepsis
                        energy = np.abs(np.random.normal(base_freq * 0.1, base_freq * 0.2))
                    
                    # Add temporal variations
                    temporal_var = 1 + patterns['variability'] * np.sin(time_factor * 2 * np.pi) * 0.3
                    energy *= temporal_var
                    
                    # Add sepsis-specific noise
                    energy += np.random.normal(0, energy * 0.3)
                    
                    features.append(max(0, energy))  # Ensure non-negative
                    
                    if len(features) >= 536:
                        break
                if len(features) >= 536:
                    break
            if len(features) >= 536:
                break
        
        # Ensure exactly 536 features
        while len(features) < 536:
            features.append(np.random.exponential(10))
        
        return features[:536]
    
    def _generate_healthy_patient(self, patient_id):
        """Generate a healthy patient with normal physiological patterns"""
        np.random.seed(patient_id + 2000)  # Different seed space
        
        # Normal physiological ranges
        normal_patterns = {
            'heart_rate': (60, 85, 8),  # Normal HR
            'blood_pressure': (110, 130, 8),  # Normal BP
            'respiratory_rate': (12, 18, 3),  # Normal RR
            'temperature': (36.1, 37.2, 0.3),  # Normal temp
            'lactate': (0.5, 1.5, 0.3),  # Normal lactate
            'oxygen_sat': (97, 99, 1),  # Normal O2
            'variability': 0.2  # Low, healthy variability
        }
        
        features = []
        
        # Generate features for each vital sign
        for i, signal in enumerate(['heart_rate', 'blood_pressure_sys', 'blood_pressure_dia', 
                                   'respiratory_rate', 'oxygen_saturation', 'temperature'] + 
                                  [f'signal_{j}' for j in range(14)]):
            
            if i < 6:  # Known vital signs
                if 'heart_rate' in signal:
                    base_freq = np.random.normal(normal_patterns['heart_rate'][1], normal_patterns['heart_rate'][2])
                elif 'blood_pressure' in signal:
                    base_freq = np.random.normal(normal_patterns['blood_pressure'][1], normal_patterns['blood_pressure'][2])
                elif 'respiratory' in signal:
                    base_freq = np.random.normal(normal_patterns['respiratory_rate'][1], normal_patterns['respiratory_rate'][2])
                elif 'temperature' in signal:
                    base_freq = np.random.normal(normal_patterns['temperature'][1], normal_patterns['temperature'][2])
                elif 'oxygen' in signal:
                    base_freq = np.random.normal(normal_patterns['oxygen_sat'][1], normal_patterns['oxygen_sat'][2])
                else:
                    base_freq = np.random.normal(70, 10)
            else:
                base_freq = np.random.normal(60, 15)
            
            # Generate healthy STFT spectrum
            for freq_bin in range(26):
                for time_win in range(20):
                    freq_factor = freq_bin / 25.0
                    time_factor = time_win / 19.0
                    
                    # Healthy patterns:
                    # - Moderate energy across frequencies
                    # - Good variability in appropriate ranges
                    # - Regular, predictable patterns
                    
                    if freq_factor < 0.3:  # Low frequency - normal baseline
                        energy = np.abs(np.random.normal(base_freq * 0.3, base_freq * 0.1))
                    elif freq_factor < 0.7:  # Mid frequency - healthy variability
                        energy = np.abs(np.random.normal(base_freq * 0.4, base_freq * 0.2))
                    else:  # High frequency - normal HRV
                        energy = np.abs(np.random.normal(base_freq * 0.2, base_freq * 0.1))
                    
                    # Add healthy temporal patterns
                    temporal_var = 1 + normal_patterns['variability'] * np.sin(time_factor * 2 * np.pi) * 0.1
                    energy *= temporal_var
                    
                    # Add minimal healthy noise
                    energy += np.random.normal(0, energy * 0.1)
                    
                    features.append(max(0, energy))
                    
                    if len(features) >= 536:
                        break
                if len(features) >= 536:
                    break
            if len(features) >= 536:
                break
        
        # Ensure exactly 536 features
        while len(features) < 536:
            features.append(np.random.exponential(5))
        
        return features[:536]
    
    def generate_test_dataset(self, n_sepsis=50, n_healthy=50):
        """Generate complete test dataset"""
        print("🏥 GENERATING REALISTIC SEPSIS PATIENT DATA")
        print("=" * 50)
        
        all_patients = []
        
        # Generate sepsis patients
        print(f"Creating {n_sepsis} sepsis patients...")
        sepsis_severities = ['mild'] * 20 + ['moderate'] * 20 + ['severe'] * 10
        
        for i in range(n_sepsis):
            severity = sepsis_severities[i] if i < len(sepsis_severities) else 'moderate'
            features = self._generate_sepsis_patient(i, severity)
            
            patient = {
                'patient_id': f'SEPSIS_{i+1:03d}',
                'sepsis_label': 1,
                'severity': severity,
                'admission_time': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                'age': np.random.randint(45, 85),
                'gender': np.random.choice(['M', 'F']),
                'comorbidities': np.random.choice(['None', 'Diabetes', 'Hypertension', 'Heart Disease', 'Multiple'], 
                                                p=[0.1, 0.3, 0.3, 0.2, 0.1])
            }
            
            # Add features
            for j, feature_name in enumerate(self.feature_names):
                patient[feature_name] = features[j]
            
            all_patients.append(patient)
        
        # Generate healthy patients
        print(f"Creating {n_healthy} healthy patients...")
        for i in range(n_healthy):
            features = self._generate_healthy_patient(i)
            
            patient = {
                'patient_id': f'HEALTHY_{i+1:03d}',
                'sepsis_label': 0,
                'severity': 'none',
                'admission_time': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
                'age': np.random.randint(25, 75),
                'gender': np.random.choice(['M', 'F']),
                'comorbidities': np.random.choice(['None', 'Diabetes', 'Hypertension'], p=[0.7, 0.2, 0.1])
            }
            
            # Add features
            for j, feature_name in enumerate(self.feature_names):
                patient[feature_name] = features[j]
            
            all_patients.append(patient)
        
        # Create DataFrame
        df = pd.DataFrame(all_patients)
        
        # Shuffle the data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Generated {len(df)} total patients")
        print(f"   Sepsis cases: {df['sepsis_label'].sum()}")
        print(f"   Healthy cases: {len(df) - df['sepsis_label'].sum()}")
        
        return df
    
    def create_test_files(self):
        """Create test CSV files"""
        print("📁 CREATING TEST CSV FILES")
        print("=" * 30)
        
        # Create data directory
        os.makedirs('data/test_patients', exist_ok=True)
        
        # Generate full dataset
        full_dataset = self.generate_test_dataset(50, 50)
        
        # Save full dataset
        full_dataset.to_csv('data/test_patients/complete_test_dataset.csv', index=False)
        print("✅ Saved: complete_test_dataset.csv (100 patients)")
        
        # Create sepsis-only dataset
        sepsis_only = full_dataset[full_dataset['sepsis_label'] == 1].copy()
        sepsis_only.to_csv('data/test_patients/sepsis_patients_only.csv', index=False)
        print(f"✅ Saved: sepsis_patients_only.csv ({len(sepsis_only)} sepsis patients)")
        
        # Create healthy-only dataset
        healthy_only = full_dataset[full_dataset['sepsis_label'] == 0].copy()
        healthy_only.to_csv('data/test_patients/healthy_patients_only.csv', index=False)
        print(f"✅ Saved: healthy_patients_only.csv ({len(healthy_only)} healthy patients)")
        
        # Create small mixed sample for quick testing
        small_sample = full_dataset.sample(10, random_state=42)
        small_sample.to_csv('data/test_patients/quick_test_sample.csv', index=False)
        print(f"✅ Saved: quick_test_sample.csv ({len(small_sample)} mixed patients)")
        
        # Create feature-only files (without metadata) for API testing
        feature_cols = [col for col in full_dataset.columns if col.startswith(('heart_rate', 'blood_pressure', 'respiratory', 'oxygen', 'temperature', 'signal_', 'derived_'))]
        
        # Small API test files
        api_sepsis = sepsis_only[feature_cols].head(5)
        api_sepsis.to_csv('data/test_patients/api_test_sepsis.csv', index=False)
        print(f"✅ Saved: api_test_sepsis.csv (5 sepsis patients, features only)")
        
        api_healthy = healthy_only[feature_cols].head(5)
        api_healthy.to_csv('data/test_patients/api_test_healthy.csv', index=False)
        print(f"✅ Saved: api_test_healthy.csv (5 healthy patients, features only)")
        
        # Create summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_patients': len(full_dataset),
            'sepsis_patients': int(full_dataset['sepsis_label'].sum()),
            'healthy_patients': int(len(full_dataset) - full_dataset['sepsis_label'].sum()),
            'features_per_patient': len(feature_cols),
            'feature_types': {
                'vital_signs': 6,
                'stft_features': 520,
                'derived_features': 10
            },
            'files_created': [
                'complete_test_dataset.csv',
                'sepsis_patients_only.csv', 
                'healthy_patients_only.csv',
                'quick_test_sample.csv',
                'api_test_sepsis.csv',
                'api_test_healthy.csv'
            ],
            'usage_instructions': {
                'dashboard_testing': 'Upload any CSV file to the dashboard',
                'api_testing': 'Use api_test_*.csv files for API endpoints',
                'model_validation': 'Use complete_test_dataset.csv for full validation',
                'quick_demo': 'Use quick_test_sample.csv for demonstrations'
            }
        }
        
        with open('data/test_patients/dataset_info.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print("✅ Saved: dataset_info.json (metadata)")
        
        return summary

def main():
    """Main function"""
    print("🚀 SEPSIS PATIENT DATA GENERATOR")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    generator = SepsisDataGenerator()
    summary = generator.create_test_files()
    
    print(f"\n🎉 DATA GENERATION COMPLETE!")
    print("=" * 50)
    print(f"📊 Total patients: {summary['total_patients']}")
    print(f"🚨 Sepsis cases: {summary['sepsis_patients']}")
    print(f"💚 Healthy cases: {summary['healthy_patients']}")
    print(f"📈 Features per patient: {summary['features_per_patient']}")
    print()
    print("📁 Files created in data/test_patients/:")
    for file in summary['files_created']:
        print(f"   ✅ {file}")
    print()
    print("🔧 Usage:")
    print("   - Upload CSV files to dashboard for testing")
    print("   - Use with API endpoints for validation")
    print("   - Test model performance with realistic data")
    print()
    print("✨ Ready for dashboard testing!")

if __name__ == '__main__':
    main()