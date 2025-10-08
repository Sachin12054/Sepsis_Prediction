#!/usr/bin/env python3
"""
🏥 ICU Data Converter for Sepsis Prediction
==========================================
Converts raw ICU patient data (PSV format) to 532 STFT features required by sepsis prediction model

This module handles:
- Real hospital ICU data in PSV format
- Missing value interpolation
- Signal preprocessing and STFT feature extraction
- Compatibility with existing 532-feature sepsis model
"""

import numpy as np
import pandas as pd
import warnings
from scipy import signal, interpolate, stats
from typing import Dict, List, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

class ICUDataConverter:
    """
    Converts raw ICU patient data to 532 STFT features for sepsis prediction
    """
    
    def __init__(self, sampling_rate=1.0):  # ICU data typically 1 hour intervals
        self.fs = sampling_rate
        self.required_signals = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'Resp', 'pH', 'BaseExcess', 'Glucose', 'PaCO2', 'Hct']
        self.target_features = 532
        
        # Clinical normal ranges for validation
        self.normal_ranges = {
            'HR': (60, 100),
            'O2Sat': (95, 100),
            'Temp': (36.0, 37.5),
            'SBP': (90, 140),
            'MAP': (70, 105),
            'DBP': (60, 90),
            'Resp': (12, 20),
            'pH': (7.35, 7.45),
            'BaseExcess': (-2, 2),
            'Glucose': (70, 140),
            'PaCO2': (35, 45),
            'Hct': (35, 45)
        }
        
        # Load expected feature names
        self.expected_features = self._load_expected_features()
    
    def _load_expected_features(self) -> List[str]:
        """Load the 532 expected feature names"""
        try:
            with open('data/stft_features/stft_feature_columns.txt', 'r') as f:
                features = [line.strip() for line in f.readlines()]
            print(f"📋 Loaded {len(features)} expected feature names")
            return features
        except:
            print("⚠️  Could not load expected features, will generate generic names")
            return [f'feature_{i}' for i in range(532)]
    
    def process_icu_patient_file(self, file_path: str) -> Dict:
        """
        Process a single ICU patient PSV file
        
        Args:
            file_path (str): Path to patient PSV file
            
        Returns:
            dict: Processed patient data with 532 features
        """
        print(f"\n🏥 Processing ICU patient: {os.path.basename(file_path)}")
        
        # Load patient data
        try:
            patient_data = pd.read_csv(file_path, sep='|')
            print(f"   📊 Loaded {len(patient_data)} time points with {len(patient_data.columns)} variables")
        except Exception as e:
            print(f"   ❌ Error loading file: {e}")
            return self._create_default_features()
        
        # Extract patient metadata
        patient_info = self._extract_patient_info(patient_data)
        
        # Process vital signs
        processed_signals = self._process_vital_signs(patient_data)
        
        # Generate STFT features
        stft_features = self._generate_stft_features(processed_signals)
        
        # Ensure exactly 532 features
        final_features = self._ensure_532_features(stft_features)
        
        print(f"   ✅ Generated {len(final_features)} features for sepsis prediction")
        
        return {
            'patient_info': patient_info,
            'features': final_features,
            'feature_names': list(final_features.keys())
        }
    
    def _extract_patient_info(self, data: pd.DataFrame) -> Dict:
        """Extract patient metadata"""
        info = {}
        
        if 'Age' in data.columns:
            info['age'] = data['Age'].iloc[0] if not data['Age'].isna().all() else 'Unknown'
        if 'Gender' in data.columns:
            info['gender'] = data['Gender'].iloc[0] if not data['Gender'].isna().all() else 'Unknown'
        if 'ICULOS' in data.columns:
            info['icu_length'] = data['ICULOS'].max() if not data['ICULOS'].isna().all() else 0
        if 'SepsisLabel' in data.columns:
            sepsis_episodes = data['SepsisLabel'].sum()
            info['sepsis_detected'] = bool(sepsis_episodes > 0)
            info['sepsis_episodes'] = int(sepsis_episodes)
        
        return info
    
    def _process_vital_signs(self, data: pd.DataFrame) -> Dict:
        """Process and clean vital signs data"""
        processed = {}
        
        for signal_name in self.required_signals:
            if signal_name in data.columns:
                signal_data = data[signal_name].values
                
                # Clean and interpolate signal
                cleaned_signal = self._clean_signal(signal_data, signal_name)
                processed[signal_name] = cleaned_signal
                
                # Calculate data quality metrics
                original_count = np.sum(~np.isnan(signal_data))
                total_count = len(signal_data)
                quality = original_count / total_count * 100
                
                print(f"   📈 {signal_name:12}: {original_count:2d}/{total_count:2d} readings ({quality:4.1f}% complete)")
            else:
                # Create synthetic signal if missing
                processed[signal_name] = self._create_synthetic_signal(signal_name, len(data))
                print(f"   🔧 {signal_name:12}: Generated synthetic signal (missing from data)")
        
        # Add computed signals
        processed['Shock_Index'] = self._compute_shock_index(processed.get('HR'), processed.get('SBP'))
        processed['DBP'] = self._estimate_dbp(processed.get('SBP'), processed.get('MAP'))
        processed['Creatinine'] = self._create_synthetic_signal('Creatinine', len(data))
        
        return processed
    
    def _clean_signal(self, signal_data: np.ndarray, signal_name: str) -> np.ndarray:
        """Clean and interpolate a physiological signal"""
        
        # Ensure signal_data is float type to handle NaN values
        signal_data = signal_data.astype(float)
        
        # Remove obvious outliers based on clinical ranges
        if signal_name in self.normal_ranges:
            low, high = self.normal_ranges[signal_name]
            # Allow wider range for pathological cases (±50% of normal range)
            range_width = high - low
            extended_low = low - range_width * 0.5
            extended_high = high + range_width * 0.5
            
            outlier_mask = (signal_data < extended_low) | (signal_data > extended_high)
            signal_data = signal_data.copy()
            signal_data[outlier_mask] = np.nan
        
        # Interpolate missing values
        valid_indices = ~np.isnan(signal_data)
        
        if np.sum(valid_indices) >= 2:  # Need at least 2 points for interpolation
            x_valid = np.where(valid_indices)[0]
            y_valid = signal_data[valid_indices]
            
            # Interpolate missing values
            x_all = np.arange(len(signal_data))
            interpolated = interpolate.interp1d(
                x_valid, y_valid, 
                kind='linear', 
                bounds_error=False, 
                fill_value='extrapolate'
            )(x_all)
            
            return interpolated
        else:
            # Not enough valid data, create synthetic signal
            return self._create_synthetic_signal(signal_name, len(signal_data))
    
    def _create_synthetic_signal(self, signal_name: str, length: int) -> np.ndarray:
        """Create a synthetic physiological signal when data is missing"""
        
        # Clinical baseline values
        baselines = {
            'HR': 75,
            'O2Sat': 98,
            'Temp': 36.8,
            'SBP': 120,
            'MAP': 80,
            'DBP': 80,
            'Resp': 16,
            'pH': 7.4,
            'BaseExcess': 0,
            'Glucose': 100,
            'PaCO2': 40,
            'Hct': 40,
            'Creatinine': 1.0
        }
        
        if signal_name not in baselines:
            baseline = 1.0
        else:
            baseline = baselines[signal_name]
        
        # Create realistic physiological variation
        t = np.linspace(0, length-1, length)
        
        # Add multiple frequency components for realistic variation
        variation = (
            0.1 * np.sin(2 * np.pi * t / (length * 0.8)) +  # Slow drift
            0.05 * np.sin(2 * np.pi * t / (length * 0.2)) +  # Medium variation
            0.02 * np.random.normal(0, 1, length)            # Noise
        )
        
        synthetic_signal = baseline * (1 + variation)
        return synthetic_signal
    
    def _compute_shock_index(self, hr: np.ndarray, sbp: np.ndarray) -> np.ndarray:
        """Compute shock index (HR/SBP)"""
        if hr is not None and sbp is not None:
            # Avoid division by zero
            sbp_safe = np.where(sbp > 0, sbp, 120)  # Use default if SBP is 0
            return hr / sbp_safe
        else:
            return np.full(len(hr) if hr is not None else 50, 0.6)  # Normal shock index
    
    def _estimate_dbp(self, sbp: np.ndarray, map_pressure: np.ndarray) -> np.ndarray:
        """Estimate DBP from SBP and MAP using: MAP = DBP + (SBP-DBP)/3"""
        if sbp is not None and map_pressure is not None:
            # DBP = MAP - (SBP - MAP)/2 (rearranged formula)
            dbp = 2 * map_pressure - sbp / 3
            return np.clip(dbp, 40, 100)  # Reasonable DBP range
        else:
            return self._create_synthetic_signal('DBP', len(sbp) if sbp is not None else 50)
    
    def _generate_stft_features(self, signals: Dict) -> Dict:
        """Generate STFT features from processed signals"""
        all_features = {}
        
        # Process each signal with STFT
        for signal_name, signal_data in signals.items():
            if signal_data is not None and len(signal_data) > 0:
                stft_features = self._compute_signal_stft_features(signal_data, signal_name)
                all_features.update(stft_features)
        
        return all_features
    
    def _safe_feature_value(self, value: float) -> float:
        """Ensure feature value is valid (not NaN or inf)"""
        if np.isnan(value) or np.isinf(value):
            return 0.0
        return float(value)
    
    def _compute_signal_stft_features(self, signal_data: np.ndarray, signal_name: str) -> Dict:
        """Compute STFT features for a single signal"""
        features = {}
        
        # STFT parameters adapted for ICU data (typically hourly samples)
        min_length = 8  # Minimum 8 time points needed
        if len(signal_data) < min_length:
            # Pad signal if too short
            padded_signal = np.pad(signal_data, (0, min_length - len(signal_data)), mode='edge')
        else:
            padded_signal = signal_data
        
        nperseg = min(8, len(padded_signal) // 2)
        noverlap = nperseg // 2
        
        try:
            # Compute STFT
            frequencies, times, stft_matrix = signal.stft(
                padded_signal,
                fs=self.fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann'
            )
            
            stft_magnitude = np.abs(stft_matrix)
            stft_power = stft_magnitude ** 2
            
            # Generate the expected 38 features per signal (532/14 signals ≈ 38)
            feature_templates = [
                'total_power', 'mean_power', 'max_power', 'spectral_centroid', 'spectral_spread',
                'spectral_entropy', 'spectral_rolloff', 'spectral_flatness',
                'ultra_slow_mean_power', 'ultra_slow_max_power', 'ultra_slow_std_power',
                'ultra_slow_power_ratio', 'ultra_slow_spectral_centroid', 'ultra_slow_temporal_variability',
                'slow_mean_power', 'slow_max_power', 'slow_std_power', 'slow_power_ratio',
                'slow_spectral_centroid', 'slow_temporal_variability',
                'moderate_mean_power', 'moderate_max_power', 'moderate_std_power',
                'moderate_power_ratio', 'moderate_spectral_centroid', 'moderate_temporal_variability',
                'fast_mean_power', 'fast_max_power', 'fast_std_power', 'fast_power_ratio',
                'fast_temporal_variability', 'temporal_power_mean', 'temporal_power_std',
                'temporal_power_max', 'temporal_power_min', 'temporal_trend', 'temporal_concentration'
            ]
            
            # Basic power features
            total_power = np.sum(stft_power)
            features[f'{signal_name}_stft_total_power'] = self._safe_feature_value(total_power)
            
            mean_power = np.mean(stft_power)
            features[f'{signal_name}_stft_mean_power'] = self._safe_feature_value(mean_power)
            
            max_power = np.max(stft_power)
            features[f'{signal_name}_stft_max_power'] = self._safe_feature_value(max_power)
            
            # Spectral features
            if len(frequencies) > 1:
                # Spectral centroid
                power_sum = np.sum(stft_power, axis=1)
                if np.sum(power_sum) > 0:
                    spectral_centroid = np.sum(frequencies * power_sum) / np.sum(power_sum)
                else:
                    spectral_centroid = frequencies[len(frequencies)//2]
                features[f'{signal_name}_stft_spectral_centroid'] = self._safe_feature_value(spectral_centroid)
                
                # Spectral spread
                spread = np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * power_sum) / np.sum(power_sum))
                features[f'{signal_name}_stft_spectral_spread'] = self._safe_feature_value(spread)
                
                # Spectral entropy
                normalized_power = power_sum / (np.sum(power_sum) + 1e-10)
                spectral_entropy = -np.sum(normalized_power * np.log2(normalized_power + 1e-10))
                features[f'{signal_name}_stft_spectral_entropy'] = self._safe_feature_value(spectral_entropy)
                
                # Spectral flatness
                geometric_mean = stats.gmean(power_sum + 1e-10)
                arithmetic_mean = np.mean(power_sum)
                spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
                features[f'{signal_name}_stft_spectral_flatness'] = self._safe_feature_value(spectral_flatness)
                
                # Spectral rolloff (85% of power)
                cumulative_power = np.cumsum(power_sum)
                total_power = cumulative_power[-1]
                rolloff_threshold = 0.85 * total_power
                rolloff_index = np.where(cumulative_power >= rolloff_threshold)[0]
                if len(rolloff_index) > 0:
                    spectral_rolloff = frequencies[rolloff_index[0]]
                else:
                    spectral_rolloff = frequencies[-1]
                features[f'{signal_name}_stft_spectral_rolloff'] = self._safe_feature_value(spectral_rolloff)
            
            # Frequency band analysis (adapted for ICU data)
            freq_bands = {
                'ultra_slow': (0, 0.1),
                'slow': (0.1, 0.2),
                'moderate': (0.2, 0.3),
                'fast': (0.3, 0.5)
            }
            
            for band_name, (f_low, f_high) in freq_bands.items():
                band_mask = (frequencies >= f_low) & (frequencies <= f_high)
                if np.any(band_mask):
                    band_power = stft_power[band_mask, :]
                    
                    if band_power.size > 0:
                        features[f'{signal_name}_stft_{band_name}_mean_power'] = self._safe_feature_value(np.mean(band_power))
                        features[f'{signal_name}_stft_{band_name}_max_power'] = self._safe_feature_value(np.max(band_power))
                        features[f'{signal_name}_stft_{band_name}_std_power'] = self._safe_feature_value(np.std(band_power))
                        
                        # Power ratio
                        band_total = np.sum(band_power)
                        overall_total = np.sum(stft_power)
                        power_ratio = band_total / (overall_total + 1e-10)
                        features[f'{signal_name}_stft_{band_name}_power_ratio'] = self._safe_feature_value(power_ratio)
                        
                        # Band spectral centroid
                        band_frequencies = frequencies[band_mask]
                        band_power_sum = np.sum(band_power, axis=1)
                        if np.sum(band_power_sum) > 0 and len(band_frequencies) > 0:
                            band_centroid = np.sum(band_frequencies * band_power_sum) / np.sum(band_power_sum)
                            features[f'{signal_name}_stft_{band_name}_spectral_centroid'] = self._safe_feature_value(band_centroid)
                        
                        # Temporal variability in this band
                        temporal_var = np.std(np.mean(band_power, axis=0))
                        features[f'{signal_name}_stft_{band_name}_temporal_variability'] = self._safe_feature_value(temporal_var)
            
            # Additional spectral centroid for fast band (if exists)
            if f'{signal_name}_stft_fast_spectral_centroid' not in features:
                features[f'{signal_name}_stft_fast_spectral_centroid'] = features.get(f'{signal_name}_stft_spectral_centroid', 0)
            
            # Temporal features
            temporal_powers = np.mean(stft_power, axis=0)  # Power over time
            features[f'{signal_name}_stft_temporal_power_mean'] = self._safe_feature_value(np.mean(temporal_powers))
            features[f'{signal_name}_stft_temporal_power_std'] = self._safe_feature_value(np.std(temporal_powers))
            features[f'{signal_name}_stft_temporal_power_max'] = self._safe_feature_value(np.max(temporal_powers))
            features[f'{signal_name}_stft_temporal_power_min'] = self._safe_feature_value(np.min(temporal_powers))
            
            # Temporal trend (linear regression slope)
            if len(temporal_powers) > 1:
                time_indices = np.arange(len(temporal_powers))
                slope, _ = np.polyfit(time_indices, temporal_powers, 1)
                features[f'{signal_name}_stft_temporal_trend'] = self._safe_feature_value(slope)
            else:
                features[f'{signal_name}_stft_temporal_trend'] = 0.0
            
            # Temporal concentration (how concentrated power is in time)
            normalized_temporal = temporal_powers / (np.sum(temporal_powers) + 1e-10)
            temporal_entropy = -np.sum(normalized_temporal * np.log2(normalized_temporal + 1e-10))
            features[f'{signal_name}_stft_temporal_concentration'] = self._safe_feature_value(temporal_entropy)
            
        except Exception as e:
            print(f"     ⚠️  STFT computation failed for {signal_name}: {e}")
            # Create default features for this signal
            for template in ['total_power', 'mean_power', 'max_power', 'spectral_centroid']:
                features[f'{signal_name}_stft_{template}'] = 0.0
        
        return features
    
    def _ensure_532_features(self, features: Dict) -> Dict:
        """Ensure exactly 532 features to match the sepsis model"""
        
        # If we have the expected feature list, use it
        if len(self.expected_features) == 532:
            result = {}
            for feature_name in self.expected_features:
                if feature_name in features:
                    value = features[feature_name]
                    # Ensure no NaN values
                    if np.isnan(value) or np.isinf(value):
                        result[feature_name] = self._generate_default_feature_value(feature_name)
                    else:
                        result[feature_name] = float(value)
                else:
                    # Generate a reasonable default value
                    result[feature_name] = self._generate_default_feature_value(feature_name)
            return result
        
        # Otherwise, pad or truncate to 532 features
        feature_list = list(features.items())
        
        if len(feature_list) >= 532:
            # Truncate to 532 and ensure no NaN
            result = {}
            for i, (name, value) in enumerate(feature_list[:532]):
                if np.isnan(value) or np.isinf(value):
                    result[name] = 0.0
                else:
                    result[name] = float(value)
            return result
        else:
            # Pad to 532 and ensure no NaN
            result = {}
            for name, value in feature_list:
                if np.isnan(value) or np.isinf(value):
                    result[name] = 0.0
                else:
                    result[name] = float(value)
            
            for i in range(len(feature_list), 532):
                result[f'generated_feature_{i}'] = 0.0
            return result
    
    def _generate_default_feature_value(self, feature_name: str) -> float:
        """Generate a reasonable default value for a missing feature"""
        if 'power' in feature_name.lower():
            return 0.1  # Small but non-zero power
        elif 'centroid' in feature_name.lower():
            return 0.5  # Mid-range frequency
        elif 'ratio' in feature_name.lower():
            return 0.1  # Small ratio
        elif 'entropy' in feature_name.lower():
            return 1.0  # Moderate entropy
        elif 'trend' in feature_name.lower():
            return 0.0  # No trend
        else:
            return 0.0  # Default to zero
    
    def _create_default_features(self) -> Dict:
        """Create default feature set when file processing fails"""
        print("   ⚠️  Creating default features due to processing error")
        
        default_features = {}
        for i, feature_name in enumerate(self.expected_features):
            default_features[feature_name] = self._generate_default_feature_value(feature_name)
        
        return {
            'patient_info': {'error': 'Failed to process file'},
            'features': default_features,
            'feature_names': self.expected_features
        }

# Test function
def test_converter():
    """Test the ICU data converter with sample patient"""
    converter = ICUDataConverter()
    
    # Test with patient p000001
    result = converter.process_icu_patient_file('data/raw/training_setA (1)/p000001.psv')
    
    print(f"\n📊 Conversion Results:")
    print(f"   Patient Info: {result['patient_info']}")
    print(f"   Features Generated: {len(result['features'])}")
    print(f"   Feature Names: {len(result['feature_names'])}")
    
    # Show some sample features
    print(f"\n🔍 Sample Features:")
    for i, (name, value) in enumerate(list(result['features'].items())[:10]):
        print(f"   {name}: {value:.6f}")
    
    return result

if __name__ == "__main__":
    test_converter()