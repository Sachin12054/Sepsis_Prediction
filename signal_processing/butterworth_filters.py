#!/usr/bin/env python3
"""
🔧 Butterworth Signal Processing Module
=====================================
Advanced Butterworth filtering for physiological signals in sepsis prediction

This module provides clinical-grade Butterworth filters optimized for:
- Heart Rate (HR) signals
- Blood Pressure (BP) signals  
- Temperature and other vital signs
- Noise reduction for STFT feature extraction
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import butter, filtfilt, sosfiltfilt
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ButterworthProcessor:
    """
    Clinical-grade Butterworth filter processor for sepsis prediction
    
    Features:
    - Multiple filter types (lowpass, highpass, bandpass, bandstop)
    - Optimized for physiological signals
    - Clinical parameter presets
    - STFT-ready output
    """
    
    def __init__(self, sampling_rate=100):
        """
        Initialize Butterworth processor
        
        Args:
            sampling_rate (float): Sampling frequency in Hz (default: 100 Hz)
        """
        self.fs = sampling_rate
        self.nyquist = sampling_rate / 2.0
        
        # Clinical frequency bands for physiological signals
        self.clinical_bands = {
            'heart_rate': {
                'low_freq': 0.5,    # Remove DC and very low frequency artifacts
                'high_freq': 5.0,   # Typical HR range: 30-300 BPM -> 0.5-5 Hz
                'order': 4
            },
            'blood_pressure': {
                'low_freq': 0.1,    # Preserve BP variations
                'high_freq': 10.0,  # Remove high-frequency noise
                'order': 6
            },
            'temperature': {
                'low_freq': 0.01,   # Very slow variations
                'high_freq': 0.5,   # Remove rapid fluctuations
                'order': 8
            },
            'respiratory': {
                'low_freq': 0.1,    # Typical breathing: 6-60 breaths/min
                'high_freq': 1.0,   # -> 0.1-1 Hz
                'order': 4
            }
        }
    
    def design_filter(self, filter_type, cutoff, order=4, btype='low'):
        """
        Design Butterworth filter
        
        Args:
            filter_type (str): 'clinical' or 'custom'
            cutoff (float or tuple): Cutoff frequency(ies)
            order (int): Filter order
            btype (str): Filter type ('low', 'high', 'band', 'bandstop')
        
        Returns:
            sos: Second-order sections filter coefficients
        """
        if filter_type == 'clinical':
            # Use clinical presets
            if isinstance(cutoff, str) and cutoff in self.clinical_bands:
                band = self.clinical_bands[cutoff]
                if cutoff == 'heart_rate' or cutoff == 'respiratory':
                    # Bandpass for HR and respiratory
                    sos = signal.butter(
                        band['order'], 
                        [band['low_freq'], band['high_freq']], 
                        btype='band', 
                        fs=self.fs, 
                        output='sos'
                    )
                else:
                    # Lowpass for BP and temperature
                    sos = signal.butter(
                        band['order'], 
                        band['high_freq'], 
                        btype='low', 
                        fs=self.fs, 
                        output='sos'
                    )
            else:
                raise ValueError(f"Unknown clinical preset: {cutoff}")
        else:
            # Custom filter design
            if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
                # Normalize frequencies
                cutoff_norm = np.array(cutoff) / self.nyquist
            else:
                cutoff_norm = cutoff / self.nyquist
            
            sos = signal.butter(order, cutoff_norm, btype=btype, output='sos')
        
        return sos
    
    def apply_filter(self, data, sos, axis=-1):
        """
        Apply Butterworth filter to data
        
        Args:
            data (array): Input signal
            sos (array): Filter coefficients
            axis (int): Axis along which to filter
        
        Returns:
            filtered_data (array): Filtered signal
        """
        # Use sosfiltfilt for zero-phase filtering
        filtered_data = signal.sosfiltfilt(sos, data, axis=axis)
        return filtered_data
    
    def process_physiological_signal(self, signal_data, signal_type='heart_rate', 
                                   custom_params=None):
        """
        Process physiological signal with optimal Butterworth filtering
        
        Args:
            signal_data (array): Raw physiological signal
            signal_type (str): Type of signal ('heart_rate', 'blood_pressure', etc.)
            custom_params (dict): Custom filter parameters
        
        Returns:
            dict: Processed signal data with filtering info
        """
        if custom_params:
            sos = self.design_filter('custom', **custom_params)
            filter_info = custom_params
        else:
            sos = self.design_filter('clinical', signal_type)
            filter_info = self.clinical_bands[signal_type]
        
        # Apply filtering
        filtered_signal = self.apply_filter(signal_data, sos)
        
        # Calculate signal quality metrics
        original_power = np.var(signal_data)
        filtered_power = np.var(filtered_signal)
        noise_reduction = 1 - (filtered_power / original_power)
        
        return {
            'original_signal': signal_data,
            'filtered_signal': filtered_signal,
            'filter_coefficients': sos,
            'filter_info': filter_info,
            'signal_type': signal_type,
            'noise_reduction': noise_reduction,
            'quality_improvement': filtered_power / original_power
        }
    
    def batch_process_patient_data(self, patient_data, signal_columns=None):
        """
        Process multiple physiological signals for a patient
        
        Args:
            patient_data (DataFrame): Patient physiological data
            signal_columns (dict): Mapping of column names to signal types
        
        Returns:
            DataFrame: Processed data with filtered signals
        """
        if signal_columns is None:
            # Default column mapping
            signal_columns = {
                'HR': 'heart_rate',
                'SBP': 'blood_pressure',
                'DBP': 'blood_pressure', 
                'Temp': 'temperature',
                'Resp': 'respiratory'
            }
        
        processed_data = patient_data.copy()
        filter_summary = {}
        
        for column, signal_type in signal_columns.items():
            if column in patient_data.columns:
                print(f"🔧 Processing {column} as {signal_type}...")
                
                # Process signal
                result = self.process_physiological_signal(
                    patient_data[column].values, 
                    signal_type
                )
                
                # Store filtered signal
                processed_data[f'{column}_filtered'] = result['filtered_signal']
                filter_summary[column] = {
                    'signal_type': signal_type,
                    'noise_reduction': result['noise_reduction'],
                    'filter_info': result['filter_info']
                }
        
        return processed_data, filter_summary
    
    def generate_stft_features(self, filtered_signals, window_size=256, overlap=0.75):
        """
        Generate STFT features from filtered signals for sepsis prediction
        
        Args:
            filtered_signals (dict): Dictionary of filtered signals
            window_size (int): STFT window size
            overlap (float): Window overlap ratio
        
        Returns:
            dict: STFT features ready for ML model
        """
        stft_features = {}
        
        for signal_name, signal_data in filtered_signals.items():
            # Compute STFT
            nperseg = window_size
            noverlap = int(nperseg * overlap)
            
            frequencies, times, stft_matrix = signal.stft(
                signal_data, 
                fs=self.fs,
                nperseg=nperseg,
                noverlap=noverlap,
                window='hann'
            )
            
            # Extract features
            stft_magnitude = np.abs(stft_matrix)
            stft_power = stft_magnitude ** 2
            
            # Clinical frequency bands
            freq_bands = {
                'very_low': (0, 0.04),
                'low': (0.04, 0.15),
                'mid': (0.15, 0.4),
                'high': (0.4, 1.0)
            }
            
            band_features = {}
            for band_name, (f_low, f_high) in freq_bands.items():
                band_mask = (frequencies >= f_low) & (frequencies <= f_high)
                if np.any(band_mask):
                    band_power = np.mean(stft_power[band_mask, :], axis=0)
                    band_features[f'{signal_name}_{band_name}_power'] = np.mean(band_power)
                    band_features[f'{signal_name}_{band_name}_std'] = np.std(band_power)
                    band_features[f'{signal_name}_{band_name}_max'] = np.max(band_power)
            
            stft_features.update(band_features)
        
        return stft_features
    
    def visualize_filtering_results(self, original_signal, filtered_signal, 
                                  signal_type='heart_rate', save_path=None):
        """
        Visualize filtering results for clinical validation
        
        Args:
            original_signal (array): Original signal
            filtered_signal (array): Filtered signal
            signal_type (str): Type of physiological signal
            save_path (str): Path to save plot
        """
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        # Time vector
        time = np.arange(len(original_signal)) / self.fs
        
        # Time domain comparison
        axes[0].plot(time, original_signal, 'b-', alpha=0.7, label='Original', linewidth=1)
        axes[0].plot(time, filtered_signal, 'r-', label='Butterworth Filtered', linewidth=2)
        axes[0].set_xlabel('Time (seconds)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'🔧 Butterworth Filtering Results - {signal_type.title()}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Frequency domain comparison
        freq_orig, psd_orig = signal.welch(original_signal, fs=self.fs, nperseg=256)
        freq_filt, psd_filt = signal.welch(filtered_signal, fs=self.fs, nperseg=256)
        
        axes[1].semilogy(freq_orig, psd_orig, 'b-', alpha=0.7, label='Original PSD')
        axes[1].semilogy(freq_filt, psd_filt, 'r-', label='Filtered PSD', linewidth=2)
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power Spectral Density')
        axes[1].set_title('📊 Power Spectral Density Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Filter response
        if signal_type in self.clinical_bands:
            band = self.clinical_bands[signal_type]
            sos = self.design_filter('clinical', signal_type)
            w, h = signal.sosfreqz(sos, worN=2048, fs=self.fs)
            
            axes[2].plot(w, 20 * np.log10(np.abs(h)), 'g-', linewidth=2)
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Magnitude (dB)')
            axes[2].set_title(f'🎛️ Butterworth Filter Response - {signal_type.title()}')
            axes[2].grid(True, alpha=0.3)
            
            # Mark cutoff frequencies
            if 'low_freq' in band and 'high_freq' in band:
                axes[2].axvline(band['low_freq'], color='orange', linestyle='--', 
                              label=f"Low cutoff: {band['low_freq']} Hz")
                axes[2].axvline(band['high_freq'], color='red', linestyle='--', 
                              label=f"High cutoff: {band['high_freq']} Hz")
                axes[2].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Filtering visualization saved to {save_path}")
        
        plt.show()

def create_enhanced_stft_features_with_butterworth(patient_data_path, output_path):
    """
    Create enhanced STFT features using Butterworth preprocessing
    
    Args:
        patient_data_path (str): Path to patient data
        output_path (str): Path to save enhanced features
    """
    print("🔧 Creating Enhanced STFT Features with Butterworth Filtering")
    print("=" * 60)
    
    # Initialize processor
    processor = ButterworthProcessor(sampling_rate=100)
    
    # Load patient data (assuming CSV format)
    try:
        patient_data = pd.read_csv(patient_data_path)
        print(f"📊 Loaded data: {patient_data.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Process signals
    processed_data, filter_summary = processor.batch_process_patient_data(patient_data)
    
    # Generate enhanced STFT features
    filtered_signals = {}
    for col in processed_data.columns:
        if col.endswith('_filtered'):
            signal_name = col.replace('_filtered', '')
            filtered_signals[signal_name] = processed_data[col].values
    
    if filtered_signals:
        stft_features = processor.generate_stft_features(filtered_signals)
        
        # Save results
        feature_df = pd.DataFrame([stft_features])
        feature_df.to_csv(output_path, index=False)
        
        print(f"✅ Enhanced STFT features saved to {output_path}")
        print(f"📈 Generated {len(stft_features)} Butterworth-enhanced features")
        
        # Print filter summary
        print("\n🔧 Butterworth Filter Summary:")
        for signal, info in filter_summary.items():
            print(f"   {signal}: {info['noise_reduction']:.1%} noise reduction")
    
    return processed_data, stft_features, filter_summary

if __name__ == "__main__":
    # Example usage
    print("🔧 Butterworth Signal Processing for Sepsis Prediction")
    print("=" * 50)
    
    # Create synthetic physiological data for demonstration
    fs = 100  # 100 Hz sampling rate
    duration = 300  # 5 minutes
    t = np.linspace(0, duration, fs * duration)
    
    # Synthetic HR signal with noise
    hr_clean = 70 + 10 * np.sin(2 * np.pi * 0.2 * t)  # Baseline + slow variation
    hr_noise = 5 * np.random.randn(len(t))  # Noise
    hr_artifacts = 15 * np.sin(2 * np.pi * 20 * t)  # High-frequency artifacts
    hr_signal = hr_clean + hr_noise + hr_artifacts
    
    # Initialize processor
    processor = ButterworthProcessor(sampling_rate=fs)
    
    # Process HR signal
    result = processor.process_physiological_signal(hr_signal, 'heart_rate')
    
    print(f"🔧 Noise reduction: {result['noise_reduction']:.1%}")
    print(f"📊 Signal quality improvement: {result['quality_improvement']:.3f}")
    
    # Visualize results
    processor.visualize_filtering_results(
        hr_signal, 
        result['filtered_signal'], 
        'heart_rate'
    )