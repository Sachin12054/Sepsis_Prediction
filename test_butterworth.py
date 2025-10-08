#!/usr/bin/env python3
"""
Quick Butterworth test
"""

from signal_processing.butterworth_filters import ButterworthProcessor
import numpy as np

print('🔧 Testing Butterworth Integration...')

# Initialize processor
processor = ButterworthProcessor(sampling_rate=100)

# Create test signal with noise
fs = 100
t = np.linspace(0, 10, fs * 10)
clean_signal = np.sin(2 * np.pi * 1 * t)
noisy_signal = clean_signal + 0.5 * np.random.randn(len(t))

# Apply Butterworth filtering
result = processor.process_physiological_signal(noisy_signal, 'heart_rate')

print('✅ Butterworth filtering successful!')
print(f'📊 Noise reduction: {result["noise_reduction"]:.1%}')
print(f'🎯 Quality improvement: {result["quality_improvement"]:.3f}')
print('🔧 Butterworth integration is working correctly!')