import pandas as pd
import numpy as np

# Check STFT features structure
train_stft = pd.read_csv('data/stft_features/train_stft_scaled.csv')
test_stft = pd.read_csv('data/stft_features/test_stft_scaled.csv')

print("Train STFT shape:", train_stft.shape)
print("Test STFT shape:", test_stft.shape)
print("Train columns:", list(train_stft.columns)[:10], "...")
print("Test columns:", list(test_stft.columns)[:10], "...")

# Check if there are any patient files with labels
import os
sample_files = [f for f in os.listdir('data/raw/training_setA (1)') if f.endswith('.psv')][:3]
print(f"\nSample raw files: {sample_files}")

# Load a sample file to see structure
if sample_files:
    sample_data = pd.read_csv(f'data/raw/training_setA (1)/{sample_files[0]}', sep='|')
    print(f"\nSample file {sample_files[0]} columns:", list(sample_data.columns))
    print("Sample file shape:", sample_data.shape)
    if 'SepsisLabel' in sample_data.columns:
        print("Sepsis label distribution:", sample_data['SepsisLabel'].value_counts())