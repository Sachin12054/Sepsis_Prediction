import pickle
import pandas as pd
import numpy as np

# Check STFT metadata
with open('data/stft_features/stft_metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("STFT metadata type:", type(metadata))
if isinstance(metadata, dict):
    print("STFT metadata keys:", list(metadata.keys()))
    for key, value in metadata.items():
        print(f"{key}: {type(value)}", end="")
        if hasattr(value, 'shape'):
            print(f" shape: {value.shape}")
        elif hasattr(value, '__len__'):
            print(f" length: {len(value)}")
        else:
            print("")

# Check patient files
train_patients = np.load('data/processed/train_patients.npy')
test_patients = np.load('data/processed/test_patients.npy')
print(f"\nTrain patients: {len(train_patients)}")
print(f"Test patients: {len(test_patients)}")
print(f"Example train patients: {train_patients[:5]}")
print(f"Example test patients: {test_patients[:3]}")

# Load actual labels from patient files
def load_patient_labels(patient_ids, split_name):
    labels = []
    for patient_id in patient_ids:
        try:
            data = pd.read_csv(f'data/raw/training_setA (1)/{patient_id}.psv', sep='|')
            # Get the maximum sepsis label for this patient (1 if sepsis occurs, 0 otherwise)
            label = data['SepsisLabel'].max()
            labels.append(label)
        except Exception as e:
            print(f"Error loading {patient_id}: {e}")
            labels.append(0)  # Default to no sepsis
    
    labels = np.array(labels)
    print(f"{split_name} labels - Sepsis: {labels.sum()}, No Sepsis: {len(labels) - labels.sum()}")
    return labels

train_labels = load_patient_labels(train_patients, "Train")
test_labels = load_patient_labels(test_patients, "Test")