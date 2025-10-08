import pickle
import pandas as pd
import numpy as np

# Check patient files with pickle allowed
train_patients = np.load('data/processed/train_patients.npy', allow_pickle=True)
test_patients = np.load('data/processed/test_patients.npy', allow_pickle=True)
print(f"Train patients: {len(train_patients)}")
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

# Save the labels for use in validation
np.save('data/processed/y_train_stft.npy', train_labels)
np.save('data/processed/y_test_stft.npy', test_labels)
print("\nSaved labels to y_train_stft.npy and y_test_stft.npy")