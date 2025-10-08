import pickle
import pandas as pd

# Check cached data
with open('data/processed/cached_data_mini.pkl', 'rb') as f:
    data = pickle.load(f)

print("Type:", type(data))
if isinstance(data, dict):
    print("Keys:", list(data.keys()))
else:
    print("Not a dictionary")
    if hasattr(data, 'columns'):
        print("Columns:", list(data.columns))
    if hasattr(data, 'shape'):
        print("Shape:", data.shape)