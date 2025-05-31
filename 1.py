import numpy as np
import os
from skrebate import ReliefF

# Load dataset
sequence = np.load("dataset/dataset1/sequence_encoded.npy")[:10000]  # shape: (47473, 2000)
labels = np.load("dataset/dataset1/expression_label.npy") [:10000]   # shape: (47473,)

print("Original sequence shape:", sequence.shape)
print("Labels shape:", labels.shape)

# Apply ReliefF to select top 1000 features
relief = ReliefF(n_features_to_select=1000, n_neighbors=20)
relief.fit(sequence, labels)
sequence_reduced = relief.transform(sequence)  # shape: (47473, 1000)

print("Reduced sequence shape:", sequence_reduced.shape)

# Save output
output_dir = "dataset/dataset1/relief_output"
os.makedirs(output_dir, exist_ok=True)

np.save(f"{output_dir}/relief_sequences.npy", sequence_reduced)
np.save(f"{output_dir}/relief_labels.npy", labels)

print(f"Saved top 1000 features using ReliefF to: {output_dir}")
