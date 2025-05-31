import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
x = np.load("dataset/dataset1/sequence_encoded.npy")
y = np.load("dataset/dataset1/expression_label.npy")

# Get lengths of each sequence
sequence_lengths = [len(seq) for seq in x]

# Check if all sequences have the same length
all_same_length = len(set(sequence_lengths)) == 1

print("Are all sequences the same length?", all_same_length)
print("Sequence length list :", sequence_lengths)
if not all_same_length:
    print("Unique sequence lengths:", set(sequence_lengths))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
x = np.load("dataset/dataset1/sequence_encoded.npy")
y = np.load("dataset/dataset1/expression_label.npy")

print("x shape:", x.shape)  # (47473, 2000)
print("y shape:", y.shape)  # (47473,)

# Reshape y and concatenate
y_reshaped = y.reshape(-1, 1)
combined = np.concatenate([x, y_reshaped], axis=1)

unique,count=np.unique(y,return_counts=True)
print("the unique:",unique)
print("count:",count)
print("Combined shape:", combined.shape)

# Get one sample for each class
samples = []
for label in [0, 1, 2]:
    index = np.where(y == label)[0][0]  # first index of class `label`
    samples.append((x[index], label))

# Plot
plt.figure(figsize=(12, 6))
for i, (sequence, label) in enumerate(samples):
    plt.plot(range(len(sequence)), sequence, label=f"Class {label}")  # Explicit x-axis
plt.xlim(0, 2000)  # Force full range
plt.title("DNA Encoded Sequence Sample from Each Class")
plt.xlabel("Sequence Position")
plt.ylabel("Encoded Value")
plt.legend()
plt.grid(True)
plt.show()

