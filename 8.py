import heapq
from collections import Counter

import numpy as np
import tensorflow as tf
from keras.src.utils import to_categorical
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, BatchNormalization, Dropout


# from sklearn.preprocessing import LabelEncoder
#
# # Example DNA sequence (you can extend this with more labeled sequences)
# sequence = "ATGGCCGCCGCCGCCGCCACTTTCGGCTTCCTCCATCCTCCAATCCGGAAACCTGCAGTCCCACCACTGTACATTCTCCGGCTTCCCACCAAACCCCACTCCAAAACGCACCCTCGTTCTCCCCCCCTCCTCTTCCTCCTCCTGGGCCGCCGCCGGGGAGGCCCCATCGCCGCCTTCCCCAACACCACATCTTCATCCACGAATGCCCCTGCCTCGCCCACCTACGACGTCCGGGAGGCAGAGGCCGCCGTCGCGGATCTCCTCCGCGAGGGCGGCGCCTCCGCGGACGACGCCGCCTCCATCGCCGCGCGCGCGCCGGCCTACGCCGCTATGCTCGCCGATGGCGTCCGCGAGCTGGACGAGCTTGGCCTCTGGGCGTCGTGGAGCTCCGGTGCCAGGGCCCGGCTGGGCCTCAGCGGGGTCGTCGAGATGGAGATGGGGAGGCTCGGTTTTAGGAGGAAGGTGTATCTCATGGGACGGAGCAAGCCTGACCACGGCGTGGTGCCGCTCCTCGAGAGCTTGGGAATGCGTCTCTCCTCGGCCAAACTCATCGCGCCTTACGTCGCGGCTGCGGGCCTTACTGTGCTGATTGATAGGGTTAAGTTTTTGAAGGAAATGTTATTTTCAAGCAGTGATTATGCAATACTAATTGGAAGGAATGCTAAGCGGATGATGACATACTTATCAATACCTGCAGATGATGCACTCCAAAGTACTTTATCTTTTTTTGAAAAAATGGAGGCTAGGTATGGTGGTGTTAGCATGTTGGGACATGGAGATGTGTCATTTCCTTACCTCATTGAATCATTTCCGATGCTTCTTCTCTGCTCAGAAGATAATCATCTCAAGCCGTTAGTTGATTTTCTCGAGCACATTGGAATTCCAAAGCCAAAGATTGCATCAGTTCTGCTGCTATTTCCTCCTATCATTCTTTCTGATGTTGAAAATGATATTAAGCCTAGGATTCGTGAATGGGAGAAGGCTGGCATTGAACAAGACTATGTTAGTAGGATGTTGTTGAAGTATCCATGGATTCTTTCAACGAGTGTGATAGAGAACTACAGTCAAATGCTGTTGTTTTTCAACCAAAAAAGGATTTCCAGTACAGTCCTCGCTATTGCTGTGAAAAGTTGGCCTCATATTCTTGGCTCCTCTTCAAAAAGAATGAATTCAGTTTTGGAGCTGTTTCATGTTCTGGGCATCAGTAAAAAAATGGTGGTTCCAGTCATTACATCAAGTCCACAGTTATTACTGAGAAAACCTGATCAGTTTATGCAGAATGTTTTGTTTTTCAGAGAAATGGGTGTTGATAAGAAAACAACAGGAAAAATTCTGTGTCGTTCGCCTGAAATATTTGCTTCAAACGTGGATAACACCCTCAAGAAGAAAATCGATTTTCTTACCAACTTTGGTGTTTCTAAACATCATCTTCCTCGCATCATTCGGAAGTATCCAGAACTTTTATTGTTGGACATAAATTGTACATTGCTCCCTAGGATGAACTACTTATTGGAGATGGGTTTGTCTAAGAAAGATCTGTGCTCAATGATCTTTAGATTTTCCCCACTTCTAGGTTACAGTATTGAACTTGTTATGAAACCAAAGCTTGAGTTTCTGCTAAGAACCATGAAGAAGCCACTTAAAGCAGTTGTAGAATACCCAAGGTACTTCAGTTATTCACTCGAGGGGAAGATCAAACCGCGGTTCTGGGTATTGCAGAGTAGAAACATAGACTGCACTCTGACAGAGATGTTAGCAAAGAACGATGAACTCTTTGCTGAAGAGTACTTGGGACTTGGAGGATTGCTCGAGAAACCTCTACAATCAAGCATAGGCAGTTAA"  # Truncated for brevity
#
# # Step 1: One-hot encode the DNA sequence
def one_hot_encode(seq):
    mapping = {'A': [1,0,0,0],
               'C': [0,1,0,0],
               'G': [0,0,1,0],
               'T': [0,0,0,1],
               'N': [0,0,0,0]}  # Unknown nucleotide
    return np.array([mapping.get(nuc, [0,0,0,0]) for nuc in seq])
#

# # Step 2: Create sequences (e.g., windows of 100 bases)
def create_dataset(sequence, window_size=100, stride=25):
    X = []
    y = []  # Dummy binary labels; replace with real ones
    for i in range(0, len(sequence) - window_size, stride):
        window = sequence[i:i + window_size]
        if 'N' not in window:  # Skip windows with unknown bases
            X.append(one_hot_encode(window))
            y.append(1 if i % 2 == 0 else 0)   # Simulated labels for demonstration
    return np.array(X), np.array(y)

def encode_sequence(seq):
        mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        return [mapping.get(base.upper(), 4) for base in seq]
# X, y = create_dataset(sequence)
#
# # Step 3: Split the data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Step 4: Build a simple neural network
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(X.shape[1], 4)),
#     tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
#
# # Step 5: Compile and train
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()
# model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
#
# # Evaluate
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"Test Accuracy: {accuracy:.2f}")


import numpy as np
from Bio import SeqIO
from sklearn.model_selection import train_test_split
from collections import Counter

def get_kmers(sequence, k=3):
    # Extract all k-mers as strings
    return [str(sequence[i:i+k]) for i in range(len(sequence) - k + 1)]

def build_kmer_vocab(sequences, k=3):
    vocab = {}
    idx = 1  # Start at 1, reserve 0 for k-mers with 'N'
    for seq in sequences:
        kmers = get_kmers(seq, k)
        for kmer in kmers:
            if 'N' in kmer:
                # skip adding k-mers containing 'N' to vocab
                continue
            if kmer not in vocab:
                vocab[kmer] = idx
                idx += 1
    return vocab

def kmers_to_ints(sequence, vocab, k=3, unknown_token=0):
    kmers = get_kmers(sequence, k)
    int_seq = []
    for kmer in kmers:
        if 'N' in kmer:
            int_seq.append(unknown_token)  # Assign unique number for ambiguous k-mers
        else:
            int_seq.append(vocab.get(kmer, unknown_token))  # fallback to unknown_token if not found
    return int_seq
#
# # Parameters
# k = 3
# unknown_kmer_id = 4
#
# # Read sequences and TPM values, store sequences for vocab build
# all_seqs = []
# all_lab = []
# raw_label=[]
# with open("dataset/dataset1/Original_dataset/zea_mays_train.fa", "r") as fasta_file:
#     for record in SeqIO.parse(fasta_file, "fasta"):
#         tpm = record.description.split('|')
#         tpm.pop(0)  # remove id from description parts
#         tpm_values = list(map(float, tpm))
#         tpm_mean = np.mean(tpm_values)
#         all_seqs.append(record.seq)
#         all_lab.append(tpm_mean)
#         raw_label.append(tpm_values)
#
# print(f"Loaded {len(all_seqs)} sequences")
#
# np.save("dataset/dataset1/Original_dataset/Raw_labels.npy",raw_label)
# # Build vocab from all sequences (excluding N-containing k-mers)
# vocab = build_kmer_vocab(all_seqs, k)
# print(f"Vocabulary size (unique {k}-mers excluding 'N'): {len(vocab)}")
# print("Sample vocabulary 3-mers:", list(vocab.keys())[:10])
#
# # Convert all sequences to integer k-mer encoded sequences
# all_feat = [kmers_to_ints(seq, vocab, k, unknown_token=unknown_kmer_id) for seq in all_seqs]
#
# all_feat = np.array(all_feat)
# all_lab = np.array(all_lab)
#
# print("Shape of integer k-mer features:", all_feat.shape)
# print("Shape of labels:", all_lab.shape)
#
# # TPM classification thresholds and labeling
# # Define binary threshold (e.g., median)
# threshold = np.percentile(all_lab, 50)
#
# def categorize_tpm(tpm):
#     if tpm <= threshold:
#         return 0  # Low
#     else:
#         return 1  # High
#
#
# labels = np.array([categorize_tpm(tpm) for tpm in all_lab])
#
# print("Class distribution:", Counter(labels))
#
# # Save numpy arrays
# np.save("dataset/dataset1/Original_dataset/seq_kmer_int.npy", all_feat)
# np.save("dataset/dataset1/Original_dataset/labels.npy", labels)
#
# # print("Data saved successfully")

# Load a subset and split for train/test
feat = np.load("dataset/dataset1/Original_dataset/seq_kmer_int.npy")[:20000]
labels = np.load("dataset/dataset1/Original_dataset/Raw_labels.npy")[:20000]

labels=np.array(labels)
lab=[np.mean(data) for data in labels]

X_train, X_test, y_train, y_test = train_test_split(feat, lab, test_size=0.2, random_state=42)

unique, counts = np.unique(lab, return_counts=True)
# print("Label counts:", dict(zip(unique, counts)))




# Step 4: Build a simple neural network
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(feat.shape[1], 4)),
#     tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
#     tf.keras.layers.MaxPooling1D(pool_size=2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

class KNN:
    def __init__(self, k):
        # Wrap KNN with scaling to help with numerical stability
        self.model = make_pipeline(
            StandardScaler(),  # Normalize features
            KNeighborsRegressor(n_neighbors=k, weights='distance', n_jobs=-1)  # Use distance-weighted voting
        )

    def fit(self, X_train, y_train):
        X_train = np.array(X_train).reshape(len(X_train), -1)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = np.array(X_test).reshape(len(X_test), -1)
        return self.model.predict(X_test)

    def score(self, X_test, y_test):
        X_test = np.array(X_test).reshape(len(X_test), -1)
        y_test = np.array(y_test)
        return self.model.score(X_test, y_test)


# Example usage:
# Assuming X_train, y_train, X_test, y_test are already loaded

model2 = KNN(k=5)
model2.fit(X_train, y_train)
predictions = model2.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"KNN Regression MAE: {mae:.4f}")
print(f"KNN Regression MSE: {mse:.4f}")
print(f"KNN Regression R² Score: {r2:.4f}")



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
X_test=np.array(X_test)
y_test=np.array(y_test)
X_train=np.array(X_train)
y_train=np.array(y_train)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense, BatchNormalization

print("the length o fvocab is:",feat[:10])
vocab_size = 85  # +1 for unknown token (0)
embedding_dim = 8  # or 16, 32, tune as you like
sequence_length = 5998
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=sequence_length),
    Conv1D(128, kernel_size=20, activation='relu'),
    MaxPooling1D(pool_size=4),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(64, kernel_size=20, activation='relu'),
    MaxPooling1D(pool_size=4),
    BatchNormalization(),
    Dropout(0.3),

    Conv1D(128, kernel_size=20, activation='relu'),
    MaxPooling1D(pool_size=4),
    BatchNormalization(),
    Dropout(0.4),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)



model.fit(X_train, y_train,
          validation_split=0.1,
          epochs=10,
          batch_size=64,
          callbacks=[early_stop])


from sklearn.metrics import classification_report, confusion_matrix

y_pred_probs = model.predict(X_test)


print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#
# model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# model.summary()
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# y_pred=model.predict(X_test)
# # Evaluate
# loss, accuracy = model.evaluate(y_pred, y_test)
# print(f"Test Accuracy: {loss:.2f}")




