import heapq
from collections import Counter
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

feat=np.load("dataset/dataset1/seq_kmer_int.npy")

lab=np.load("dataset/dataset1/Raw_labels.npy")
lab=np.array(lab)
mean_lab=[data[0] for data in lab]
# mean_lab=[np.mean(data) for data in lab]

print(len(mean_lab))

def categorize_tpm(tpm):
    if tpm <=0.5:
        return 0  # Low
    elif tpm> 7:
        return 2
    else:
        return 1  # High


l = np.array([categorize_tpm(tpm) for tpm in mean_lab])

print(Counter(l))

class_0_index=np.where(l==0)[0]
class_1_index=np.where(l==1)[0]


np.random.seed(42)
selected_class_0 = np.random.choice(class_0_index, 1000, replace=False)
selected_class_1 = np.random.choice(class_1_index, 1000, replace=False)


# Combine and shuffle selected indices
selected_indices = np.concatenate([selected_class_0, selected_class_1])
np.random.shuffle(selected_indices)

# Subset the features and labels
balanced_feat = feat[selected_indices]
balanced_labels = l[selected_indices]


# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    balanced_feat, balanced_labels, test_size=0.2, random_state=42
)

print(Counter(balanced_labels))
print(balanced_feat.shape)


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train).flatten()

    def _euclidean_distance(self, a, b):
        return np.linalg.norm(a - b)

    def _get_neighbors(self, test_sample):
        distances = [
            (self.y_train[i], self._euclidean_distance(self.X_train[i], test_sample))
            for i in range(len(self.X_train))
        ]
        nearest = heapq.nsmallest(self.k, distances, key=lambda x: x[1])
        return [label for label, _ in nearest]

    def _majority_vote(self, neighbors):
        vote_count = Counter(neighbors)
        return max(vote_count.items(), key=lambda x: (x[1], -x[0]))[0]

    def predict(self, X_test):
        predictions = []
        for test_sample in X_test:
            neighbors = self._get_neighbors(test_sample)
            label = self._majority_vote(neighbors)
            predictions.append(label)
        return np.array(predictions)

# model=KNN(3)
# model.fit(X_train,y_train)
# prediction=model.predict(X_test)
# accuracy=accuracy_score(y_test,prediction)
# report=classification_report(y_train,prediction)
# print("The accuracy of the model is :",accuracy)
# print("the classifcation report is:",report)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(balanced_feat.shape[1],)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(filters=32,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Step 5: Compile and train
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test,y_test)
print(f"Test Accuracy: {accuracy:.2f}")

