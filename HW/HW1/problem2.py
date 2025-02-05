import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import time

# -------------------------------
# Part 1: Training Time Measurements
# -------------------------------
print("=== Training Time Measurements ===")
# List of training set sizes for measuring training time.
n_training = [1000, 10000, 100000]
# kNN algorithms to compare.
algorithms = ['brute', 'ball_tree', 'kd_tree']
# Dictionary to store training times.
training_times = {algo: [] for algo in algorithms}

for n in n_training:
    # Generate synthetic 2D training data.
    X_train = np.random.randn(n, 2)
    y_train = np.sign(np.random.randn(n))
    print(f"\nTraining set size n = {n}")
    
    for algo in algorithms:
        # Initialize the kNN classifier with k=5 neighbors.
        clf = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
        start_time = time.time()
        clf.fit(X_train, y_train)
        end_time = time.time()
        elapsed = end_time - start_time
        training_times[algo].append(elapsed)
        print(f"  Algorithm: {algo:8s} -> fit() time: {elapsed:.6f} seconds")

# Optionally, you can also print a summary table for training times.
print("\nSummary of Training Times:")
print("  n       |  brute    |  ball_tree |  kd_tree")
print("------------------------------------------------")
for i, n in enumerate(n_training):
    bt = training_times['brute'][i]
    btree = training_times['ball_tree'][i]
    kdtree = training_times['kd_tree'][i]
    print(f" {n:7d} | {bt:9.6f} | {btree:9.6f} | {kdtree:9.6f}")

# -------------------------------
# Part 2: Inference Time Measurements
# -------------------------------
print("\n=== Inference Time Measurements ===")
# List of training set sizes for measuring inference time.
n_inference = [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000]
# Dictionary to store inference times.
inference_times = {algo: [] for algo in algorithms}
# Generate a fixed test set of 5000 examples.
n_test = 5000
X_test = np.random.randn(n_test, 2)

for n in n_inference:
    # Generate synthetic 2D training data.
    X_train = np.random.randn(n, 2)
    y_train = np.sign(np.random.randn(n))
    print(f"\nTraining set size n = {n}")
    
    for algo in algorithms:
        clf = KNeighborsClassifier(n_neighbors=5, algorithm=algo)
        clf.fit(X_train, y_train)
        start_time = time.time()
        clf.predict(X_test)
        end_time = time.time()
        elapsed = end_time - start_time
        inference_times[algo].append(elapsed)
        print(f"  Algorithm: {algo:8s} -> predict() time: {elapsed:.6f} seconds")

# Plotting the inference times.
plt.figure(figsize=(10, 6))
for algo in algorithms:
    plt.plot(n_inference, inference_times[algo], marker='o', label=algo)
plt.xscale('log')
plt.xlabel('Number of Training Samples, n')
plt.ylabel('Inference Time [s] for 5000 Test Examples')
plt.title('Inference Time vs. Training Set Size for Different kNN Algorithms (d=2, k=5)')
plt.legend()
plt.grid(True)
plt.show()
