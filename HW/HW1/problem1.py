import numpy as np
import matplotlib.pyplot as plt


dims = [2, 4, 6, 8, 10]
n_vals = [100, 200, 500, 1000, 2000, 5000]
n_test = 100

# Dictionary to store results: key=d, value=list of average nearest neighbor distances for each n
results = {}

# Loop over dimensions
for d in dims:
    avg_dists_for_d = []
    # Loop over different numbers of training samples
    for n in n_vals:
        # Generate training data: each row is a point in R^d on the unit ball approximately
        X_train = np.random.randn(n, d) / np.sqrt(d)
        
        # Compute the nearest neighbor distances for n_test test examples
        test_dists = []
        for _ in range(n_test):
            # Generate a test point; we also normalize by sqrt(d) for consistency
            x = np.random.randn(d) / np.sqrt(d)
            # Compute Euclidean distances from x to each training point
            distances = np.linalg.norm(X_train - x, axis=1)
            # Record the minimum distance
            test_dists.append(np.min(distances))
        
        # Average the distances over the test examples
        avg_dists_for_d.append(np.mean(test_dists))
    
    # Store the result for dimension d
    results[d] = avg_dists_for_d

# Now plot the results using a semilogx plot (log scale on the x-axis)
plt.figure(figsize=(10, 6))
for d in dims:
    plt.semilogx(n_vals, results[d], marker='o', label=f'd = {d}')

plt.xlabel('Number of Training Samples (n)')
plt.ylabel('Expected Distance to the Nearest Neighbor, E[d(x)]')
plt.title('Distance to the Nearest Neighbor vs. Training Set Size with Various Dimensions')
plt.legend()
plt.grid(True)
plt.show()