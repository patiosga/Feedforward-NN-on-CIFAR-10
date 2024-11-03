import numpy as np
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from read_data import read_data, read_test_data

# Load CIFAR-10 dataset
(x_train, y_train) = read_data()
(x_test, y_test) = read_test_data()

# take only 1000 samples
x_train = x_train[:900]
y_train = y_train[:900]

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
x_train_tsne = tsne.fit_transform(x_train)

# Plot the results
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x_train_tsne[:, 0], x_train_tsne[:, 1], x_train_tsne[:, 2], c=y_train, cmap='tab10')
plt.colorbar(scatter)
plt.title('t-SNE visualization of CIFAR-10 training data')
plt.show()
