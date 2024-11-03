from read_data import read_data
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from normalize_data import normalize_data


# Perform PCA and find how many components are needed to explain 95% of the variance
def pca_experiment(data):
    # Normalize the data
    data = normalize_data(data)

    # PCA analysis
    pca: PCA = PCA()
    pca.fit(data)
    cumsum: int = 0
    for i, explained_variance_ratio in enumerate(pca.explained_variance_ratio_):
        cumsum += explained_variance_ratio
        if cumsum >= 0.95:  # σταματάει όταν το cumsum γίνει >= 0.95
            print(f'Number of components required to explain 95% of the variance: {i + 1}')
            break
    # 448 principal components !!!


def pca_decomposition(data, n_components):
    # data is a 2D NumPy array of shape (50000, 3072) and with pca we reduce the dimensionality to 50000 x n_components
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)
    return data_pca


# Plot the data
def plot_2D(data, labels):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA analysis of CIFAR-10 training data')
    plt.show()
# The plot shows that the data points are not clearly separable in the 2D space of the first two principal components.
# This is expected since the CIFAR-10 dataset contains 10 classes and the data points are not linearly separable in the original 3072-dimensional space.
# The PCA analysis provides a visualization of the data in a lower-dimensional space, but it does not guarantee a clear separation of the classes.
# More advanced techniques such as deep learning models may be required for accurate classification of the CIFAR-10 images.

def main():
   # Read CIFAR-10 training data
    data, labels = read_data()
    print(data.shape)  # (50000, 3072)
    print(labels.shape)  # (50000,)

    # Perform PCA analysis
    pca_experiment(data)

    # data_pca = pca_decomposition(data, 448)
    # print(data_pca.shape)  # (50000, 448)





if __name__ == '__main__':
    main()