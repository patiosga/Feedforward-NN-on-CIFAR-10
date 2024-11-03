import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import variables as var
from pca_analysis import pca_decomposition
from read_data import read_data, read_test_data
from normalize_data import normalize_data
from accuracy_metrics import cls_report


# Perform k-NN classification on the CIFAR-10 dataset before and after PCA
def knn_experiment(data, labels, test_data, test_labels, neighbors, PCA=False):
    if PCA == True:
        data = pca_decomposition(data, var.pca_components)
        test_data = pca_decomposition(test_data, var.pca_components)
                                      
    # Create and fit k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(data, labels)
    # Predict labels for test data
    predicted_labels = knn.predict(test_data)
    # Calculate accuracy
    class_report = cls_report(test_labels, predicted_labels)

    return class_report


def run_knn(data, labels, test_data, test_labels, neighbors, PCA=False):
    # Perform k-NN classification on the CIFAR-10 dataset before and after PCA
    # Number of neighbors = 1
    print(f'Number of neighbors = {neighbors}')
    print('----------------------')
    class_report_before_pca = knn_experiment(data, labels, test_data, test_labels, neighbors)
    class_report_after_pca = knn_experiment(data, labels, test_data, test_labels, neighbors, PCA=True)

    print('Before PCA:')
    print(class_report_before_pca)
    print('')
    print('After PCA:')
    print(class_report_after_pca)
    print('')
    print('')


def main():
    training_data, training_labels = read_data()
    test_data, test_labels = read_test_data()

    # Normalize the data
    training_data = normalize_data(training_data)
    test_data = normalize_data(test_data)

    # Run k-NN classification
    # run_knn(training_data, training_labels, test_data, test_labels, 1)
    run_knn(training_data, training_labels, test_data, test_labels, 3)


if __name__ == '__main__':
    main()

