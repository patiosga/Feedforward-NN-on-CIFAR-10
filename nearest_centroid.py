from sklearn.neighbors import NearestCentroid
import numpy as np
import variables as var
from pca_analysis import pca_decomposition
from read_data import read_data, read_test_data
from normalize_data import normalize_data
from accuracy_metrics import cls_report


def nearest_centroid_experiment(data, labels, test_data, test_labels, PCA=False):
    if PCA == True:
        data = pca_decomposition(data, var.pca_components)
        test_data = pca_decomposition(test_data, var.pca_components)
                                      
    # Create and fit k-NN classifier
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(data, labels)
    # Predict labels for test data
    predicted_labels = nearest_centroid.predict(test_data)
    # Calculate accuracy
    class_report = cls_report(test_labels, predicted_labels)

    return class_report


def main():
    training_data, training_labels = read_data()
    test_data, test_labels = read_test_data()

    # Normalize the data
    training_data = normalize_data(training_data)
    test_data = normalize_data(test_data)

    # Perform nearest centroid classification on the CIFAR-10 dataset before and after PCA
    class_report_before_pca = nearest_centroid_experiment(training_data, training_labels, test_data, test_labels)
    class_report_after_pca = nearest_centroid_experiment(training_data, training_labels, test_data, test_labels, PCA=True)

    print('Before PCA:')
    print(class_report_before_pca)
    print('')
    print('After PCA:')
    print(class_report_after_pca)
    print('')
    print('')
    

if __name__ == '__main__':
    main()


 