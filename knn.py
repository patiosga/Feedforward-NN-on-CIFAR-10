import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import variables as var
from pca_analysis import pca_decomposition
from read_data import read_data, read_test_data
from normalize_data import normalize_data
from accuracy_metrics import cls_report
from preprocess_data import Preprocessed_data


# Perform k-NN classification on the CIFAR-10 dataset before and after PCA
def knn_experiment(pd: Preprocessed_data, neighbors, PCA=False, mean_data=False):
    if PCA == True and mean_data == True:
        data = pd.all_data_normalized
        test_data = pd.all_test_data_normalized
    elif PCA == True:
        data = pd.pca_data
        test_data = pd.pca_test_data
    elif mean_data == True:
        data = pd.normalized_mean_data
        test_data = pd.normalized_mean_test_data
    else:
        data = pd.training_data
        test_data = pd.test_data
                                      
    # Create and fit k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=neighbors)
    knn.fit(data, pd.training_labels)
    # Predict labels for test data
    predicted_labels = knn.predict(test_data)
    # Calculate accuracy
    class_report = cls_report(pd.test_labels, predicted_labels)

    return class_report


def run_knn(pd: Preprocessed_data, neighbors):
    # Perform k-NN classification on the CIFAR-10 dataset before and after PCA
    print(f'Number of neighbors = {neighbors}')
    print('----------------------')

    class_report_before_pca = knn_experiment(pd, neighbors)
    class_report_after_pca = knn_experiment(pd, neighbors, PCA=True)
    class_report_mean = knn_experiment(pd, neighbors, mean_data=True)
    class_report_mean_pca = knn_experiment(pd, neighbors, PCA=True, mean_data=True)
    

    print('Before PCA:')
    print(class_report_before_pca)
    print('')
    print('After PCA:')
    print(class_report_after_pca)
    print('')
    print('Using mean values extraction:')
    print(class_report_mean)
    print('')
    print('Using mean values extraction with PCA:')
    print(class_report_mean_pca)
    


def main():
    training_data, training_labels = read_data()
    test_data, test_labels = read_test_data()

    processed_data = Preprocessed_data(training_data, training_labels, test_data, test_labels)
    processed_data.preprocess_data()
    # Run k-NN classification
    run_knn(processed_data, 1)
    run_knn(processed_data, 3)


if __name__ == '__main__':
    main()

