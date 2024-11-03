from sklearn.neighbors import NearestCentroid
import numpy as np
import variables as var
from pca_analysis import pca_decomposition
from read_data import read_data, read_test_data
from normalize_data import normalize_data
from accuracy_metrics import cls_report
from feature_extraction import mean_values_extraction
from preprocess_data import Preprocessed_data


def nearest_centroid_experiment(pd: Preprocessed_data, PCA=False, mean_data=False):
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
    nearest_centroid = NearestCentroid()
    nearest_centroid.fit(data, pd.training_labels)
    # Predict labels for test data
    predicted_labels = nearest_centroid.predict(test_data)
    # Calculate accuracy
    class_report = cls_report(pd.test_labels, predicted_labels)

    return class_report



def main():
    training_data, training_labels = read_data()
    test_data, test_labels = read_test_data()

    processed_data = Preprocessed_data(training_data, training_labels, test_data, test_labels)
    processed_data.preprocess_data()

    class_report_before_pca = nearest_centroid_experiment(processed_data)
    class_report_after_pca = nearest_centroid_experiment(processed_data, PCA=True)
    class_report_mean = nearest_centroid_experiment(processed_data, mean_data=True)
    class_report_mean_pca = nearest_centroid_experiment(processed_data, PCA=True, mean_data=True)
    
    print('Before PCA:')
    print(class_report_before_pca)
    print('')
    print('After PCA:')
    print(class_report_after_pca)
    print('')
    print('Mean values extraction:')
    print(class_report_mean)
    print('')
    print('Mean values extraction with PCA:')
    print(class_report_mean_pca)
    print('')
    print('')
    

if __name__ == '__main__':
    main()


 