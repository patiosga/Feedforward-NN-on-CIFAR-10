import numpy as np
import variables as var
from pca_analysis import pca_decomposition
from normalize_data import normalize_data
from feature_extraction import mean_values_extraction


class Preprocessed_data():
    def __init__(self, training_data, training_labels, test_data, test_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.test_data = test_data
        self.test_labels = test_labels

    def preprocess_data(self):
        # Normalize the data
        self.training_data = normalize_data(self.training_data)
        self.test_data = normalize_data(self.test_data)

        # PCA with mean values extraction
        self.pca_data = pca_decomposition(self.training_data, var.pca_components)
        self.pca_test_data = pca_decomposition(self.test_data, var.pca_components)

        # Mean data extraction
        mean_data = mean_values_extraction(self.training_data)
        mean_test_data = mean_values_extraction(self.test_data)

        self.normalized_mean_data: np.ndarray = normalize_data(mean_data)
        self.normalized_mean_test_data: np.ndarray = normalize_data(mean_test_data)

        # Combination of pca and mean data and normalization again
        self.all_data_normalized = np.concatenate((self.pca_data, self.normalized_mean_data), axis=1)
        self.all_test_data_normalized = np.concatenate((self.pca_test_data, self.normalized_mean_test_data), axis=1)  # στο axis=1 γιατί θέλουμε να ενώσουμε τα δεδομένα κατά μήκος του δεύτερου αξονα




