import numpy as np
import variables as var
from pca_analysis import pca_decomposition
from normalize_data import normalize_data
from feature_extraction import mean_values_extraction
from read_data import read_data, read_test_data
import pickle

class Preprocessed_data():
    def __init__(self, training_data, training_labels, test_data, test_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.test_data = test_data
        self.test_labels = np.array(test_labels)

    def preprocess_data(self):
        # Normalize the data
        self.training_data = normalize_data(self.training_data)
        self.test_data = normalize_data(self.test_data)

        # PCA decomposition - Πρώτα PCA και έπειτα κανονικοποίηση
        self.pca_data = (pca_decomposition(self.training_data, var.pca_components))
        self.pca_test_data = (pca_decomposition(self.test_data, var.pca_components))

        # Mean data extraction
        mean_data = mean_values_extraction(self.training_data)
        mean_test_data = mean_values_extraction(self.test_data)

        self.normalized_mean_data: np.ndarray = normalize_data(mean_data)
        self.normalized_mean_test_data: np.ndarray = normalize_data(mean_test_data)

        # Combination of pca and mean data - normalized pca features για να μην εχουν μεγάλες τιμές σε σχέση με τα mean values
        self.all_data_normalized = np.concatenate((normalize_data(self.pca_data), self.normalized_mean_data), axis=1)
        self.all_test_data_normalized = np.concatenate((normalize_data(self.pca_test_data), self.normalized_mean_test_data), axis=1)  # στο axis=1 γιατί θέλουμε να ενώσουμε τα δεδομένα κατά μήκος του δεύτερου αξονα

    @staticmethod
    def write_to_pickle_file(data_object: 'Preprocessed_data'):
        with open('preprocessed_data.pkl', 'wb') as f:
            pickle.dump(data_object, f)

    @staticmethod
    def read_from_pickle_file() -> 'Preprocessed_data':
        with open('preprocessed_data.pkl', 'rb') as f:
            return pickle.load(f)

def main():
    # Write object to pickle for future use - φουυυυυλ γρηγορο
    training_data, training_labels = read_data()
    test_data, test_labels = read_test_data()

    processed_data = Preprocessed_data(training_data, training_labels, test_data, test_labels)
    processed_data.preprocess_data()
    Preprocessed_data.write_to_pickle_file(processed_data)

   
if __name__ == '__main__':
    main()




