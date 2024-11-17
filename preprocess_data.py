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
        self.test_labels = np.array(test_labels)  # για καποιον λογο το επαιρνε ως λιστα και οχι ως numpy array

    def preprocess_data(self):
        # Normalize the data
        self.training_data /= 255.0
        self.test_data /= 255.0


        # One hot encoding of the labels
        self.one_hot_training_labels = np.eye(var.num_of_classes)[self.training_labels]  # διαλέγει τις training_label γραμμές από τον μοναδιαίο πίνακα 
        self.one_hot_test_labels = np.eye(var.num_of_classes)[self.test_labels]

    @staticmethod
    def create_3d_data(data: np.ndarray) -> np.ndarray:
        # Create a 3D NumPy array of shape (50000, 32, 32) for the red, green, or blue channel
        data_3d = data.reshape(-1, 3, 32, 32)
        # print(data_3d.shape)  # (50000, 3, 32, 32)
        return data_3d
    
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




