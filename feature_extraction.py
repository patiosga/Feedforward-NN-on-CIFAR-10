from read_data import read_data
from normalize_data import normalize_data
import numpy as np


def preprocess_data(data: np.ndarray) -> np.ndarray:
    # Create a 3D NumPy array of shape (50000, 32, 32) for the red, green, or blue channel
    data_3d = data.reshape(-1, 3, 32, 32)
    # print(data_3d.shape)  # (50000, 3, 32, 32)

    return data_3d


def mean_column_values_per_colour(data):
    """
    Calculate the mean of each column in the data.
    """
    d3_image = preprocess_data(data)
    true_mean_values = d3_image.mean(axis=2)  # μέση τιμή για κάθε στήλη για κάθε χρώμα --> (50000)x3x32
    # axis=1 για να κανουν colapse οι γραμμές πάνω στις στήλες δηλ. στον αξονα 2

    return true_mean_values


def mean_row_values_per_colour(data):
    """
    Calculate the mean of each row in the data.
    """
    mean_image = preprocess_data(data)
    true_mean_values = mean_image.mean(axis=3)  # μέση τιμή για κάθε γραμμή για κάθε χρώμα --> (50000)x3x32

    return true_mean_values


def mean_values_extraction(data):
    mean_columns: np.ndarray = mean_column_values_per_colour(data)
    mean_columns = mean_columns.reshape(-1, 96)  # (50000)x3x32 --> (50000)x96
    # print(mean_columns.shape)  # (50000, 96)
    mean_rows: np.ndarray = mean_row_values_per_colour(data)
    mean_rows = mean_rows.reshape(-1, 96)  # (50000)x3x32 --> (50000)x96
    # print(mean_rows.shape)  # (50000, 96)
    
    mean_values = np.concatenate((mean_columns, mean_rows), axis=1)  # concatenate κατά μήκος του δευτερου αξονα ωστε να λειτουργουν ως εξτρα features
    # print(mean_values.shape)  # (50000, 192)

    return mean_values


def main():
    # Read CIFAR-10 training data
    data, labels = read_data()
    print(data.shape)  # (50000, 3072)
    print(labels.shape)  # (50000,)

    flattened_mean_data = mean_values_extraction(data)

    print(flattened_mean_data.shape)  # (192,)
    print(flattened_mean_data)
    

if __name__ == '__main__': 
    main()