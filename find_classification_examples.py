# Find example images where the main model is wrong and right

import main_model as mm
import matplotlib.pyplot as plt
import torch
import numpy as np
from read_data import read_test_data
from preprocess_data import Preprocessed_data
from accuracy_metrics import cls_report
from normalize_data import normalize_data
import variables as var

def find_classification_examples():
    # Load the test data
    test_data, test_labels = read_test_data()
    test_labels = np.array(test_labels)  # για καποιον λογο το επαιρνε ως λιστα και οχι ως numpy array
    og_test_data = test_data.copy()

    # Preprocess the data
    test_data = normalize_data(test_data)
    test_data = Preprocessed_data.create_3d_data(test_data)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    # Create the model
    temp = mm.Model_trainer()
    model = temp.load_model_from_file()
    # Get the predictions
    with torch.no_grad():
        predictions = model(test_data)
    # Get the class labels
    y_pred = torch.argmax(predictions, dim=1)
    y_pred = y_pred.numpy()
    print(f'Predictions: {y_pred[:10]}')
    print(f'Test labels: {test_labels[:10]}')

    # Find the indices of the wrong predictions
    wrong_indices = np.where(y_pred != test_labels)[0]
    # Get 6 random indices
    np.random.seed(42)  # Για να παιρνω καθε φορά τα 6 ίδια indices -- αμα δεν πάρω ραντομ επιστρέφει πολλά βατράχια
    wrong_indices = np.random.choice(wrong_indices, 6, replace=False)
    # Find the indices of the right predictions
    right_indices = np.where(y_pred == test_labels)[0]
    # Get 6 random indices
    right_indices = np.random.choice(right_indices, 6, replace=False)
    
    class_name = var.labels
    # Plot the first 6 wrong and right predictions
    fig, axs = plt.subplots(2, 6, figsize=(20, 10))
    for i in range(6):
        axs[0, i].imshow(og_test_data[wrong_indices[i]].reshape(3, 32, 32).transpose(1, 2, 0))
        axs[0, i].set_title(f'Pred: {class_name[y_pred[wrong_indices[i]]]}, True: {class_name[test_labels[wrong_indices[i]]]}')
        axs[0, i].axis('off')
        axs[1, i].imshow(og_test_data[right_indices[i]].reshape(3, 32, 32).transpose(1, 2, 0))
        axs[1, i].set_title(f'Pred: {class_name[y_pred[right_indices[i]]]}, True: {class_name[test_labels[right_indices[i]]]}')
        axs[1, i].axis('off')
        axs[1, i].set_title(f'Pred: {class_name[y_pred[right_indices[i]]]}, True: {class_name[test_labels[right_indices[i]]]}')
        axs[1, i].axis('off')
    fig.text(0.04, 0.75, 'Wrong Predictions', va='center', rotation='vertical', size='large')
    fig.text(0.04, 0.25, 'Right Predictions', va='center', rotation='vertical', size='large')
    plt.show()


def main():
    find_classification_examples()


if __name__ == '__main__':
    main()