from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import numpy as np


def accuracy(test_labels, predicted_labels):
    acc = np.mean(predicted_labels == test_labels)
    return acc 


def confusion_matrix(test_labels, predicted_labels):
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    return conf_matrix


def cls_report(test_labels, predicted_labels):
    # Αναλυτική αναφορά ταξινόμησης (precision, recall, f1-score, support) για κάθε κλάση ξεχωριστά
    report = classification_report(test_labels, predicted_labels)
    return report
