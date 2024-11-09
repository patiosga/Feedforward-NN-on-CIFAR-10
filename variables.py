# Labels dictionary
labels: dict[int: str] = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

pca_components:int = 221  # μετά από PCA analysis στο ομόνυμο αρχείο - τόσα components χρειάζονται για το 95% της διακύμανσης μετά την κανονικοποίηση
num_of_classes:int = 10