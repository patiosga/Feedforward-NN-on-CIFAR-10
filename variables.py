# Labels dictionary
labels: list[str] = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

pca_components:int = 221  # μετά από PCA analysis στο ομόνυμο αρχείο - τόσα components χρειάζονται για το 95% της διακύμανσης μετά την κανονικοποίηση
num_of_classes:int = 10