import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier


# Labels dictionary
labels = {
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

# Load and transform the CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize
])

# Load CIFAR-10 dataset (you can specify train=False for the test set)
train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)

# Convert the dataset to NumPy arrays
X = []  # Feature matrix
y = []  # Labels

for img, label in train_dataset[:100]:
    # Flatten the image (3, 32, 32) -> (3072,)
    img_flat = img.numpy().flatten()
    X.append(img_flat)
    y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Create a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=1)

# Fit the model on the training data
knn.fit(X, y)  # Fit on the first 100 samples for demonstration

# Example: Query with a specific image (here we take the first image as an example)
query_image = X[100].reshape(1, -1)  # Reshape to 1 sample with 3072 features
predicted_label = knn.predict(query_image)

print(f"Predicted label for the query image: {labels[predicted_label[0]]}")
print("Actual label for the query image: ", labels[y[0]])
