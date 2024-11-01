from sklearn.neighbors import NearestCentroid
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np


def main():
    # Define transforms for data augmentation and normalization
    print("Defining tranforms for data augmentation and normalization...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # Random crop with padding
        transforms.RandomHorizontalFlip(),      # Random horizontal flip
        transforms.ToTensor(),                  # Convert image to PyTorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize with mean and std
    ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 training dataset
    print("Loading CIFAR-10 training dataset...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    # Convert the dataset to NumPy arrays
    X = []  # Feature matrix
    y = []  # Labels

    for img, label in train_dataset:
        # Flatten the image (3, 32, 32) -> (3072,)
        img_flat = img.numpy().flatten()
        X.append(img_flat)
        y.append(label)

    query_image = test_dataset[0][0].numpy().flatten()

    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)


    clf = NearestCentroid()
    clf.fit(X, y)

    print(f"Predicted label for the query image: {clf.predict(query_image.reshape(1, -1))[0]}")
    print("Actual label for the query image: ", test_dataset[0][1])


if __name__ == '__main__':
    main()


 