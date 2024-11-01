import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def main():
    # Define transforms for data augmentation and normalization
    print("Defining tranforms for data augmentation and normalization...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # Random crop with padding
        transforms.RandomHorizontalFlip(),      # Random horizontal flip
        transforms.ToTensor(),                  # Convert image to PyTorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize with mean and std
    ])

    # Load CIFAR-10 training dataset
    print("Loading CIFAR-10 training dataset...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    
    # Create data loaders for batching
    print("Creating data loaders for batching...")
    # train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

    # Check the shape of one batch
    print("Checking the shape of one batch...")
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)
    images, labels = train_dataset[0]
    print("Batch of images shape:", images.shape)
    print("Example of one image:", images[0])
    print("Batch of labels shape:", labels.shape)
    

if __name__ == '__main__':
    main()
