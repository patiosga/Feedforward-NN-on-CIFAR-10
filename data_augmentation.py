import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from read_data import read_data
from feature_extraction import create_3d_data

from PIL import Image


from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def augment_data(data: np.ndarray) -> np.ndarray:
    '''
    Augment the data using random horizontal flips, rotations, crops, and color jittering.
    '''
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor()
    ])
    
    augmented_data = []
    for i in range(data.shape[0]):
        # Reorder channels for PIL compatibility
        img = data[i].transpose(1, 2, 0)  # Convert (3, 32, 32) to (32, 32, 3)
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        
        # Apply transformations and convert back to a tensor
        img_transformed = train_transforms(img)
        
        augmented_data.append(img_transformed.numpy())
    
    return np.array(augmented_data)

def main():
    data, _ = read_data()
    data = create_3d_data(data)
    print(data[:10])
    augmented_data = augment_data(data)
    print(augmented_data[:10])
    print(augmented_data.shape)







if __name__ == '__main__':
    main()


