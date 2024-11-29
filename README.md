# Feedforward Neural Network on CIFAR-10

This project implements a feedforward neural network to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.  
It is an exploratory project aimed to better understand the behavior and performance of feedforward neural networks on image classification tasks.  
  
the results and findings, which are detailed in the [final report](report.pdf).


## Project Structure

- `data/`: Folder in which the CIFAR-10 dataset should be placed.
- `code/`: Python scripts for training and evaluating the model.
- `README.md`: Project documentation.

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Matplotlib

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/FeedforwardNN-CIFAR10.git
    cd FeedforwardNN-CIFAR10
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the main training script:
    ```bash
    python code/main_model.py
    ```


## Results

The model achieves an accuracy of approximately 70% on the CIFAR-10 test set. Further improvements can be made by tuning hyperparameters and experimenting with different architectures as  
it is discussed in the report.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- The CIFAR-10 dataset is provided by the Canadian Institute for Advanced Research.
- This project is inspired by various online tutorials and resources on neural networks and deep learning.
