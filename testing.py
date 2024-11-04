import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from preprocess_data import Preprocessed_data
from sklearn.model_selection import train_test_split

''' !!! Σημαντικό !!!
The Dataset is responsible for accessing and processing single instances of data.
The DataLoader pulls instances of data from the Dataset (either automatically or with a sampler that you define), collects them in batches, and returns them for consumption by your training loop.
'''

def define_model(inputs: int, classes: int) -> nn.Module:
    model = nn.Sequential(
        nn.Linear(inputs, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, classes),
        nn.Softmax(dim=1)
    )
    return model

def main():
    # Load the CIFAR-10 dataset from the pickle file
    processed_data: Preprocessed_data = Preprocessed_data.read_from_pickle_file()

    # Seperate the training data into 90% training and 10% validation and create a DataLoader object for each --> straified sampling για να εχω ισαξια εκπροσωπηση των κλασεων στο training και validation set
    # Εδω χρησιμοποιω τις μεσες τιμες των στηλων/γραμμων
    # ΠΡΟΣΟΧΗ το εχω αλλαξει ωστε να εχω 10% training και 90% validation για να εχω μικρο dataset για να δω αν δουλευει το μοντελο
    X_train, X_val, y_train, y_val = train_test_split(processed_data.normalized_mean_data, processed_data.training_labels, test_size=0.95, random_state=42, stratify=processed_data.training_labels)
    print("Did stratified sampling")
    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_val.shape)
    # print(y_val.shape)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)  # Το reshape ειναι για να εχει το y_train το σωστο shape για το CrossEntropyLoss ???????????
    # print(X_train.shape)
    # print(y_train.shape)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    # Create the test data and labels
    X_test = torch.tensor(processed_data.normalized_mean_test_data, dtype=torch.float32)
    y_test = torch.tensor(processed_data.test_labels, dtype=torch.float32).reshape(-1, 1)
    print("Created tensor data and labels")

    # ΘΕΛΟΥΝ ΟΛΑ ΤΑ y ΜΕΤΑΤΡΟΠΗ ΣΕ ΟΝΕ-ΗΟΤ ΔΙΑΝΥΣΜΑΤΑ !!!!!!!!!!!!!!!



    model: nn.Module = define_model(inputs=192, classes=10)
    print(model)
    print("Defined model")

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()  # cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Defined loss function and optimizer")

    n_epochs = 50    # number of epochs to run
    batch_size = 64  # size of each batch
    batches_per_epoch = len(X_train) // batch_size

    for epoch in range(n_epochs):
        for i in range(batches_per_epoch):
            start = i * batch_size
            # take a batch
            Xbatch = X_train[start:start+batch_size]
            ybatch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(Xbatch)
            print(y_pred.shape)
            print(ybatch.shape)
            loss = loss_fn(y_pred, ybatch)
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()

    # evaluate trained model with test set
    with torch.no_grad():
        y_pred = model(X_test)
    accuracy = (y_pred.round() == y_test).float().mean()
    print("Accuracy {:.2f}".format(accuracy * 100))

    


if __name__ == '__main__':
    main()
