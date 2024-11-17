import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from preprocess_data import Preprocessed_data
from feature_extraction import create_3d_data
from sklearn.model_selection import train_test_split
import variables as var
import read_data
from accuracy_metrics import cls_report
from data_augmentation import augment_data


def define_model() -> nn.Module:
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ELU(alpha=1.0),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout(0.15),

        nn.Conv2d(16, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.ELU(alpha=1.0),
        nn.MaxPool2d(kernel_size=2, stride=2),  # 16 x 8 x 8
        nn.Flatten(),

        nn.Linear(16 * 8 * 8, 128),
        nn.BatchNorm1d(128),
        nn.ELU(alpha=1.0),
        nn.Dropout(0.3),

        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ELU(alpha=1.0),
        nn.Dropout(0.3),

        nn.Linear(64, 10)
    )
    return model

def main():
    # Load the CIFAR-10 dataset from the pickle file
    # processed_data: Preprocessed_data = Preprocessed_data.read_from_pickle_file()
    print("Loaded preprocessed data")

    tr_data, tr_labels = read_data.read_data()
    tr_data = create_3d_data(tr_data)
    # tr_data = augment_data(tr_data)
    # print("Augmented data")
    # print(tr_data.shape)
    # Normalize the data
    tr_data = tr_data / 255

    one_hot_tr_labels = np.eye(var.num_of_classes)[tr_labels]  # διαλέγει τις training_label γραμμές από τον μοναδιαίο πίνακα 
    
    te_data, te_labels = read_data.read_test_data()
    te_labels = np.array(te_labels)
    te_data = create_3d_data(te_data)
    te_data = te_data / 255.0
    one_hot_te_labels = np.eye(var.num_of_classes)[te_labels]

    print("Loaded raw data")
    print(tr_data.shape)
    print(one_hot_tr_labels.shape)
    print(te_data.shape)
    print(one_hot_te_labels.shape)


    # Seperate the training data into 90% training and 10% validation and create a DataLoader object for each --> straified sampling για να εχω ισαξια εκπροσωπηση των κλασεων στο training και validation set
    # Εδω χρησιμοποιω τις μεσες τιμες των στηλων/γραμμων
    # ΠΡΟΣΟΧΗ το εχω αλλαξει ωστε να εχω 10% training και 90% validation για να εχω μικρο dataset για να δω αν δουλευει το μοντελο
    # Χρειάζομαι τα one hot labels
    X_train, X_val, y_train, y_val = train_test_split(tr_data, one_hot_tr_labels, test_size=0.8, random_state=42, stratify=tr_labels)
    print("Did stratified sampling")
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)
    
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_train_one_label = torch.argmax(y_train, dim=1)
    print(X_train.shape)
    print(y_train.shape)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    y_val_one_label = torch.argmax(y_val, dim=1)

    # Create the test data and labels
    X_test = torch.tensor(te_data, dtype=torch.float32)
    y_test = torch.tensor(one_hot_te_labels, dtype=torch.float32)
    y_test_one_label = torch.tensor(te_labels, dtype=torch.float32)
    print("Created tensor data and labels")
    

    model: nn.Module = define_model()
    print(model)
    print("Defined model")

    # loss function and optimizer
    loss_fn = nn.CrossEntropyLoss() # CrossEntropyLoss expects the output of the model to be logits, not probabilities
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=1, threshold=0.01)
    print("Defined loss function, optimizer and scheduler")

    n_epochs = 10   # number of epochs to run
    batch_size = 12  # size of each batch
    batches_per_epoch = len(X_train) // batch_size

    # collect statistics
    train_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}")
        for i in range(batches_per_epoch):
            start = i * batch_size
            # take a batch
            Xbatch = X_train[start:start+batch_size]
            ybatch = y_train[start:start+batch_size]
            # forward pass
            y_pred = model(Xbatch)
            # print(y_pred.shape)
            # print(ybatch.shape)
            loss: torch.Tensor = loss_fn(y_pred, ybatch)
        
            # backward pass
            loss.backward()
            # update weights
            optimizer.step()
            optimizer.zero_grad()

        # evaluate model at end of epoch
        with torch.no_grad():
            y_pred_val = model(X_val)
            y_pred_train = model(X_train)
        # Loss
        loss_train = loss_fn(y_pred_train, y_train)
        train_loss.append(float(loss_train))

        y_pred_val = torch.argmax(y_pred_val, dim=1)
        y_pred_train = torch.argmax(y_pred_train, dim=1)
        # Accuracy
        accuracy_val = (y_pred_val == y_val_one_label).float().mean()
        accuracy_train = (y_pred_train == y_train_one_label).float().mean()

        # LR scheduler - μειώνω lr αν η ακρίβεια στο validation set δεν αυξηθεί για 3 συνεχόμενα epochs
        print("LR value used:", scheduler.get_last_lr())
        scheduler.step(accuracy_val)
        

        val_acc.append(float(accuracy_val))
        train_acc.append(float(accuracy_train))
        print(f"End of {epoch+1}, training accuracy {train_acc[-1]}, validation set accuracy {val_acc[-1]}")


    # evaluate model with test set
    with torch.no_grad():
        y_pred = model(X_test)
    y_pred = torch.argmax(y_pred, dim=1)
    test_acc = (y_pred == y_test_one_label).float().mean()
    print(cls_report(np.array(y_test_one_label), np.array(y_pred)))  # BASED ON THE LAST EPOCH
    print("Training done")
    print("TEST ACC:", test_acc)



    import matplotlib.pyplot as plt
    # Plot the loss metrics
    plt.plot(train_loss)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.ylim(0)
    plt.show()
    
    
    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="test")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0)
    plt.show()

    

if __name__ == '__main__':
    main()
