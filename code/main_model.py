import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time

from preprocess_data import Preprocessed_data
from read_data import read_data, read_test_data
from accuracy_metrics import cls_report
from read_data import read_data, read_test_data


class CIFAR10Model(nn.Module):
    def __init__(self, model=None, input_channels=3, output_size=10, n=16):
        super(CIFAR10Model, self).__init__()
        if model is None:
            first_layer = 2*n  # 32 default
            second_layer = n
            third_layer = n**2  # 256
            fourth_layer = n**2 // 2  # 128
            self.model = nn.Sequential(
                nn.Conv2d(input_channels, first_layer, kernel_size=3, padding=1),
                nn.BatchNorm2d(first_layer),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.15),

                nn.Conv2d(first_layer, second_layer, kernel_size=3, padding=1),
                nn.BatchNorm2d(second_layer),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),  # second_layer x 8 x 8
                nn.Flatten(),

                nn.Linear(second_layer * 8 * 8, third_layer),
                nn.BatchNorm1d(third_layer),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(third_layer, fourth_layer),
                nn.BatchNorm1d(fourth_layer),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(fourth_layer, output_size)
            )
        else: 
            self.model = model

    def forward(self, x):
        return self.model(x)
    
    def __str__(self):
        return str(self.model)
    

class Model_trainer:
    def __init__(self, epochs=5, batch_size=32, model=None, loss_fn=None, optimizer=None, scheduler=None, neurons=16, learning_rate=0.001):
        '''
        Κατασκευαστής για την κλάση Model_trainer
        Αρχικοποιεί τα epochs, batch_size, μοντέλο, συνάρτηση απώλειας, βελτιστοποιητή και scheduler
        '''
        self.epochs = epochs
        self.batch_size = batch_size

        # Ορίζω το μοντέλο, τη συνάρτηση απώλειας, τον βελτιστοποιητή και τον scheduler
        # Το αφήνω να δημιουργηθεί από τον χρήστη αν δεν δοθεί για λόγους debugging
        if model is None:
            self.model = CIFAR10Model(n=neurons)
        else:
            self.model = model
        
        if loss_fn is None:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = loss_fn

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = optimizer
        
        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, threshold=0.005)
        else:
            self.scheduler = scheduler
        
        self.train_loss = []
        self.train_acc = []
        self.val_acc = []
        self.LR_values = []

    def process_data(self):
        '''
        Διαβάζει, κανονικοποιεί, διαχωρίζει τα δεδομένα σε train και validation και τα μετατρέπει σε 3D torch tensors
        '''
        training_data, training_labels = read_data()
        test_data, test_labels = read_test_data()

        # Η κλαση Preprocessed_data κάνει την κανονικοποίηση, το split και τη μετατροπή σε tensors
        self.pdd = Preprocessed_data(training_data, training_labels, test_data, test_labels)
        self.pdd.X_train = Preprocessed_data.create_3d_data(self.pdd.X_train)
        self.pdd.X_test = Preprocessed_data.create_3d_data(self.pdd.X_test)
        self.pdd.split_data(test_size=0.2)
        self.pdd.convert_to_tensor()

        # Αντιγράφω τα δεδομένα στις μεταβλητές της κλάσης για ευκολία
        self.X_train, self.X_val, self.y_train, self.y_val = self.pdd.X_train, self.pdd.X_val, self.pdd.y_train, self.pdd.y_val
        self.one_hot_y_train = self.pdd.one_hot_y_train
        self.X_test = self.pdd.X_test
        self.y_test = self.pdd.y_test

    def train(self):
        '''
        Εκπαιδεύει το μοντέλο και κρατάει τα αποτελέσματα για το training και το validation set
        '''
        batches_per_epoch = len(self.X_train) // self.batch_size

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}")
            for i in range(batches_per_epoch):
                start = i * self.batch_size  # Αρχή του batch
                Xbatch = self.X_train[start : start + self.batch_size]
                ybatch = self.one_hot_y_train[start : start + self.batch_size]  
                # παίρνω τα one-hot labels για υπολογισμό του loss αλλιώς χρησιμοποιώ κανονικά labels

                # Υπολογισμός του loss και backpropagation
                y_pred = self.model(Xbatch)
                loss = self.loss_fn(y_pred, ybatch)

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Υπολογισμός του loss και της ακρίβειας στο validation set και το training set μετά το τέλος κάθε εποχής
            with torch.no_grad():
                y_pred_val = self.model(self.X_val)
                y_pred_train = self.model(self.X_train)

            loss_train = self.loss_fn(y_pred_train, self.one_hot_y_train)
            self.train_loss.append(float(loss_train))

            y_pred_val = torch.argmax(y_pred_val, dim=1)
            y_pred_train = torch.argmax(y_pred_train, dim=1)

            accuracy_val = (y_pred_val == self.y_val).float().mean()
            accuracy_train = (y_pred_train == self.y_train).float().mean()

            print("LR value used:", self.scheduler.get_last_lr()[0])  # επέστρεφε λίστα
            self.LR_values.append(self.scheduler.get_last_lr()[0])
            # Μειώνω LR αν χρειάζεται
            self.scheduler.step(accuracy_val)  # μειώνω το learning rate με βάση την ακρίβεια στο validation set

            # Τα κρατάω σε λίστες για να τα πλοτάρω στο τέλος
            self.val_acc.append(float(accuracy_val))
            self.train_acc.append(float(accuracy_train))
            print(f"End of {epoch+1}, training accuracy {self.train_acc[-1]}, validation set accuracy {self.val_acc[-1]}")


    def test(self):
        '''
        Υπολογισμός της ακρίβειας στο test set και εκτύπωση του classification report
        '''
        with torch.no_grad():
            y_pred = self.model(self.X_test)
        y_pred = torch.argmax(y_pred, dim=1)
        test_acc = (y_pred == self.y_test).float().mean()
        print(cls_report(np.array(self.y_test), np.array(y_pred)))
        print("Test accuracy:", test_acc)
        return test_acc


    def plot_training_progress(self):
        '''
        Πλοτάρει την ακρίβεια και το loss στο training και validation set
        '''
        epochs = range(self.epochs)
        # Training loss
        plt.plot(epochs, self.train_loss, label='Training Loss')
        plt.title("Training loss reduction")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.ylim(0)
        plt.show()
    
        # Training and validation accuracy
        plt.plot(epochs, self.train_acc, label='Training Accuracy')
        plt.plot(epochs, self.val_acc, label='Validation Accuracy')
        plt.title("Training set and validation set accuracy progression")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.ylim((0, 1))
        plt.legend()
        plt.show()

        # Learning rate reduction
        plt.plot(epochs, self.LR_values, label='Learning Rate')
        plt.yscale('log', base=2)  # αλλάζω την κλίμακα του y αξονα γιατί αλλιώς δεν φαίνονται καλά τα αποτελέσματα
        plt.title("Learning Rate Reduction")
        plt.xlabel('Epochs')
        plt.ylabel('LR')
        plt.ylim(0)
        plt.legend()
        plt.show()


    def write_model_to_file(self):
        torch.save(self.model.state_dict(), 'cifar10_main_model.pth')


    def load_model_from_file(self):
        self.model.load_state_dict(torch.load('cifar10_main_model.pth'))
        self.model.eval()
        return self.model


    def run(self, load_model=False, process_data=True):
        if process_data:
            self.process_data()

        if load_model:
            self.load_model_from_file()  # παίρνω το αρχείο από τον δίσκο -- by default κάνει train νέο μοντέλο
        else:
            self.train()
            self.write_model_to_file()  # αποθηκεύω το μοντέλο στον δίσκο
            self.plot_training_progress()  # πλοτάρω την πρόοδο του μοντέλου

        return self.test()
        
    

def main():
    start = time.time()
    trainer = Model_trainer(epochs=5)
    print(trainer.run(load_model=False))
    print("Time taken: ", time.time()-start)  # Για καλύτερη χρονομέτρηση γίνονται comment out στη run() οι μέθοδοι write_model_to_file() και plot_training_progress()



if __name__ == '__main__':
    main()





