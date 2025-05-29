import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_dim=256, num_layers=2):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
            hidden_dim (int): size of the hidden layers (default 256)
        """
        super().__init__() 
        # take input size, and return hidden layer size
        
        assert num_layers >= 1, "You need at least one hidden layer." # we can create as many layers as we want

        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(input_size, hidden_dim))
        
        for i in range(num_layers-1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            
        self.output_layer=(nn.Linear(hidden_dim, n_classes))
            
            
        #self.fc1 = nn.Linear(input_size, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        #self.fc3 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        #x = F.relu(self.fc1(x)) # used func
        #x = F.relu(self.fc2(x))
        #preds = self.fc(x) 
        
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
            
        preds=self.output_layer(x)
        
        return preds

#class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    #def __init__(self, input_channels, n_classes, num_layers=8, base_channels=32):
        #"""
        #Initialize the network.

        #You can add arguments if you want, but WITH a default value, e.g.:
        #    __init__(self, input_channels, n_classes, my_arg=32)

        #Arguments:
        #    input_channels (int): number of channels in the input
        #    n_classes (int): number of classes to predict
        #"""
        #super().__init__()
        
        #assert num_layers >= 2, "Need at least input and output conv layers"

        #self.convs = nn.ModuleList()

        # First layer: input → base_channels, reduce spatial size
        #self.convs.append(
        #    nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1)  # halves H, W
        #)

        # Intermediate layers: keep same size and channels
        #for _ in range(num_layers - 2):
        #    self.convs.append(
        #        nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)  # preserve H, W
        #    )

        # Optionally: one final conv to reduce further (or just flatten after this)
        #self.global_pool = nn.AdaptiveAvgPool2d((4, 4))  # fixed output size (can be 7, 2, etc.)
        #self.flatten = nn.Flatten()
        #self.classifier = nn.Sequential(
        #    nn.Linear(base_channels * 4 * 4, 128),
        #    nn.ReLU(),
        #    nn.Linear(128, n_classes)
        #)
        
        
class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
        
        # Feature extractor: Conv (feature map) → ReLU → MaxPool → Conv → ReLU → MaxPool
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  # output: (N, 32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)              # output: (N, 64, 14, 14)
        self.pool = nn.MaxPool2d(2, 2)  # halves H and W

        # Fully connected classifier: FC → ReLU → FC
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.conv1(x))   # (N, 32, 28, 28)
        x = self.pool(x)           # (N, 32, 14, 14)
        x = F.relu(self.conv2(x))  # (N, 64, 14, 14)
        x = self.pool(x)           # (N, 64, 7, 7)
        x = self.flatten(x)        # (N, 64*7*7)
        x = F.relu(self.fc1(x))    # (N, 128)
        preds = self.fc2(x)        # (N, 7)
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs.

        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            # Optional, put it cuz yes
            avg_loss, avg_acc = self.train_one_epoch(dataloader, ep)
            #if(ep%10 == 0):
            #    print(f"Epoch {ep+1}/{self.epochs} — Loss: {avg_loss:.4f} — Accuracy: {avg_acc:.2f}%")
            print(f"Epoch {ep+1}/{self.epochs} — Loss: {avg_loss:.4f} — Accuracy: {avg_acc:.2f}%")

    def train_one_epoch(self, dataloader, ep): # to see
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        self.model.train()  # Important for dropout/batchnorm if used
        running_loss = 0.0
        correct = 0
        total = 0

        for x_batch, y_batch in dataloader:

            self.optimizer.zero_grad()
            outputs = self.model(x_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * x_batch.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = running_loss / total
        avg_acc = (correct / total) * 100
        return avg_loss, avg_acc

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation,
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        self.model.eval()
        pred_list = []

        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                outputs = self.model(x)
                preds = torch.argmax(outputs, dim=1)
                pred_list.append(preds)

   
        return torch.cat(pred_list, dim=0)

    def fit(self, training_data, training_labels): # not modifeid
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels).long())
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data): #not modifed
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()
