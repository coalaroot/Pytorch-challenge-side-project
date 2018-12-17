import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
      
        self.conv1 = nn.Conv2d(1, 3, 2, padding=1)      
        self.conv2 = nn.Conv2d(3, 6, 2, padding=1)      
        self.conv3 = nn.Conv2d(6, 12, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)       
        self.fc1 = nn.Linear(8 * 8* 12, 250)       
        self.fc2 = nn.Linear(250 , 100)
        self.fc3 = nn.Linear(100, 10)       
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input and add hidden layers with dropout and
        # activation function ReLu
        x = self.dropout(x.view(-1, 8*8*12))     
        x = self.dropout(F.relu(self.fc1(x)))       
        x = self.dropout(F.relu(self.fc2(x)))        
        x = self.fc3(x)
        return x