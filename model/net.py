import torch
import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
      
        self.conv1_1 = nn.Conv2d(1, 8, 3, padding=1)      
        self.conv1_2 = nn.Conv2d(8, 8, 3, padding=1)      
        self.conv1_3 = nn.Conv2d(8, 8, 3, padding=1)
        
        self.pool1 = nn.MaxPool2d(2, 2) 
        
        self.conv2_1 = nn.Conv2d(8, 16, 3, padding=1)      
        self.conv2_2 = nn.Conv2d(16, 16, 3, padding=1)      
        self.conv2_3 = nn.Conv2d(16, 16, 3, padding=1)

        self.pool2 = nn.MaxPool2d(2, 2) 
        
        self.conv3_1 = nn.Conv2d(16, 32, 3, padding=1)      
        self.conv3_2 = nn.Conv2d(32, 32, 3, padding=1)      
        self.conv3_3 = nn.Conv2d(32, 32, 3, padding=1)
        
        self.pool3 = nn.MaxPool2d(2, 2) 
        
        self.conv4_1 = nn.Conv2d(32, 64, 3, padding=1)      
        self.conv4_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4_3 = nn.Conv2d(64, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(64, 64)
        self.do = nn.Dropout(p=0.35)
        self.fc2 = nn.Linear(64, 10)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # 64x64x1 => 32x32x8
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = F.relu(self.conv1_3(x))
        x = self.pool1(x)
        
        # 32x32x8 => 16x16x16
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = self.pool2(x)
        
        # 16x16x16 => 8x8x32
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)
        
        # 8x8x32 => 8x8x10
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))

        x = self.do(self.fc1(x.view(-1, 64, 8*8).sum(dim=-1)))
        x = F.log_softmax(self.fc2(x), dim=-1)

        return x