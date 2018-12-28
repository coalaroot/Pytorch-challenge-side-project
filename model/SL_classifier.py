import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models, utils
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

import zipfile
with zipfile.ZipFile('Sign-language-digits-dataset.zip', 'r') as f:
    f.extractall()
#data upload
X_0 = np.load("X.npy")
Y = np.load("Y.npy")

#add an extra channel to our images to be able to use convolutional
# neural networks and convert our numpy arrays to Pytorch tensors.
X = X_0.reshape((-1, 1, 64, 64))
images = torch.from_numpy(X)
labels = torch.from_numpy(Y_l)

# create a custom-made Dataset, where each image has a corresponding
# label, allowing us to shuffle, split and manipulate our dataset.
class Dataset(Dataset):
  #creating a custom dataset
  def __init__(self, images, labels):
        #init images and data
        self.images = images
        self.labels = labels

  def __len__(self):
        #getting lenght of dataset
        return len(self.images)

  def __getitem__(self, idx):
        # Select sample
        image = self.images[idx]
        label = self.labels[idx]

        return image, label
sign_language_dataset = Dataset(images, labels)

# initiate parameters for splitting our data
num_workers = 1
batch_size =16
valid_size = 0.1
test_size = 0.1

num_train = len(sign_language_dataset)
indices = list(range(num_train))
np.random.shuffle(indices)

split = int(np.floor(0.8 * num_train))
split_1 = int(np.floor(0.1 * num_train))
split_2 = split+split_1

train_idx, valid_idx, test_idx = (indices[:split], indices[split:split_2],
                                  indices[split_2+1:])

# define samplers for obtaining training, validation and test batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler =  SubsetRandomSampler(test_idx)
# load our batches
train_loader = torch.utils.data.DataLoader(sign_language_dataset,
                                           batch_size=batch_size,
                                           sampler=train_sampler,
                                           num_workers=num_workers)
valid_loader = torch.utils.data.DataLoader(sign_language_dataset,
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(sign_language_dataset,
                                           batch_size=batch_size,
                                           sampler=test_sampler,
                                           num_workers=num_workers)

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

model = Net()
# move tensors to GPU if CUDA is available
if train_on_gpu:
    model.cuda()

#loss function
criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = optim.Adam(model.parameters(), lr=0.003)

# number of epochs to train the model
n_epochs = 20
valid_loss_min = np.Inf # track change in validation loss

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    # train the model

    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)

    # validate the model

    model.eval()
    for data, target in valid_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()

        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)

    train_loss = train_loss/len(train_idx)
    valid_loss = valid_loss/len(valid_idx)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model_sl.pt')
        valid_loss_min = valid_loss

# track test loss
test_loss = 0.0
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

model.eval()
# iterate over test data
for data, target in test_loader:
    # move tensors to GPU if CUDA is available
    if train_on_gpu:
        data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())

    # calculate test accuracy for each object class
    for i in range(2):
        label = labels.tolist()[i]
        class_correct[label] += correct_tensor.tolist()[i]
        class_total[label] += 1

# # average test loss
test_loss = test_loss/len(test_loader.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
            classes[i], 100 * class_correct[i] / class_total[i],
            np.sum(class_correct[i]), np.sum(class_total[i])))
    else:
        print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
    100. * np.sum(class_correct) / np.sum(class_total),
    np.sum(class_correct), np.sum(class_total)))

# obtain one batch of test images
dataiter = iter(test_loader)
images, labels = dataiter.next()
images.numpy()

# move model inputs to cuda, if GPU available
if train_on_gpu:
    images = images.cuda()

output = model(images)
_, preds_tensor = torch.max(output, 1)
preds = np.squeeze(preds_tensor.numpy()) if not train_on_gpu else np.squeeze(preds_tensor.cpu().numpy())

# plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(15,5))
for idx in np.arange(12):
    ax = fig.add_subplot(2, 12/2, idx+1, xticks=[], yticks=[])
    plt.imshow(images[idx][0])
    ax.set_title("Pred {} (True {})".format(classes[preds[idx].tolist()], classes[labels[idx].tolist()]),
                 color=("green" if preds[idx]==labels[idx].item() else "red"))
