import time
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


# if torch.cuda.is_availabel():
#     torch.backends.cudnn.deterministic = True

"""
#### Setting 
"""
# Device
# device = torch.device("cuda:3" if torch.cuda.is_availabel() else "cpu")

# Hyperparameters
random_seed = 1
learing_rate = 0.1
num_epochs = 10
batch_size = 128
num_classes = 10

"""
### MINIST DATASET
"""
# transforms.ToTensor() : scales input images to 0-1 range
train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# checking the dataset
# for images, label in train_loader:
   # print('Image batch dimensions:', images.size())  28, 1, 28,28
   # print('Image label dimensions:', label.shape)    128


"""
    MODEL
"""


class ConvNet(torch.nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        # 28*28*1  => 28*28*4
        self.conv_1 = torch.nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        # 28*28*4  => 14*14*4
        self.pool_1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # 14*14*4  => 14*14*8
        self.conv_2 = torch.nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1)
        # 14*14*8  => 7*7*8
        self.pool_2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Weight sharing in last layer
        # self.linear_1 = torch.nn.Linear(7*7*8, num_classes)
        self.linear_1 = torch.nn.Linear(7*7*8, 1, bias=False)
        # define bias manually
        self.linear_1_bias = torch.nn.Parameter(torch.tensor(torch.zeros(num_classes),
                                                             dtype=self.linear_1.weight.dtype))

    def forward(self, x):
        out = self.conv_1(x)
        out = F.relu(out)
        out = self.pool_1(out)

        out = self.conv_2(out)
        out = F.relu(out)
        out = self.pool_2(out)

        # weight sharing in last layer, duplicate outputs over all output units
        logits = self.linear_1(out.view(-1, 7*7*8))
        # then manually add bias
        logits = logits + self.linear_1_bias

        probas = F.softmax(logits, dim=1)
        return logits, probas


torch.manual_seed(random_seed)
model = ConvNet(num_classes)

# model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learing_rate)


"""
   Training
"""


def compute_accuracy(model, data_loader):
    correct_pred , num_examples = 0, 0
    for features, targets in data_loader:
        # features = features.to(device)
        # targets = targets.to(device)
        logits, probas = model(features)
        _, pred = torch.max(probas, 1)
        correct_pred += (pred == targets).sum()
        num_examples += targets.size(0)
        return correct_pred.float() / num_examples * 100

start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        # features = features.to(device)
        # targets = targets.to(device)
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if not batch_idx % 50:
            print('Epoch: %03d/%03d  | Batch: %03d/%03d  | Cost: %.4f' % (epoch+1, num_epochs, batch_idx,
                                                                         len(train_loader), cost))
    model.eval()
    with torch.set_grad_enabled(False):    # save memory during inference
        print('Epoch: %03d/%03d  | training_acc: %.3f%%' %(epoch+1, num_epochs, compute_accuracy(model, train_loader)))
    print('Time eclapsed: %.2f min' % ((time.time() - start_time)/60))
print('Total Training Time: %.3f min' % ((time.time() - start_time)/60))




