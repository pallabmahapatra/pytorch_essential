'''

 output_image = (input_width - filter_size + 2*Padding)/starides + 1
                    (32 - 5 + 2*0) / 1 + 1 = 28

'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np

# check whether GPU is present.
# "GPU" holds the info to tranfer the variable to GPU  
if torch.cuda.is_available():
    GPU = torch.device('cuda:0')

NUM_EPOCHS = 4
BATCH_SIZE = 4
LEARNING_RATE = 0.001

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5) # 3 channels because each image has 3 colors
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # input channel size should be equal to output_channel size
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16, kernel_size=5)

        # fully connected layer
        self.fc1 = nn.Linear(in_features=16*5*5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10) # 10 different classifications
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16*5*5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

model = ConvNet().to(GPU)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

n_total_steps = len(train_loader)

for epoch in range(NUM_EPOCHS):
    for i, (images, lables) in enumerate(train_loader):
        # original shape = [4, 3, 32, 32] # 4 images have 3 colors and size of 32*32
        # input_layer =  3 input channels, 6 output channels, 5 kernel size
        images = images.to(GPU)
        lables = lables.to(GPU)

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, lables)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 2000 == 0:
            print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
        
print('Finished Training...')


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, lables in test_loader:
        images = images.to(GPU)
        lables = lables.to(GPU)
        outputs = model(images)

        _, predicted = torch.max(output, 1)
        n_samples += lables.size(0)
        n_correct += (predicted == lables).sum().item()

        for i in range(BATCH_SIZE):
            lable = lables[i]
            pred = predicted[i]
            if (lable == pred):
                n_class_correct[lable] += 1
            n_class_samples[lable] += 1
    
    acc = 100 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')
