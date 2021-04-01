import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

if torch.cuda.is_available():
    GPU = torch.device('cuda:0')

X, y = load_diabetes(return_X_y=True)

X = X.astype(np.float32)
y = y.astype(np.float32)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# changing y_test and y_train to clumn vectors
y_train = y_train.reshape(y_train.shape[0],-1)
y_test = y_test.reshape(y_test.shape[0],-1)

# converting numpy vectors to GPU tensors
X_train = torch.from_numpy(X_train).to(GPU)
X_test = torch.from_numpy(X_test).to(GPU)
y_test = torch.from_numpy(y_test).to(GPU)
y_train = torch.from_numpy(y_train).to(GPU)

# hyper parameters

NUM_INPUT_FEATURES = X_test.shape[1]
EPOCHS = 5
LEARNING_RATE = 0.01

class LinearRegresssion(nn.Module):
    def __init__(self, input_features):
        super(LinearRegresssion, self).__init__()
        self.linear = nn.Linear(input_features, 1)

    def forward(self, x):
        y_predicted = self.linear(x)
        return y_predicted

model = LinearRegresssion(NUM_INPUT_FEATURES).to(GPU)

# loss optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # if (epoch+1) % 10 == 0:
    #     print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# model evalution

with torch.no_grad():
    y_predicted = model(X_test).detach()


#convert pytorch tensor to numpy array

X_test = X_test.to('cpu').numpy()

y_predicted = y_predicted.to('cpu').numpy()
y_test = y_test.to('cpu').numpy()

sns.scatterplot(x=y_test,y=y_predicted,index=[0])
plt.plot()