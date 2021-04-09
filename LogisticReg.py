'''
1. Logistic Regression using class
2. storing the hyperparameters in log files

'''
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import logging

# setting the logging level

logger = logging.getLogger(__name__) 
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s: %(lineno)d: %(message)s')

file_handler = logging.FileHandler('Regression.log')
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


# load breast cancer data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_sample, n_features = X.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))

y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

logger.info(" --------------------------- --------------------- ----------------------")

logger.info("shape of X_train {}".format(X_train.shape))
logger.info("shape of X_test {}".format(X_test.shape))

y_train = y_train.view(X_train.shape[0],-1)
y_test = y_test.view(X_test.shape[0],-1)

logger.info("shape of y_train {}".format(y_train.shape))
logger.info("shape of y_test {}".format(y_test.shape))

# 1 model

class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
    
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x))
        return y_predicted

model = LogisticRegression(n_features)

# loss optimizer
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# training loop
num_epochs = 100
logger.info("learning rate = {}, total epochs = {}".format(learning_rate,num_epochs))
for epoch in range(num_epochs):
    # forward pass and loss
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)

    loss.backward()

    optimizer.step()

    # reset gradient values to zero
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
    
    # evalution should not be the part of compuration graph
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {acc:.4f}')
    logger.info("accuracy = {0:.4f}".format(acc))
        
