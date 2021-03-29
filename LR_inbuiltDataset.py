'''
1. Linear Regression model trained on sythetic dataset
2. after training, model is tested with the data
3. two graphs are compared
4. model is saved i the hard disk

'''


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import pickle

if torch.cuda.is_available():
    gpu = torch.device('cuda:0')

X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)


X_numpy = X_numpy.astype(np.float32)
Y_numpy = Y_numpy.astype(np.float32)

x = torch.from_numpy(X_numpy).to(device=gpu)
y = torch.from_numpy(Y_numpy).to(device=gpu)



y = y.view(x.shape[0],-1)

n_samples, n_features = x.shape

input_size = n_features
output_size = 1
learning_rate = 0.01

print("n_sample = {} and n_features = {}".format(n_samples, n_features))
print("input_size {} and output_size {}".format(input_size, output_size))

model = nn.Linear(input_size, output_size).to(gpu)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

num_epochs = 100

for epoch in range(num_epochs):

    y_predicted = model(x)
    loss = criterion(y_predicted, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # if (epoch+1) % 10 == 0:
    #     print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')



#x = torch.tensor([0.56],dtype=torch.float32, device='cuda:0')
#print(model(x).detach().item())

# saving the trained model in hard disk
#torch.save(model.state_dict(),'LR_sample.torch')

predicted = model(x).detach().to('cpu').numpy()

plt.scatter(X_numpy, Y_numpy)
plt.plot(X_numpy, predicted,color='red')
plt.show()