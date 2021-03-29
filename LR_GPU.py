'''
1. desing model input, output, forward pass
2. construct loss and optimizer
3. training loop
    - forward pass: computer prediction
    - backward pass: gradients
    - updated weights

'''


import torch
import torch.nn as nn

if torch.cuda.is_available():
    gpu = torch.device("cuda:0")

X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32, device=gpu)
Y = torch.tensor([[1], [4], [6], [8]], dtype=torch.float32, device=gpu)

X_test = torch.tensor([5], dtype=torch.float32, device=gpu)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features


model = nn.Linear(input_size,output_size)
model.to(gpu)

print(f'Prediction before training: f(5) = {model(X_test).item()}')

learning_rate = 0.01
n_iters = 100000

loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    y_pred = model(X)

    l = loss(Y, y_pred)

    l.backward()  # dl/dw

    optimizer.step()
    
    # we reseting the .grad attribute to zero for every iteration, else the grad value accumulates
    optimizer.zero_grad()



    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training: f(5) = {model(X_test).item()}')


