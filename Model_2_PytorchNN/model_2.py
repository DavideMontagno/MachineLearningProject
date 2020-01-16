import torch
import numpy

dataset_tr = numpy.genfromtxt(
    '../project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)

X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]

D_in = 20
H = 26
D_out = 2

# No bias
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, H-2),
    torch.nn.ReLU(),
    torch.nn.Linear(H-2, H-4),
    torch.nn.ReLU(),
    torch.nn.Linear(H-4, D_out),
)

print(model)


def init_weights(m):
    if type(m) == torch.nn.Linear:
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.xavier_uniform(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        # m.bias.data.fill_(0.01)

#Euclidian Distance
loss_fn = lambda x1, x2: torch.sqrt(torch.norm(x1 - x2, 2))
learning_rate = 0.002
momentum = 0.9
lambda_param = 0.01

model.apply(init_weights)

optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=lambda_param)

batch_size = 64
nEpoch = 1000
for epoch in range(nEpoch):
    for i in range(int(len(X) / batch_size)):
        # TODO requires_grad=True ???
        # TODO: validation set
        optimizer.zero_grad()
        x = torch.tensor(list(X[i * batch_size:(i + 1) * batch_size]), dtype=torch.float, requires_grad=True)
        y = torch.tensor(list(Y[i * batch_size:(i + 1) * batch_size]), dtype=torch.float, requires_grad=True)
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        if (i == int(len(X) / batch_size) - 1):
            print((int)(epoch), loss.item())
        loss.backward()
        optimizer.step()
