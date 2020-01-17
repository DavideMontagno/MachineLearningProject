import torch
import numpy
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

dataset_tr = numpy.genfromtxt(
    '../project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)

X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]

D_in = 20
H = 26
D_out = 2


def init_weights(m):
    if type(m) == torch.nn.Linear:
        # torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        # m.bias.data.fill_(0.01)


#Euclidian Distance
def loss_fn(x1, x2): return torch.sqrt(torch.norm(x1 - x2, 2))


eta = 0.002
alpha = 0.9
lambda_param = 0.001

batch_size = 64
nEpoch = 200
kfold = KFold(n_splits=10, random_state=None)

nFold = 0
forLegend = []
for train_index, test_index in kfold.split(X):
    x_tr = X[train_index]
    y_tr = Y[train_index]
    x_ts = X[test_index]
    y_ts = Y[test_index]

    # No bias
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(H, H-2, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(H-2, H-4, bias=False),
        torch.nn.ReLU(),
        torch.nn.Linear(H-4, D_out, bias=False),
    )

    model.apply(init_weights)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=eta, momentum=alpha, weight_decay=lambda_param)

    score_tr = []
    score_ts = []
    for epoch in range(nEpoch):
        loss = torch.zeros(1)
        for i in range(int(len(x_tr) / batch_size)):
            # TODO requires_grad=True ???
            # TODO: validation set
            optimizer.zero_grad()

            x = torch.tensor(list(
                x_tr[i * batch_size:(i + 1) * batch_size]), dtype=torch.float, requires_grad=True)
            y = torch.tensor(list(
                y_tr[i * batch_size:(i + 1) * batch_size]), dtype=torch.float, requires_grad=True)

            y_pred = model(x)
            loss = loss_fn(y, y_pred)
            loss.backward()
            optimizer.step()
        score_tr.append(loss.item())
        y_pred_ts = model(torch.tensor(list(x_ts), dtype=torch.float, requires_grad=True))
        loss_ts = loss_fn(torch.tensor(list(y_ts), dtype=torch.float, requires_grad=True), y_pred_ts)
        score_ts.append(loss_ts.item())

    averageLoss = 0
    for cv_value in score_tr:
        averageLoss += cv_value
    averageLoss /= len(score_tr)
    if nFold % 2 == 0:
            plt.plot(score_tr)
            plt.plot(score_ts)
            plt.title('Model loss ' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
                lambda_param) + '_' + str(batch_size))
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            forLegend.append('Train ' + str(nFold))
            forLegend.append('Validation ' + str(nFold))
    nFold += 1
print("Eta: " + str(eta) + "  Alpha: " + str(alpha) + " nEpoch: " + str(nEpoch) + " Lambda: " + str(
    lambda_param) + " nUnitPerLayer: " + str(H) + " Batch size: " + str(
    batch_size) + " AverageLoss (on validation set): " + str(
    averageLoss))
plt.legend(forLegend, loc='upper right')
plt.savefig('./plots/learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
    lambda_param) + '_' + str(batch_size) + '_' + str(
    averageLoss) + '.png', dpi=500)
plt.close()