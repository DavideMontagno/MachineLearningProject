import torch
import numpy
import math
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


dataset_tr = numpy.genfromtxt(
    '../project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)

X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]
D_in = 20
nUnitLayer = 26
D_out = 2
loss_fn = lambda a,b : torch.sqrt(torch.norm(a - b, 2)) #Euclidian Distance
etas = [0.005]
alphas = [ 0.75, 0.8, 0.85, 0.9]
lambdas = [0.001]
batch_size = 64

nFold=0
forLegend = []
nEpochs = [2000]

dataset_tr = numpy.genfromtxt(
    '../project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, nUnitLayer, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.input = torch.nn.Linear(D_in, nUnitLayer)
        self.hidden1 = torch.nn.Linear(nUnitLayer, nUnitLayer)
        self.hidden1_activation = torch.nn.ReLU()
        self.hidden2 = torch.nn.Linear(nUnitLayer, nUnitLayer)
        self.hidden2_activation = torch.nn.ReLU()
        self.output = torch.nn.Linear(nUnitLayer, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        hidden_t = self.input(x)
        h_relu = self.hidden1_activation(hidden_t)
        h_relu2 = self.hidden2_activation(h_relu)
        y_pred = self.output(h_relu2)
        return y_pred


def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)

model = TwoLayerNet(D_in, nUnitLayer, D_out)
model.apply(init_weights)
model.cuda()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(device.type=='cuda'):
    print(torch.cuda.get_device_name(torch.cuda.current_device()),'enabled:',torch.backends.cudnn.enabled)
if device.type == 'cuda':
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(device)/1024**3,1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(device)/1024**3,1), 'GB')


for nEpoch in nEpochs:
    for eta in etas:
        for alpha in alphas:
            for lambda_param in lambdas:
                cvscores = []
                nFold = 0
                forLegend = []
                kfold = KFold(n_splits=10, random_state=None)
                for traing_index, test_index in kfold.split(X):
                    x_tr = X[traing_index] 
                    y_tr = Y[traing_index] 
                    x_ts = X[traing_index] 
                    y_ts = Y[traing_index]
                    model = TwoLayerNet(D_in, nUnitLayer, D_out)
                    model.apply(init_weights)
                    model.cuda()
                    optimizer = optimizer = torch.optim.SGD(
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
                                x_tr[i * batch_size:(i + 1) * batch_size]), dtype=torch.float, requires_grad=True).cuda(device.type)
                            y = torch.tensor(list(
                                y_tr[i * batch_size:(i + 1) * batch_size]), dtype=torch.float, requires_grad=True).cuda(device.type)

                            y_pred = model(x)
                            loss = loss_fn(y, y_pred)
                            loss.backward()
                            optimizer.step()
                        score_tr.append(loss.item())
                        y_pred_ts = model(torch.tensor(list(x_ts), dtype=torch.float, requires_grad=True).cuda(device.type))
                        loss_ts = loss_fn(torch.tensor(list(y_ts), dtype=torch.float, requires_grad=True).cuda(device.type), y_pred_ts)
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
                                    lambda_param) + " nUnitPerLayer: " + str(nUnitLayer) + " Batch size: " + str(
                                    batch_size) + " AverageLoss (on validation set): " + str(
                                    averageLoss))
                plt.legend(forLegend, loc='upper right')
                plt.savefig('./plots/learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
                                                                        lambda_param) + '_' + str(batch_size) + '_' + str(
                                                                        averageLoss) + '.png', dpi=500)
                plt.close()


