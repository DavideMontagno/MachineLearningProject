from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

dataset_tr = np.genfromtxt(
    '../project/ML-CUP19-TR.csv', delimiter=',', dtype=np.float64)

X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]
Y1 = dataset_tr[:, -1]
Y2 = dataset_tr[:, -2]

def call_loss(y_real, y_pred):
    sum_tot = 0
    for i in range(len(y_real)):
        sum_tot+= np.sqrt(np.sum(np.power((y_real[i]-y_pred[i]), 2)))
    return sum_tot / len(y_real)
def loss_fn(y_real, y_pred): return call_loss(y_real, y_pred)
    #return np.mean(np.sqrt(np.sum(np.power((y_real-y_pred), 2))))

splits_kfold = 10
Cs = [10]
gammas = [0.05]
epsilons = [0.1]


def plot_learning_curve():
    kfold = KFold(n_splits=splits_kfold,
                  random_state=None, shuffle=True)
    for traing_index, test_index in kfold.split(X):
        x_tr = X[traing_index]
        y_tr = Y[traing_index]
        x_ts = X[test_index]
        y_ts = Y[test_index]
        all_loss = []
        n_examples = []
        for step in range(2, 101, 1):
            ind_x = int(step * (len(x_tr)/100))
            ind_y = int(step * (len(y_tr)/100))
            this_x_tr = x_tr[0:ind_x, :]
            this_y_tr = y_tr[0:ind_y, :]
            svr = SVR()
            mor = MultiOutputRegressor(svr)
            mor.fit(this_x_tr, this_y_tr)
            y_pred = mor.predict(x_ts)
            this_loss = loss_fn(y_pred, y_ts)
            n_examples.append(int(step * (len(x_tr)/100)))
            all_loss.append(this_loss)
        plt.plot(n_examples, all_loss)
        plt.title("Learning Curve SVM")
        plt.xlabel("Number of training examples")
        plt.ylabel("Loss (Mean Euclidian Distance)")
        plt.legend(["Loss on validation set"])
        plt.show()


plot_learning_curve()
