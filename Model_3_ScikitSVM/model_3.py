from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

dataset_tr = np.genfromtxt(
    './project/ML-CUP19-TR.csv', delimiter=',', dtype=np.float64)

X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]
splits_kfold = 10


def call_loss(y_real, y_pred):
    sum_tot = 0
    for i in range(len(y_real)):
        sum_tot += np.sqrt(np.sum(np.power((y_real[i]-y_pred[i]), 2)))
    return sum_tot / len(y_real)


def loss_fn(y_real, y_pred): return call_loss(y_real, y_pred)
#return np.mean(np.sqrt(np.sum(np.power((y_real-y_pred), 2))))



s = 50
a = 0.4
def best_model3(cross_validation):
    best_C = 10
    best_gamma= 0.05
    best_epsilon = 0.1
    if(cross_validation):
        kfold = KFold(n_splits=splits_kfold, random_state=None, shuffle=True)
        min_loss=float('inf')
        Cs = [8] #low confidence with data => Low C - not too much (maybe underfitting)
        gammas = [0.07] #took from scikit (relantionship with C)
        epsilons = [0.1,0.2] 
        for epsilon in epsilons:
            for C in Cs:
                for gamma in gammas:
                    all_loss = []
                    for traing_index, test_index in kfold.split(X):
                        x_tr = X[traing_index]
                        y_tr = Y[traing_index]
                        x_ts = X[test_index]
                        y_ts = Y[test_index]

                        svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                        mor = MultiOutputRegressor(svr)
                        mor.fit(x_tr, y_tr)
                        y_pred = mor.predict(x_ts)
                        all_loss.append(loss_fn(y_ts, y_pred))
                        
                tmp = np.mean(all_loss)
                if(tmp<min_loss):
                    min_loss=tmp
                    best_C = C
                    best_epsilon=epsilon
                    best_gamma = gamma

    print(best_C,'-',best_gamma,'-',best_epsilon)  
    plot_learning_curve(best_C,best_gamma,best_epsilon)
    svr = SVR(kernel='rbf', C=best_C, gamma=best_gamma, epsilon=best_epsilon)
    mor = MultiOutputRegressor(svr)
    mor.fit(X, Y)
    dataset_bs = np.genfromtxt('./project/ML-CUP19-TS.csv', delimiter=',', dtype=np.float64)
    return mor.predict(dataset_bs[:,1:])


def plot_learning_curve(C,gamma,epsilon):
    kfold = KFold(n_splits=splits_kfold,
                  random_state=None, shuffle=True)
    for traing_index, test_index in kfold.split(X):
        x_tr = X[traing_index]
        y_tr = Y[traing_index]
        x_ts = X[test_index]
        y_ts = Y[test_index]
        all_loss = []
        n_examples = []
        for step in range(2, 102, 2):
            ind_x = int(step * (len(x_tr)/100))
            ind_y = int(step * (len(y_tr)/100))
            this_x_tr = x_tr[0:ind_x, :]
            this_y_tr = y_tr[0:ind_y, :]
            svr = SVR(C=C,gamma=gamma,epsilon=epsilon,verbose=False)
            mor = MultiOutputRegressor(svr)
            mor.fit(this_x_tr, this_y_tr)
            y_pred = mor.predict(x_ts)
            this_loss = loss_fn(y_pred, y_ts)
            n_examples.append(int(step * (len(x_tr)/100)))
            all_loss.append(this_loss)
        plt.plot( n_examples, all_loss)
        plt.title("Learning Curve SVM")
        plt.xlabel("Number of training examples")
        plt.ylabel("Loss (Mean Euclidian Distance)")
        plt.legend(["Loss on validation set"])
        plt.savefig('./plots_final/svm_learning_curve_' + str(C) + '_' + str(gamma) +'.png', dpi=500)
        plt.close()
        return


