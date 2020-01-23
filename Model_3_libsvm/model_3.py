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
    sum_tot=0
    for i in range(len(y_real)):
        sum_tot += np.sqrt(np.sum((y_real[i]-y_pred[i])**2))
    return  sum_tot/len(y_real)

splits_kfold = 10
Cs = [10,20,30]
gammas = [0.1,0.5]
epsilons = [0.1,0.3]

kfold = KFold(n_splits=splits_kfold,
    random_state=None, shuffle=True)

for epsilon in epsilons:
    for C in Cs:
        for gamma in gammas:
            all_loss = []
            for traing_index, test_index in kfold.split(X):
                x_tr = X[traing_index]
                y_tr = Y[traing_index]
                #y_tr1 = Y1[traing_index]
                #y_tr2 = Y2[traing_index]                 
                x_ts = X[test_index] 
                #y_ts1 = Y1[test_index]
                #y_ts2 = Y2[test_index]
                
                y_ts = Y[test_index]

                #svr_rbf1 = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                #svr_rbf1.fit(x_tr, np.ravel(y_tr1))

                #svr_rbf2 = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                #svr_rbf2.fit(x_tr, np.ravel(y_tr2))

                #y_pred1 = svr_rbf1.predict(x_ts)
                #y_pred2 = svr_rbf2.predict(x_ts)

                mor = MultiOutputRegressor(SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon))
                mor.fit(x_tr,y_tr)

                '''y_pred_couple = []
                for i in range(len(y_pred1)):
                    y_pred_couple.append([y_pred1[i], y_pred2[i]])
                '''
                y_pred =  mor.predict(x_ts)
                #all_loss.append(call_loss(y_ts, y_pred_couple))
                all_loss.append(call_loss(y_ts, y_pred))
            print("\nMy loss : ",C,' ',gamma,' ',epsilon,' ',np.mean(all_loss),'-', np.var(all_loss)**2)

