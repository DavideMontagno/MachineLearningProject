from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor

dataset_tr = np.genfromtxt(
    '../project/ML-CUP19-TR.csv', delimiter=',', dtype=np.float64)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if(device.type == 'cuda'):
    print(torch.cuda.get_device_name(torch.cuda.current_device()),
          'enabled:', torch.backends.cudnn.enabled)


X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]
Y1 = dataset_tr[:, -1]
Y2 = dataset_tr[:, -2]
def loss_fn(y_real, y_pred): return torch.div(
    torch.sum(F.pairwise_distance(y_real, y_pred, p=2)), len(y_real))

splits_kfold = 10
Cs = [100]
gammas = [0.05,0.01,0.1]
epsilons = [0.1]

kfold = KFold(n_splits=splits_kfold,
    random_state=None, shuffle=True)
def cross_validation3():
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
                    
                    svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
                    mor = MultiOutputRegressor(svr)
                    mor.fit(x_tr,y_tr)

                    y_pred =  mor.predict(x_ts)
                    y_ts_cuda = torch.tensor(y_ts).cuda(device.type)
                    y_pred_cuda = torch.tensor(y_pred).cuda(device.type)
                    all_loss.append(loss_fn( y_ts_cuda, y_pred_cuda))
                    #score = mor.score(x_ts, y_ts)
                    
                    plt.figure()
                    s = 50
                    a = 0.4
                    plt.scatter(y_ts[:,0],y_ts[:,1],edgecolor='k',c="navy", s=s, marker="s", alpha=a, label="Data")
                    plt.scatter(y_pred[:,0],y_pred[:,1], c="cornflowerblue", s=s, alpha=a)
                    plt.title("Support Vector Regression")
                    plt.legend()
                    plt.show()
                    '''
                    print('Creating plot...')
                    plt.savefig('./plots/data_'+str(epsilon)+'_'+str(C)+'_'+str(gamma)+'_'+str(score)+'.png', dpi=1000)
                    plt.close()
                    print('Completed!')
                    '''
                all_loss = torch.tensor(list(all_loss)).cuda(device.type)
                print("\nMy loss : ",C,' ',gamma,' ',epsilon,'Mean:',torch.mean(all_loss).item(),'Variance:',torch.var(all_loss).item())
                






cross_validation3()