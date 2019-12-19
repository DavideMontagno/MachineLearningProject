
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from sklearn.model_selection import KFold
#https://matplotlib.org/index.html

dataset_tr = numpy.loadtxt('../project/ML-CUP19-TR.csv', delimiter=',', dtype = numpy.float64)

X = dataset_tr[:, 1:-2]
#y_tr = dataset_tr[:, -2:]

#x_ts = dataset_ts[:, 1:]

'''
w_initial
kernel initialization
'''
#(nesterov only for batch)
def train(x_tr,y_tr,x_ts,y_ts,eta = 0.015,alpha = 0.7 ,nEpoch = 350, lambda_param = 0.01, nLayer = 3 , nUnitPerLayer = 20, batch_size = 100):
    model = Sequential()
    model.add(Dense(20,input_dim=20, kernel_initializer='glorot_normal', activation='sigmoid'))
    for i in range(0,nLayer):
        model.add(Dense(nUnitPerLayer, kernel_regularizer=l2(lambda_param), kernel_initializer='glorot_normal', activation='sigmoid'))
    model.add(Dense(2, kernel_initializer='glorot_normal', activation='linear'))

    sgd = SGD(learning_rate=eta, momentum=alpha,nesterov=False)
    model.compile(optimizer=sgd, loss='mean_squared_error',metrics=['mae'])
    history = model.fit(x_tr,y_tr, epochs=nEpoch, batch_size=batch_size,verbose=0)
    score = model.evaluate(x_ts, y_ts, verbose=0)
    return history, score


etas = [0.1, 0.2]
alphas = [0.7,0.9]
for eta in etas:
    for alpha in alphas:
        kfold = KFold(n_splits=10, random_state=None, shuffle=False)
        cvscores = []
        for train_index, test_index in kfold.split(X):
            x_tr = dataset_tr[train_index,1:-2]
            y_tr = dataset_tr[train_index,-2:]
            x_ts = dataset_tr[test_index,1:-2]
            y_ts = dataset_tr[test_index,-2:]
            history, score = train(x_tr,y_tr,x_ts,y_ts,eta,alpha)
            cvscores.append(score)
        averageLoss = 0
        for score in cvscores:
            averageLoss = averageLoss + score[0]
        averageLoss = averageLoss / len(cvscores)
        print("Eta: "+str(eta) + "  Alpha: " + str(alpha) + " AverageLoss: " + str(averageLoss))

