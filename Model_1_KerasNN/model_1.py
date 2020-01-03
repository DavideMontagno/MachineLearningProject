import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
import keras.backend as K
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os
# Just disables the warning about AVX AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


dataset_tr = numpy.loadtxt('../project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)

X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]

'''
w_initial
kernel initialization
'''

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


def coeff_determination(y_true, y_pred):
    # r squared metric
    # https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


# (nesterov only for batch)
def trainAndEvaluate(x_tr, y_tr, x_ts, y_ts, eta=0.015, alpha=0.7, nEpoch=350, lambda_param=0.01, nUnitPerLayer=20, nLayer=3, 
          batch_size=100):
    model = Sequential()
    model.add(Dense(20, input_dim=20, kernel_initializer='glorot_normal', activation='sigmoid'))
    for i in range(0, nLayer):
        model.add(Dense(nUnitPerLayer, kernel_regularizer=l2(lambda_param), kernel_initializer='glorot_normal',
                        activation='sigmoid'))

    model.add(Dense(2, kernel_initializer='glorot_normal', activation='linear'))

    sgd = SGD(learning_rate=eta, momentum=alpha, nesterov=False)
    model.compile(optimizer=sgd, loss=euclidean_distance_loss, metrics=['mse', 'mae', coeff_determination])
    #todo add in fit validation set
    history = model.fit(x_tr, y_tr, epochs=nEpoch, batch_size=batch_size, verbose=0)
    score = model.evaluate(x_ts, y_ts, verbose=0)
    return history, score


etas = [0.001, 0.03, 0.7]
alphas = [0.6, 0.75, 0.8]
nEpocs = [325, 450]
lambdas = [0.01,0.02,0.03]
nUnitLayers = [18,20,23]

for eta in etas:
    for alpha in alphas:
        for nEpoc in nEpocs:
            for lambda_param in lambdas:
                for nUnitLayer in nUnitLayers:
                    kfold = KFold(n_splits=10, random_state=None, shuffle=True)
                    cvscores = []
                    nFold = 0
                    forLegend = []
                    for train_index, test_index in kfold.split(X):
                        x_tr = X[train_index]
                        y_tr = Y[train_index]
                        x_ts = X[test_index]
                        y_ts = Y[test_index]
                        history, score = trainAndEvaluate(x_tr, y_tr, x_ts, y_ts, eta, alpha,nEpoc,lambda_param,nUnitLayer)
                        # Plot training loss values (just half of them)
                        if nFold % 2 == 0:
                            plt.plot(history.history['loss'])
                            plt.title('Model loss '+str(eta)+'_' + str(alpha) +'_'+ str(nEpoc)+'_'+str(lambda_param)+'_'+str(nUnitLayer))
                            plt.ylabel('Loss')
                            plt.xlabel('Epoch')
                            #plt.show()
                            cvscores.append(score)
                            forLegend.append('Train '+str(nFold))
                        nFold +=1
                    plt.legend(forLegend, loc='upper right')
                    plt.savefig('./plots/learning_curve_'+str(eta)+'_' + str(alpha) +'_'+ str(nEpoc) +'_'+str(lambda_param)+'_'+str(nUnitLayer)+'.png',dpi=1000)
                    plt.close()
                    averageLoss = 0
                    averageMSE = 0
                    averageMAE = 0
                    averageR2 = 0
                    for score in cvscores:
                        averageLoss += score[0]
                        averageMSE += score[1]
                        averageMAE += score[2]
                        averageR2 += score[3]
                    averageLoss /= len(cvscores)
                    averageMSE /= len(cvscores)
                    averageMAE /= len(cvscores)
                    averageR2 /= len(cvscores)
                    print("Eta: " + str(eta) + "  Alpha: " + str(alpha) + " nEpoch: " + str(nEpoc) + " Lambda: " + str(lambda_param) + " nUnitPerLayer: " + str(nUnitLayer)+ " AverageLoss (on validation set): " + str(
                        averageLoss) + " AverageMSE: " + str(averageMSE) + " AverageMAE: " + str(
                        averageMAE) + " AverageR2: " + str(averageR2))
