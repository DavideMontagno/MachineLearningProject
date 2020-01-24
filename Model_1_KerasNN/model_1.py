import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils import plot_model
import keras.backend as K
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os


# Just disables the warning about AVX AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_tr = numpy.loadtxt(
    '../project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)

X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]

#print(K.sqrt(K.sum(K.square([3,0] - [4,0]), axis=-1)))
#exit(1)
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
    return K.mean(K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1)))

# (nesterov only for batch)


def trainAndEvaluate(x_tr, y_tr, x_ts, y_ts, eta=0.015, alpha=0.7, nEpoch=350, lambda_param=0.01, nUnitPerLayer=20,
                     nLayer=3,
                     batch_size=32):
    model = Sequential()
    #model.add(Dense(20, input_dim=20, kernel_initializer='glorot_normal', activation='relu'))
    for i in range(0, nLayer):
        model.add(Dense(nUnitPerLayer - 3*i, kernel_regularizer=l2(lambda_param), kernel_initializer='glorot_normal',
                        activation='relu'))

    model.add(Dense(2, kernel_initializer='glorot_normal', activation='linear'))

    sgd = SGD(learning_rate=eta, momentum=alpha, nesterov=False)
    #model.compile(optimizer=sgd, loss=euclidean_distance_loss, metrics=['mse', 'mae', coeff_determination])
    model.compile(optimizer=sgd, loss=euclidean_distance_loss)
    #to-do stopping criteria when loss < k
    history = model.fit(x_tr, y_tr, validation_data=(
        x_ts, y_ts), epochs=nEpoch, batch_size=batch_size, verbose=0)
    # score = model.evaluate(x_ts, y_ts, verbose=0)
    #plot_model(model, to_file='model_now.png',show_shapes=True)
    return history


etas = [0.001]
alphas = [0.85]
nEpoc = 100
lambdas = [0.0001]
nUnitLayer = 16
batch_size = 64
def cross_validation1():
    for eta in etas:
        for alpha in alphas:
            for lambda_param in lambdas:
                fig, (plt1, plt2) = plt.subplots(2, 1)
                #plt1.ylabel('Loss')
                #plt2.xlabel('Epoch')
                kfold = KFold(n_splits=10, random_state=None, shuffle=True)
                cvscores = []
                nFold = 0
                forLegend = []
                for train_index, test_index in kfold.split(X):
                    x_tr = X[train_index]
                    y_tr = Y[train_index]
                    x_ts = X[test_index]
                    y_ts = Y[test_index]
                    history = trainAndEvaluate(x_tr, y_tr, x_ts, y_ts, eta, alpha, nEpoc, lambda_param,
                                                    nUnitLayer, 3,
                                                    batch_size)
                    #score = [history.history['val_loss'][-1], history.history['val_mse'][-1], history.history['val_mae'][-1], history.history['val_coeff_determination'][-1]]
                    cvscores.append([history.history['loss'][-1], history.history['val_loss'][-1]])
                    # Plot training loss values (just half of them)
                    if nFold % 3 == 0:
                        #plt.subplot(2, 1, 1)
                        plt1.plot(history.history['loss'])
                        plt1.plot(history.history['val_loss'])
                        #plt.subplot(2, 1, 2)
                        plt2.plot(range(25, nEpoc),
                                  history.history['loss'][25:])
                        plt2.plot(range(25, nEpoc),
                                  history.history['val_loss'][25:])
                        forLegend.append('Train ' + str(nFold))
                        forLegend.append('Validation ' + str(nFold))
                    nFold += 1
                averageLoss = 0
                averageLossTR = 0
                for score in cvscores:
                    averageLoss += score[0]
                    averageLossTR += score[1]
                averageLoss /= len(cvscores)
                averageLossTR /= len(cvscores)
                print("Eta: " + str(eta) + "  Alpha: " + str(alpha) + " nEpoch: " + str(nEpoc) + " Lambda: " + str(
                    lambda_param) + " nUnitPerLayer: " + str(nUnitLayer) + " Batch size: " + str(
                    batch_size) + " AverageLoss (on validation set): " + str(
                    averageLossTR))
                fig.legend(forLegend, loc='center right')
                fig.suptitle('Model loss ' + str(eta) + '_' + str(alpha) + '_' + str(nEpoc) + '_' + str(
                    lambda_param) + '_' + str(batch_size))
                fig.savefig('./plots/3i_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoc) + '_' + str(
                    lambda_param) + '_' + str(batch_size) + '_' + str(
                    averageLossTS) + '.png', dpi=600)
                plt.close()
        # last computer best parameters
cross_validation1()




