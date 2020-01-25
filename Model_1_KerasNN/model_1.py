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
    './project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)

X = dataset_tr[:, 1:-2]
Y = dataset_tr[:, -2:]



def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

# (nesterov only for batch)


def train_and_learningcurve(x_tr, y_tr, x_ts, y_ts, eta=0.015, alpha=0.7, nEpoch=350, lambda_param=0.01, nUnitPerLayer=20,
                     nLayer=3,
                     batch_size=32):
    model = Sequential()
    #model.add(Dense(20, input_dim=20, kernel_initializer='glorot_normal', activation='relu'))
    for i in range(0, nLayer):
        model.add(Dense(nUnitPerLayer - 2*i, kernel_regularizer=l2(lambda_param), kernel_initializer='glorot_normal',
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



def cross_validation1( eta, alpha, lambda_param, batch_size, nUnitLayer,nEpoc):
                fig, (plt1, plt2) = plt.subplots(2, 1)
                kfold = KFold(n_splits=10, random_state=None, shuffle=True)
                cvscores = []
                nFold = 0
                forLegend = []
                for train_index, test_index in kfold.split(X):
                    x_tr = X[train_index]
                    y_tr = Y[train_index]
                    x_ts = X[test_index]
                    y_ts = Y[test_index]
                    history = train_and_learningcurve(x_tr, y_tr, x_ts, y_ts, eta, alpha, nEpoc, lambda_param,
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
                averageLossTS = 0
                for score in cvscores:
                    averageLoss += score[0]
                    averageLossTS += score[1]
                averageLoss /= len(cvscores)
                averageLossTS /= len(cvscores)
                
                
                fig.legend(forLegend, loc='center right')
                fig.suptitle('Model loss ' + str(eta) + '_' + str(alpha) + '_' + str(nEpoc) + '_' + str(
                    lambda_param) + '_' + str(batch_size))
                fig.savefig('./plots_final/Keras_Validation/Keras_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoc) + '_' + str(
                    lambda_param) + '_' + str(batch_size) + '_' + str(
                    averageLossTS) + '_'+str(nUnitLayer)+'.png', dpi=600)
                plt.close()
                return averageLossTS
                


def best_model1(cross_validation):
    best_eta = 0.001
    best_alpha = 0.85
    best_lambda = 0.0005
    best_batch_size = 64
    best_nUnitLayer = 25
    nEpoch = 85
   
    if(cross_validation):
        min_loss=float('inf')
        nUnitLayers = [25]
        etas = [0.0009,0.001,0.0015]
        alphas = [0.6,0.65,0.7]
        lambdas = [0.0009,0.001,0.0015]
        batch_sizes = [64,96]
        for nUnitLayer in nUnitLayers:
            for eta in etas:
                for alpha in alphas:
                    for _lambda in lambdas:
                        for batch_size in batch_sizes:
                            tmp = cross_validation1( eta, alpha, _lambda, batch_size, nUnitLayer,nEpoch)
                            if(tmp < min_loss):
                                    min_loss = tmp
                                    best_alpha = alpha
                                    best_batch_size = batch_size
                                    best_lambda = _lambda
                                    best_eta = eta
                                    best_nUnitLayer = nUnitLayer

    return train_and_predict(best_eta, best_alpha, best_lambda, best_batch_size, best_nUnitLayer,nEpoch)



def train_and_predict( eta, alpha, lambda_param, batch_size, nUnitPerLayer,nEpoch,nLayer=3):
    
    model = Sequential()
    #model.add(Dense(20, input_dim=20, kernel_initializer='glorot_normal', activation='relu'))
    for i in range(0, nLayer):
        model.add(Dense(nUnitPerLayer - 3*i, kernel_regularizer=l2(lambda_param), kernel_initializer='glorot_normal',
                        activation='relu'))

    model.add(Dense(2, kernel_initializer='glorot_normal', activation='linear'))

    sgd = SGD(learning_rate=eta, momentum=alpha, nesterov=False)
    #model.compile(optimizer=sgd, loss=euclidean_distance_loss, metrics=['mse', 'mae', coeff_determination])
    model.compile(optimizer=sgd, loss=euclidean_distance_loss)
  
    history = model.fit(X, Y,validation_split=0, epochs=nEpoch, batch_size=batch_size, verbose=0)
    # score = model.evaluate(x_ts, y_ts, verbose=0)
    #plot_model(model, to_file='model_now.png',show_shapes=True)
    dataset_bs = numpy.genfromtxt('./project/ML-CUP19-TS.csv', delimiter=',', dtype=numpy.float64)
   
    return model.predict(dataset_bs[:,1:])
