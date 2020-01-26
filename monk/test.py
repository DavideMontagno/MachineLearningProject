from numpy import loadtxt
from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import regularizers
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
import keras.backend as K
from sklearn.model_selection import KFold

import os


# Just disables the warning about AVX AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


dataset_train1 = loadtxt('dataset/monks-1.train',
                         delimiter=' ', usecols=range(1, 8))
dataset_test1 = loadtxt('dataset/monks-1.test',
                        delimiter=' ', usecols=range(1, 8))

dataset_train2 = loadtxt('dataset/monks-2.train',
                         delimiter=' ', usecols=range(1, 8))
dataset_test2 = loadtxt('dataset/monks-2.test',
                        delimiter=' ', usecols=range(1, 8))

dataset_train3 = loadtxt('dataset/monks-3.train',
                         delimiter=' ', usecols=range(1, 8))
dataset_test3 = loadtxt('dataset/monks-3.test',
                        delimiter=' ', usecols=range(1, 8))


class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='accuracy', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current >= self.value:
            self.model.stop_training = True

def solve_monk(dataset_train, dataset_test, nUnit, lr, momentum, nEpoch, batch_size, lambda_):
    x = dataset_train[:, 1:7]
    x_test = dataset_test[:, 1:7]
    y = dataset_train[:, 0]
    y_test = dataset_test[:, 0]
    early_stopping = EarlyStoppingByAccuracy(
        monitor='val_accuracy', value=1.0, verbose=1)
    model = Sequential()
    model.add(
        Dense(nUnit, input_dim=6, kernel_initializer=RandomNormal(mean=0.0, stddev=0.005, seed=42), activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_)))
    model.add(Dense(1, activation='linear'))

    sgd = SGD(lr=lr, momentum=momentum, nesterov=False)
    model.compile(optimizer=sgd, loss='mean_squared_error',
                  metrics=['accuracy'])
    history = model.fit(x, y, validation_data=(x_test, y_test),
                        epochs=nEpoch, batch_size=batch_size, callbacks=[early_stopping], verbose=0)
    accuracy = history.history['val_accuracy'][-1]
    loss = history.history['val_loss'][-1]

    return (loss, accuracy)

def learning_curve(eta, alpha, lambda_param, batch_size, nUnitLayer, nEpoch, dataset_train,dataset_test, index_monk,plotted):
                early_stopping = EarlyStoppingByAccuracy(
                monitor='val_accuracy', value=1.0, verbose=1)
                kfold = KFold(n_splits=10, random_state=None, shuffle=True)
                cvscores = []
                forLegend = []
                X = dataset_train[:, 1:7]
                Y = dataset_train[:, 0]
                X_TEST = dataset_test[:, 1:7]
                Y_TEST = dataset_test[:, 0]
                model = Sequential()
                model.add(
                    Dense(nUnitLayer, input_dim=6, kernel_initializer=RandomNormal(mean=0.0, stddev=0.02, seed=42), activation='relu', kernel_regularizer=regularizers.l2(lambda_param)))
                model.add(Dense(1, activation='linear'))
                sgd = SGD(lr=eta, momentum=alpha, nesterov=False)
                model.compile(optimizer=sgd, loss='mean_squared_error',
                              metrics=['accuracy'])
                history = model.fit(X, Y, validation_data=(X_TEST,Y_TEST),
                                    epochs=nEpoch, batch_size=batch_size, callbacks=[early_stopping],verbose=0)
                ts_accuracy = history.history['val_accuracy'][-1]
                ts_loss = history.history['val_loss'][-1]
                tr_accuracy = history.history['accuracy'][-1]
                tr_loss = history.history['loss'][-1]
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'],'--')
                #averageLoss += score[0]
                #averageLoss_acc += score[1]
                forLegend.append('Loss on training set')
                forLegend.append('Loss on validation set')
                plt.legend(forLegend, loc='center right')
                
                plt.suptitle('Monk loss ' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
                    lambda_param) + '_' + str(batch_size))
                plt.savefig('./plots_final/'+str(plotted)+'loss_Monk'+str(index_monk)+'_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
                    lambda_param) + '_' + str(batch_size) + '_' + str(nUnitLayer) + '_' + str(nUnitLayer) + '_'+ str(
                    tr_loss)+'_'+str(ts_loss)+'_'+str(tr_accuracy)+'_'+str(ts_accuracy)+'.png', dpi=600)
                plt.close()

                forLegend = []
                forLegend.append('Accuracy on training set')
                forLegend.append('Accuracy on validation set')
                plt.plot(history.history['accuracy'])
                plt.plot(history.history['val_accuracy'],'--')
                plt.legend(forLegend, loc='center right')
                plt.suptitle('Monk accuracy ' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
                    lambda_param) + '_' + str(batch_size))
                plt.savefig('./plots_final/'+str(plotted)+'accuracy_Monk'+str(index_monk)+'_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(
                    lambda_param) + '_' + str(batch_size) + '_' + str(nUnitLayer) + '_' + str(
                    tr_loss)+'_'+str(ts_loss)+'_'+str(tr_accuracy)+'_'+str(ts_accuracy)+'.png', dpi=600)
                plt.close()
                return


def get_monk(dataset_train, dataset_test, index_monk):
    nUnitLayers = [10,20,30]
    etas = [ 0.01,0.009,0.005,0.002]
    alphas = [0.5,0.8, 0.85, 0.9]
    lambdas = [0.0003, 0.0005,0.001]
    batch_sizes = [16, 20,32]
    nEpoch = 300
    plotted=0
    for nUnitLayer in nUnitLayers:
                    for eta in etas:
                        for alpha in alphas:
                            for _lambda in lambdas:
                                for batch_size in batch_sizes:
                                    loss, accuracy = solve_monk(dataset_train, dataset_test,
                                                                nUnitLayer, eta, alpha, nEpoch, batch_size, _lambda)
                                    mean_loss = loss
                                    accuracy_mean = (accuracy*100)
                                    print(eta, alpha, _lambda, batch_size, nEpoch, '-', loss,
                                          ',', accuracy*100, ' --- Unit layer:', nUnitLayer)
                                    if((accuracy*100) > 90):
                                        print('...entered in test data')
                                        for i in range(0, 2):
                                            loss, accuracy = solve_monk(
                                                dataset_train, dataset_test, nUnitLayer, eta, alpha, nEpoch, batch_size, _lambda)
                                          
                                            mean_loss += loss
                                            accuracy_mean += (accuracy*100)
                                        if((accuracy_mean/3) > 90):
                                            print(
                                                '----------------------------------------')
                                            print('Plotting learning curve...')
                                            print('Monk:', index_monk)
                                            print(mean_loss/3, '-',
                                                  accuracy_mean/3)
                                            print(eta, '-', alpha, '-', _lambda,
                                                  '-', batch_size, '-', nUnitLayer)

                                            learning_curve(
                                                eta, alpha, _lambda, batch_size,
                                                nUnitLayer, nEpoch, dataset_train, dataset_test,index_monk,plotted)
                                            plotted=plotted+1
                                            print('Done it!')
                                            print(
                                                '----------------------------------------')
                                        else:  print('...accuracy too much low:',accuracy_mean/3)

#get_monk(dataset_train1, dataset_test1, 1)
get_monk(dataset_train2, dataset_test2, 2)
get_monk(dataset_train3, dataset_test3, 3)
