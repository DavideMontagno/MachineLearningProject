from numpy import loadtxt
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
import matplotlib.pyplot as plt

dataset_train1 = loadtxt('./dataset/monks-1.train',
                         delimiter=' ', usecols=range(1, 8))
dataset_test1 = loadtxt('./dataset/monks-1.test',
                        delimiter=' ', usecols=range(1, 8))

dataset_train2 = loadtxt('./dataset/monks-2.train',
                         delimiter=' ', usecols=range(1, 8))
dataset_test2 = loadtxt('./dataset/monks-2.test',
                        delimiter=' ', usecols=range(1, 8))

dataset_train3 = loadtxt('./dataset/monks-3.train',
                         delimiter=' ', usecols=range(1, 8))
dataset_test3 = loadtxt('./dataset/monks-3.test',
                        delimiter=' ', usecols=range(1, 8))


class EarlyStoppingByAccuracy(Callback):
    def __init__(self, monitor='accuracy', value=1.0, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)

        if current >= self.value:
            self.model.stop_training = True


early_stopping = EarlyStoppingByAccuracy(
    monitor='acc', value=1.0, verbose=1)


def monk_solve_plot(train, plotted, eta, alpha, batch_size, nUnit, nEpoch, lambda_param):
    x = train[:, 1:7]
    y = train[:, 0]

    model = Sequential()
    model.add(Dense(nUnit, input_dim=6, kernel_initializer='glorot_normal',
                    activation='sigmoid',  kernel_regularizer=l2(lambda_param)))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(lr=eta, momentum=alpha, nesterov=False)
    model.compile(optimizer=sgd, loss='mean_squared_error',
                  metrics=['accuracy'])
    history = model.fit(x, y, validation_split=0, epochs=nEpoch, callbacks=[
        early_stopping], batch_size=batch_size, verbose=0)
    plt.plot(history.history['loss'])
    plt.plot(history.history['acc'], '--')
    plt.legend(['Loss on training set', 'Accuracy on validation set'],
               loc='center right')
    plt.savefig('./' + str(plotted)+'_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(batch_size) +
                '_' + str(nUnit) + '_' + str(history.history['loss'][-1])+'_'+str(history.history['acc'][-1])+'.png', dpi=600)
    plt.close()
    print(history.history['acc'][-1])


#monk_solve_plot(dataset_train1, 1, 0.2, 0.75, 25, 5, 200,0)
monk_solve_plot(dataset_train2, 2, 0.88, 0.55, 169, 8, 20000, 0)
#monk_solve_plot(dataset_train3, 3, 0.3, 0.8, 25, 8, 400,0)
#monk_solve_plot(dataset_train3, 3, 0.4, 0.75, 25, 8, 300,0.0001)
