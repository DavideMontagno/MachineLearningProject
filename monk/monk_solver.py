from numpy import loadtxt
from keras.models import Sequential
from keras.callbacks import Callback
from keras.layers import Dense
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
    def __init__(self, monitor='accuracy', value=0.98, verbose=0):
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


def monk_solve_plot(train, plotted, eta, alpha, batch_size, nUnit, nEpoch):
    x = train[:, 1:7]
    y = train[:, 0]

    model = Sequential()
    model.add(Dense(nUnit, input_dim=6, kernel_initializer='random_normal',
                    activation='sigmoid'))
    model.add(Dense(1, activation='linear'))

    sgd = SGD(lr=eta, momentum=alpha, nesterov=False)
    model.compile(optimizer=sgd, loss='mean_squared_error',
                  metrics=['accuracy'])
    history = model.fit(x, y, validation_split=0, epochs=nEpoch, callbacks=[
        early_stopping], batch_size=batch_size, verbose=1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['acc'], '--')
    plt.legend(['Loss on training set', 'Loss on validation set'],
               loc='center right')
    plt.savefig('./' + str(plotted)+'_learning_curve_' + str(eta) + '_' + str(alpha) + '_' + str(nEpoch) + '_' + str(batch_size) +
                '_' + str(nUnit) + '_' + str(history.history['loss'][-1])+'_'+str(history.history['acc'][-1])+'.png', dpi=600)
    plt.close()


#monk_solve_plot(dataset_test1, 1, 0.05, 0.8, 20, 8, 300)
monk_solve_plot(dataset_test2, 2, 0.05, 0.75, 30, 10, 300)
#monk_solve_plot(dataset_test3, 3, 0.05, 0.85, 15, 15, 225)
