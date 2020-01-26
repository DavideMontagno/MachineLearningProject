from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import regularizers
from keras.initializers import RandomNormal

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


def solve_monk(dataset_tr, dataset_ts, nUnit, lr, momntum, nEpoch, batch_size, lambda_):
    x = dataset_tr[:, 1:7]
    x_test = dataset_ts[:, 1:7]
    y = dataset_tr[:, 0]
    y_test = dataset_ts[:, 0]

    model = Sequential()
    model.add(
        Dense(nUnit, input_dim=6, kernel_initializer=RandomNormal(mean=0.0, stddev=0.005, seed=42), activation='sigmoid', kernel_regularizer=regularizers.l2(lambda_)))
    model.add(Dense(1, activation='linear'))

    sgd = SGD(lr=lr, momentum=momntum, nesterov=False)
    model.compile(optimizer=sgd, loss='mean_squared_error',
                  metrics=['accuracy'])
    history = model.fit(x, y, validation_data=(x_test, y_test),
                        epochs=nEpoch, batch_size=batch_size, verbose=1)
    accuracy = history.history['val_acc'][-1]
    loss = history.history['val_loss'][-1]
    print('Loss: %.2f' % loss, 'Accuracy: %.2f' % (accuracy*100))


solve_monk(dataset_train1, dataset_test1, 30, 0.035, 0.85, 100, 16, 0.001)

#solve_monk(dataset_train1, dataset_test1, 15, 0.035, 0.8, 220, 10, 0.0005)

#solve_monk(dataset_train1, dataset_test1, 15, 0.035, 0.8, 220, 10, 0.0005)
