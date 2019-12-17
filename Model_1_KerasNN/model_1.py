
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

dataset_tr = numpy.loadtxt('../project/ML-CUP19-TR.csv', delimiter=',', dtype = numpy.float64)
dataset_ts = numpy.loadtxt('../project/ML-CUP19-TS.csv', delimiter=',', dtype = numpy.float64)

x_tr = dataset_tr[:, 1:-2]
x_ts = dataset_ts[:, 1:-2]
y1_tr = dataset_tr[:, -1]
y1_ts = dataset_ts[:, -1]
y2_tr = dataset_tr[:, -2]
y2_ts = dataset_ts[:, -2]

'''

w_initial
eta
alpha (nesterov only for batch)
nEpoch
lambda
nLayer

model = Sequential()
model.add(Dense(30, input_dim=6, kernel_initializer='random_normal', activation='sigmoid'))
model.add(Dense(1, activation='linear'))

sgd = SGD(learning_rate=0.05, momentum=0.8,nesterov=False)
model.compile(optimizer=sgd, loss='mean_squared_error',metrics=['accuracy'])
model.fit(x,y,validation_data=(x_test,y_test), epochs=150, batch_size=25,verbose=1)

#loss, accuracy = model.evaluate(dataset_train1[:, 1:7], dataset_train1[:, 0])
#print('Loss: %.2f' % loss , 'Accuracy: %.2f' %(accuracy*100))
'''