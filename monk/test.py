from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

dataset_train1 = loadtxt('dataset/monks-1.train', delimiter=' ', usecols=range(1, 8))
dataset_test1 = loadtxt('dataset/monks-1.test', delimiter=' ', usecols=range(1, 8))

dataset_train2 = loadtxt('dataset/monks-2.train', delimiter=' ', usecols=range(1, 8))
dataset_test2 = loadtxt('dataset/monks-2.test', delimiter=' ', usecols=range(1, 8))

dataset_train3 = loadtxt('dataset/monks-3.train', delimiter=' ', usecols=range(1, 8))
dataset_test3 = loadtxt('dataset/monks-3.test', delimiter=' ', usecols=range(1, 8))

x = dataset_test1[:, 1:7]
x_test = dataset_train1[:, 1:7]
y = dataset_test1[:, 0]
y_test = dataset_train1[:, 0]

model = Sequential()
model.add(Dense(30, input_dim=6, kernel_initializer='random_normal', activation='sigmoid'))
model.add(Dense(1, activation='linear'))

sgd = SGD(learning_rate=0.05, momentum=0.8,nesterov=False)
model.compile(optimizer=sgd, loss='mean_squared_error',metrics=['accuracy'])
model.fit(x,y,validation_data=(x_test,y_test), epochs=150, batch_size=25,verbose=1)

#loss, accuracy = model.evaluate(dataset_train1[:, 1:7], dataset_train1[:, 0])
#print('Loss: %.2f' % loss , 'Accuracy: %.2f' %(accuracy*100))
