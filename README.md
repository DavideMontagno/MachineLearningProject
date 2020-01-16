# MachineLearningProject

In order to start: 
 
- Create venv 
- Install numpy temsorflow keras

```

mkdir venv
python3 -m venv ./venv

source venv/bin/activate

pip install --upgrade pip

pip install numpy

pip install tensorflow

pip install keras

pip install scikit-learn

for torch 

pip3 install torch torchvision 


```

## TODO: 

Make explicit in the Model 1 - Keras Neural Network these hyperparameters: 
 
- Starting values: winitial
- Learning rate: eta
- Momentum: alpha (nesterov only for batch)
- Number of epochs: nEpoch
- Penalty term: lambda
- Number of layers: nLayer
- Number of unit for each layer

METRICHE

```
def rmse(y_true, y_pred):
    from keras import backend
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))
```
Root mean squared error: 0 indica la perfezione. Molto sensibile agli outliers, perché è proporzionale all'errore al quadrato. 

```
def mse(y_true, y_pred):
    from keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)
```
Mean square error: 0 indica perfezione, molto sensibile agli outliers. MAE forse potrebbe essere più interpretabile. 

```
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )
```
R quadro: 1 indica la perfezione. 


## POSSIBLE MODELS:

- NN in Pytorch
- SVM
- CNN
- RNN
- Random Forest
- Bayesian model
- Clustering?



