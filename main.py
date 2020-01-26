from Model_2_PytorchNN.model_2 import best_model2
from Model_1_KerasNN.model_1 import best_model1
from Model_3_ScikitSVM.model_3 import best_model3
import matplotlib.pyplot as plt
import time


def numbers_to_Models(argument): 
    switcher = { 
        1: "Model Keras", 
        2: "Model Pytorch", 
        3: "Model SVR", 
    } 
    return switcher.get(argument, "nothing") 



def create_plot(y_pred,model_index):
    s = 50
    a = 0.4
    plt.figure()
    plt.scatter(y_pred[:,0],y_pred[:,1], c="cornflowerblue", s=s, alpha=a)
    plt.xlabel("Target 1")
    plt.ylabel("Target 2")
    plt.title(numbers_to_Models(model_index))
    labels = ['Data point predicted by model: '+str(model_index)]
    plt.legend(labels)
    if(model_index==1): plt.savefig('./plots_final/Keras_BSvisualization_Model_' + str(model_index) +'.png', dpi=500)
    if(model_index==2):
        plt.savefig('./plots_final/Pytorch_BSvisualization_Model_' + str(model_index) +'.png', dpi=500)
    if(model_index==3): plt.savefig('./plots_final/SVM_BSvisualization_Model_' + str(model_index) +'.png', dpi=500)
    plt.close()
   
   
print('Using Neural Network in Keras...')
start = time.time()
create_plot(best_model1(False),1)  
end = time.time()
print('Keras:',end-start)
print('Prediction done!')   
print('Using Neural Network in Pytorch...')
start = time.time()
create_plot(best_model2(False),2)
end = time.time()
print('Pytorch:',end-start)
print('Prediction done!')  
print('Using Support Vectors Machine..') 
start = time.time()
create_plot(best_model3(False),3)
end = time.time()
print('SVR:',end-start)
print('Prediction done!')  


