from Model_2_PytorchNN.model_2 import best_model2
from Model_1_KerasNN.model_1 import best_model1
import matplotlib.pyplot as plt
#print('Starting model 1')
#model_1.cross_validation1()


def numbers_to_Models(argument): 
    switcher = { 
        1: "Model Keras", 
        2: "Model Pytorch", 
        3: "Model SVR", 
    } 
    return switcher.get(argument, "nothing") 



def create_plot(y_pred,model):
    s = 50
    a = 0.4
    plt.figure()
    plt.scatter(y_pred[:,0],y_pred[:,1], c="cornflowerblue", s=s, alpha=a)
    plt.xlabel("Target 1")
    plt.ylabel("Target 2")
    plt.title(numbers_to_Models(model))
    labels = ['Data point predicted by model: '+str(model)]
    plt.legend(labels)
   

    plt.savefig('./Prediction_model'+str(model)+'.png', dpi=1000)
    #plt.show()  
                        

create_plot(best_model1(False),1)
create_plot(best_model2(False),2)


