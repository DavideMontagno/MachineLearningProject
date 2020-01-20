from libsvm.svmutil import *
import numpy



dataset_tr = numpy.genfromtxt(
    '../project/ML-CUP19-TR.csv', delimiter=',', dtype=numpy.float64)
    
x = dataset_tr[:, 1:-2]
y = dataset_tr[:, -2:]

m = svm_train(y[:200], x[:200], '-c 4')
p_label, p_acc, p_val = svm_predict(y[200:], x[200:], m)

#format data for read csv
'''
target id_1:value id_2:value id_3:value ... id_n:value #1 line
'''


#cross-validation
'''
Function: void svm_cross_validation(const struct svm_problem *prob,
	const struct svm_parameter *param, int nr_fold, double *target);

    This function conducts cross validation. Data are separated to
    nr_fold folds. Under given parameters, sequentially each fold is
    validated using the model from training the remaining. Predicted
    labels (of all prob's instances) in 

'''


#predict value
'''
Function: double svm_predict_values(const svm_model *model,
				    const svm_node *x, double* dec_values)

    This function gives decision values on a test vector x given a
    model, and return the predicted label (classification) or
    the function value (regression).

    For a classification model with nr_class classes, this function
    gives nr_class*(nr_class-1)/2 decision values in the array
    dec_values, where nr_class can be obtained from the function
    svm_get_nr_class. The order is label[0] vs. label[1], ...,
    label[0] vs. label[nr_class-1], label[1] vs. label[2], ...,
    label[nr_class-2] vs. label[nr_class-1], where label can be
    obtained from the function svm_get_labels. The returned value is
    the predicted class for x. Note that when nr_class = 1, this
    function does not give any decision value.

    For a regression model, dec_values[0] and the returned value are
    both the function value of x calculated using the model. For a
    one-class model, dec_values[0] is the decision value of x, while
    the returned value is +1/-1.
'''




