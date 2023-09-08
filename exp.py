"""
================================
Recognizing hand-written digits
================================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

# Standard scientific Python imports
#import matplotlib.pyplot as plt
# Import performance metrics
from sklearn import metrics
import itertools
from utils import get_x_and_y,pre_process,split_train_dev_test,predict_and_eval,tune_hparams,train_model


#1. get data and its labels
X,y=get_x_and_y()
#2. Hyper-parameter tuning
best_accuracy={'train_accuracy':-1,'dev_accuracy':-1}
best_model=None
best_hparams={}
test_size_range=[0.1, 0.2, 0.3] 
dev_size_range=test_size_range
param_type=['gamma','C']
gamma_ranges=[0.001,0.01,0.1,1,10,100]
C_ranges=[0.1,1,2,5,10]
list_of_all_param_combination = [dict(zip(param_type, values)) for values in itertools.product(gamma_ranges, C_ranges)]

#2. split the data in train, test and dev set
for test_size in test_size_range:
    for dev_size in dev_size_range:
        X_train, X_test, y_train, y_test, X_dev, y_dev=split_train_dev_test(X,y,test_size=test_size,dev_size=dev_size)
        #3. preproces the data
        X_train=pre_process(X_train)
        X_test=pre_process(X_test)
        X_dev=pre_process(X_dev)
        #4. Hyper-parameter tuning
        best_hparams, best_model, best_accuracy=tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination)
        #5. Get the predictions on dev
        test_accuracy = predict_and_eval(best_model,X_test,y_test)
        print('best_hparams: ',best_hparams)
        print(f'test_size = {test_size} dev_size = {dev_size} train_size {1-(test_size+dev_size)} train_acc={best_accuracy["train_accuracy"]} dev_acc={best_accuracy["dev_accuracy"]} test_acc=={test_accuracy} ')