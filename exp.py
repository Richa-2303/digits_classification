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
import numpy as np
from joblib import load
import itertools
from utils import get_x_and_y,pre_process,split_train_dev_test,predict_and_eval,tune_hparams,train_model
from skimage.transform import resize

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
image_sizes=[4,6,8]
list_of_all_param_combination = [dict(zip(param_type, values)) 
                                 for values in itertools.product(gamma_ranges,
                                                                  C_ranges)]
print('The number of total samples in the dataset',X.shape[0])
print('height of the original images in dataset',X[0].shape[0],'width of the images in dataset',X[0].shape[1])
    
#2. split the data in train, test and dev set
test_size=0.2
dev_size=0.1
for image_size in image_sizes:
    X_resized=np.array([resize(x, (image_size,image_size)) for x in X])
    print('height of the rescaled images in dataset',X_resized[0].shape[0],
          'width of the rescaled images in dataset',X_resized[0].shape[1])
    X_train, X_test, y_train, y_test, X_dev, y_dev=split_train_dev_test(X_resized,y,test_size=0.2,dev_size=0.1)
    #3. preproces the data
    X_train=pre_process(X_train)
    X_test=pre_process(X_test)
    X_dev=pre_process(X_dev)
    #4. Hyper-parameter tuning
    best_hparams, best_model_path, best_accuracy=tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination)
    #5. Get the predictions on dev
    best_model=load(best_model_path)
    test_accuracy = predict_and_eval(best_model,X_test,y_test)
    print('best_hparams: ',best_hparams)
    print(f'image_size = {image_size}x{image_size} test_size = {test_size} dev_size = {dev_size} train_size {1-(test_size+dev_size)} train_acc={best_accuracy["train_accuracy"]} dev_acc={best_accuracy["dev_accuracy"]} test_acc=={test_accuracy} ')