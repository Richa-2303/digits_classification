import itertools
from utils import get_x_and_y,split_train_dev_test,tune_hparams,pre_process
import os
import sys

model_type_var=sys.argv[2]
print("model_type_var",model_type_var)


def create_dummy_hyperparameters(model_type):
    if model_type=='svm':
        param_type=['gamma','C']
        gamma_ranges=[0.001,0.01,0.1,1,10,100]
        C_ranges=[0.1,1,2,5,10]
        list_of_all_param_combination = [dict(zip(param_type, values)) for values in itertools.product(gamma_ranges, C_ranges)]
        all_params_combination_count=len(gamma_ranges)*len(C_ranges)
    if model_type=='descision_trees':
        param_type=['max_depth']
        max_depth_ranges=[2,10,100]
        list_of_all_param_combination = [dict(zip(param_type, values)) 
                                        for values in itertools.product(max_depth_ranges)]
        all_params_combination_count=len(max_depth_ranges)
    return list_of_all_param_combination,all_params_combination_count

def test_for_hyper_param_combinations_count():
    list_of_all_param_combination,all_params_combination_count = create_dummy_hyperparameters(model_type=model_type_var)
    assert len(list_of_all_param_combination) == all_params_combination_count

def create_dummy_data(X,y,size_limit):
    return pre_process(X[:size_limit,:,:]),y[:size_limit]

def test_model_saving():
    X,y=get_x_and_y()
    X_train,y_train=create_dummy_data(X,y,size_limit=100)
    X_dev,y_dev=create_dummy_data(X,y,size_limit=50)
    
    list_of_all_param_combination,_= create_dummy_hyperparameters(model_type=model_type_var)
    _,best_model_path,_=tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination,model_type_var)
    assert os.path.exists(best_model_path)

def test_data_splitting():
    X,y=get_x_and_y()
    X,y=create_dummy_data(X,y,size_limit=100)
    test_size=0.1
    dev_size=0.6
    print(len(X)==100)
    X_train, X_test, y_train, y_test, X_dev, y_dev=split_train_dev_test(X,y,test_size=test_size,dev_size=dev_size)
    assert(len(X_train)==30)
    assert(len(X_test)==10)
    assert(len(X_dev)==60)
    

def test_for_hyper_param_combinations_values():
    if model_type_var=="svm":
        list_of_all_param_combination,_= create_dummy_hyperparameters(model_type=model_type_var)
        expected_param_combo1={'gamma':0.001,'C':1}
        expected_param_combo2={'gamma':0.01,'C':1}
        assert (expected_param_combo1 in list_of_all_param_combination) and (expected_param_combo2 in list_of_all_param_combination)
