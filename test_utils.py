import itertools
from utils import get_x_and_y,split_train_dev_test,tune_hparams,pre_process
import os

def create_dummy_hyperparameters():
    param_type=['gamma','C']
    gamma_ranges=[0.001,0.01,0.1,1,10,100]
    C_ranges=[0.1,1,2,5,10]
    list_of_all_param_combination = [dict(zip(param_type, values)) for values in itertools.product(gamma_ranges, C_ranges)]
    return list_of_all_param_combination,len(gamma_ranges),len(C_ranges)

def test_for_hyper_param_combinations_count():
    list_of_all_param_combination,len_gamma_ranges,len_C_ranges = create_dummy_hyperparameters()
    assert len(list_of_all_param_combination) == len_gamma_ranges*len_C_ranges

def create_dummy_data(X,y,size_limit):
    return pre_process(X[:size_limit,:,:]),y[:size_limit]

def test_model_saving():
    X,y=get_x_and_y()
    X_train,y_train=create_dummy_data(X,y,size_limit=100)
    X_dev,y_dev=create_dummy_data(X,y,size_limit=50)
    
    list_of_all_param_combination,_,_ = create_dummy_hyperparameters()
    _,best_model_path,_=tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination)
    assert os.path.exists(best_model_path)

def test_data_splitting():
    X,y=get_x_and_y()
    X,y=create_dummy_data(X,y,size_limit=100)
    test_size=0.1
    dev_size=0.6
    X_train, X_test, y_train, y_test, X_dev, y_dev=split_train_dev_test(X,y,test_size=test_size,dev_size=dev_size)
    assert(len(X_train)==30)
    assert(len(X_train)==10)
    assert(len(X_train)==60)
    

def test_for_hyper_param_combinations_values():
    param_type=['gamma','C']
    gamma_ranges=[0.001,0.01,0.1,1,10,100]
    C_ranges=[0.1,1,2,5,10]
    list_of_all_param_combination = [dict(zip(param_type, values)) for values in itertools.product(gamma_ranges, C_ranges)]
    expected_param_combo1={'gamma':0.001,'C':1}
    expected_param_combo2={'gamma':0.01,'C':1}
    assert (expected_param_combo1 in list_of_all_param_combination) and (expected_param_combo2 in list_of_all_param_combination)
