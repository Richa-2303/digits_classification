
# Import datasets, classifiers 
from sklearn import datasets, svm
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import itertools
def get_x_and_y():
    digits = datasets.load_digits()
    return digits.images, digits.target

def pre_process(data):
    # flatten the images
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    return data
    
def split_train_dev_test(X,y,test_size,dev_size):
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(test_size+dev_size), shuffle=True
    )
    
    # Split train data into 90% train and 10% dev subsets
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_test, y_test, test_size=1-(dev_size/(test_size+dev_size)), shuffle=False
    )
    return X_train, X_test, y_train, y_test, X_dev, y_dev

def get_list_of_all_param_combination(model_type):
    if model_type=='svm':
        param_type=['gamma','C']
        gamma_ranges=[0.001,0.01,0.1,1,10,100]
        C_ranges=[0.1,1,2,5,10]
        list_of_all_param_combination = [dict(zip(param_type, values)) 
                                        for values in itertools.product(gamma_ranges,
                                                                        C_ranges)]
    if model_type=='descision_trees':
        param_type=['max_depth']
        max_depth_ranges=[5,10,15,20,30,40]
        list_of_all_param_combination = [dict(zip(param_type, values)) 
                                        for values in itertools.product(max_depth_ranges)]
        #print('list_of_all_param_combination',list_of_all_param_combination)
    return list_of_all_param_combination
def predict_and_eval(model,X,y):
    # Predict the value of the digit on the test subset
    predicted = model.predict(X)
    accuracy=accuracy_score(y, predicted)
    
    return accuracy

def train_model(classifier,X_train, y_train,model_params):
    # Create a classifier: a support vector classifier
    if classifier=='svm':
        model = svm.SVC(**model_params)#
    if classifier=='descision_trees':
        model = DecisionTreeClassifier(**model_params)
    # Learn the digits on the train subset
    model.fit(X_train, y_train)
    return model

def tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination,model_type): 
    best_dev_acc_so_far=-1 
    best_model_path=''
    for all_param in list_of_all_param_combination:
        model=train_model(model_type,X_train, y_train,all_param)

        #Get the accuracy on dev and test
        train_accuracy = predict_and_eval(model,X_train, y_train)
        dev_accuracy = predict_and_eval(model,X_dev, y_dev)
        if dev_accuracy>best_dev_acc_so_far:
            best_dev_acc_so_far=dev_accuracy
            best_hparams=all_param
            best_model=model
            best_model_path="./models/best_model_"+model_type+"_".join(["{}-{}".format(k,v) for k,v in best_hparams.items()])+".joblib"
            dev_accuracy = predict_and_eval(model,X_dev, y_dev)
            best_accuracy={'train_accuracy':train_accuracy,'dev_accuracy':dev_accuracy}
    dump(best_model,best_model_path)
    return best_hparams, best_model_path, best_accuracy
