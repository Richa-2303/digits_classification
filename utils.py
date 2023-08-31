
# Import datasets, classifiers 
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
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
        X, y, test_size=test_size, shuffle=False
    )
    # Split train data into 90% train and 10% dev subsets
    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train, y_train, test_size=dev_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test, X_dev, y_dev

def predict_and_eval(model,X_test,y_test):
    # Predict the value of the digit on the test subset
    predicted = model.predict(X_test)
    return predicted

def train_model(classifier,X_train, y_train,model_params):
    # Create a classifier: a support vector classifier
    if classifier=='svm':
        model = svm.SVC(**model_params)#
    # Learn the digits on the train subset
    model.fit(X_train, y_train)
    return model