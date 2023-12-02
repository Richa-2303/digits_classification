# from flask import Flask
from flask import Flask, request, jsonify
import numpy as np
import os
from joblib import load
from utils import get_x_and_y,pre_process,split_train_dev_test,predict_and_eval,tune_hparams,train_model,get_list_of_all_param_combination
app = Flask(__name__)

def load_model(model_type):
    models=os.listdir('models/')
    model_filename=[i for i in models if (model_type in i)and  ('best' in i)]
    model=load('models/+'str(model_filename))
    return model
@app.route('/compare_digits', methods=['POST'])
def compare_digits():
    # Get images from the request
    data = request.get_json()

    # Extract pixel arrays for the images from JSON data
    image1_pixels = data['image1']
    image2_pixels = data['image2']

    # Convert pixel arrays to numpy arrays for processing
    image1_np = np.array(image1_pixels)
    image2_np = np.array(image2_pixels)

    image1=pre_process(image1_np)
    image2=pre_process(image2_np)

    best_model=load('models/best_model_svmgamma-0.001_C-1.joblib')
    predicted1 = best_model.predict([np.array(image1[:,0])])
    predicted2 = best_model.predict([np.array(image2[:,0])])
    # Process images using your deep learning model and get the result
    result ='false'
    if predicted1==predicted2:result='true'

    # Return the result as JSON
    return jsonify({"result": result})

@app.route('/predict/<model_type>', methods=['POST'])
def predict(model_type):
    # Get images from the request
    data = request.get_json()

    # Extract pixel arrays for the images from JSON data
    image1_pixels = data['image1']
    
    # Convert pixel arrays to numpy arrays for processing
    image1_np = np.array(image1_pixels,dtype='float')
    
    image1=pre_process(image1_np)
    result={}
    best_model=load(model_type)
    predicted1 = best_model.predict([np.array(image1[:,0])])
    # Process images using your deep learning model and get the result
    result[model_type] =str(predicted1)
    # Return the result as JSON
    return jsonify({"result": result})

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/", methods=["POST"])
def hello_world_post():    
    return {"op" : "Hello, World POST " + request.json["suffix"]}
    
if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 5000, debug = True) 