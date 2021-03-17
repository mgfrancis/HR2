# Load libraries
import flask
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
import keras
from keras.models import load_model
from keras.models import Sequential


# instantiate flask 
app = flask.Flask(__name__)

# we need to redefine our metric function in order 
# to use it when loading the model 
# def auc(y_true, y_pred):
#     auc = tf.metrics.auc(y_true, y_pred)[1]
#     keras.backend.get_session().run(tf.local_variables_initializer())
#     return auc

# load the model, and pass in the custom metric function
#global graph
#graph = tf.get_default_graph()
#model = load_model('neural_model.h5', custom_objects={'auc': auc})

model = load_model('neural_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

# define a predict function as an endpoint 
@app.route("/predict", methods=["GET","POST"])
def predict():

    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction


    return render_template('index.html', prediction_text='Attrition Likelihood: {}'.format(prediction)) # rendering the predicted result

if __name__ == "__main__":
    app.run(debug=True)



#     data = {"success": False}

#     params = flask.request.json
#     if (params == None):
#         params = flask.request.args

#     # if parameters are found, return a prediction
#     if (params != None):
#         x=pd.DataFrame.from_dict(params, orient='index').transpose()
#         with graph.as_default():
#             data["prediction"] = str(model.predict(x)[0][0])
#             data["success"] = True

#     # return a response in json format 
#     return flask.jsonify(data)    

# # start the flask app, allow remote connections 
# app.run(host='0.0.0.0')