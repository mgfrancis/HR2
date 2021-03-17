import json
import pandas as pd
import pymysql
from sqlalchemy import create_engine
import sqlalchemy
from flask import Flask, request, render_template, jsonify
import os
import pickle
import numpy as np
from keras import model_from_json 

model = pickle.load(open('model.pkl', 'rb')) # loading the trained model


# Initialize Flask application
app = Flask(__name__)

# Set up your default route


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    # retrieving values from form
    init_features = [float(x) for x in request.form.values()]
    final_features = [np.array(init_features)]

    prediction = model.predict(final_features) # making prediction


    return render_template('index.html', prediction_text='Attrition Likelihood: {}'.format(prediction)) # rendering the predicted result



# @app.route("/result", methods = ['POST', 'GET'])
# def result():
#     if request.method=='POST':
# #geting data from html form
#         age=request.form['Age']
#         distance=request.form['DistanceFromHome']
#         time=request.form['sOverTime']
#         joblev=request.form['JobLevel']
#         jobsat=request.form['JobSatisfaction']
#         inc=request.form['MonthlyIncome']
#         hours=request.form['StandardHours']
#         years=request.form['TotalWorkingYears']
#         balance=request.form['WorkLifeBalance']
#         currole=request.form['YearsInCurrentRole']
#         promotion=request.form['YearsSinceLastPromotion']
#         curmgr=request.form['YearsWithCurrManager']
       
# # after geting data appending in a list
#         lst=list()
#         lst.append((age))
#         lst.append((distance))
#         lst.append((time))
#         lst.append((joblev))
#         lst.append((jobsat))
#         lst.append((inc))
#         lst.append((hours))
#         lst.append((years))
#         lst.append((balance))
#         lst.append((currole))
#         lst.append((promotion))
#         lst.append((curmgr))
# # converting list into 2 D numpy array
#         ans=model.predict([np.array(lst,dtype='int64')])
#         result=ans[0]
#         return render_template("index.html",result=result)

#     else:
#         return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
