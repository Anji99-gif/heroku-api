# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:01:53 2020

@author: hp
"""
import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/clustering',method=['POST])
def predict():
   # data= pd.read_csv('dataset.csv')
   # data=prepross(data)
   # data=pca(data)
    clusterer = KMeans(n_clusters=4,random_state=42,n_init=10).fit(d)
    centers = clusterer.cluster_centers_
    labels= clusterer.predict(d)
    return jsonify(labels)

'''@app.route('/predict',method=['POST])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    
    output=round(prediction[0],2)
    
    return render_template('index.html',prediction_text='shortlisting{}'.format(output))'''

if __name__=='main':
    app.run(debug=True)
