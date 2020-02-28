
import numpy as np
from flask import Flask, request, jsonify,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/clustered',method=['POST])
def predict():
    features=[x for x in request.form.values()]
    final_fea=np.array(features)
    prediction=model.predict(final_fea)
    
    return render_template('index.html',prediction_text='cluster{}'.format(prediction))
    
'''@app.route('/predict',method=['POST])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final_features=[np.array(int_features)]
    prediction=model.predict(final_features)
    
    
    output=round(prediction[0],2)
    
    return render_template('index.html',prediction_text='shortlisting{}'.format(output))'''

if __name__=='main':
    app.run(debug=True)
