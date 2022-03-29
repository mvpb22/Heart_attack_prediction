from flask import Flask, render_template, request
#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib 
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('original.php')


@app.route("/predict", methods=['GET','POST'])
def predict():
   # if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trestbps = float(request.form['trestbps'])
        chol = float(request.form['chol'])
        fbs= float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalach = float(request.form['thalach'])
        exang = float(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = float(request.form['slope'])
        ca = float(request.form['ca'])
        thal = float(request.form['thal'])

        filename = 'hear_attack.pkl'
        model = pickle.load(open(filename, 'rb'))
        #pred_args = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
        data = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,oldpeak,slope,ca,thal,exang]])
        model_predcition = model.predict(data)

       # mul_reg = open('heart_svm_13.pkl','rb')
        #ml_model = jb.load(mul_reg)
      #  filepath='C:\\Users\\maddi\\OneDrive\\Desktop\\Heart-Disease-Prediction-Web-App-master\\heart_svm_13.pkl'
       # model=pickle.load(open(filepath,'rb'))
        #model_predcition = model.predict([pred_args])
       
        if model_predcition == 1:
            res = 'Affected'
        else:
            res = 'Not affected'
        #return res
        return render_template('predict.php', prediction = res)

if __name__ == '__main__':
#    app.debug = True
    app.run()
