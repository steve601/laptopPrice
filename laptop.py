from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

def load_model():
    with open('laptop.pkl','rb') as file:
        data = pickle.load(file)
    return data

objects = load_model()
model = objects['model']
preprocessor = objects['preprocessor']

@app.route('/')
def homepge():
    return render_template('laptop.html')

@app.route('/predict',methods=['POST'])
def do_prediction():
    a = request.form.get('brand')
    b = request.form.get('processor_speed')
    c = request.form.get('ram_size')
    d = request.form.get('storage_capacity')
    e = request.form.get('screen_size')
    f = request.form.get('weight')
    
    columns = ['brand','processor_speed','ram_size','storage_capacity','screen_size','weight']
    x = pd.DataFrame([[a,b,c,d,e,f]],columns=columns)
    
    x = preprocessor.transform(x)
    prediction = model.predict(x)
    
    msg = f"Estimated price is: ${np.round(prediction,2)}"
    
    return render_template('laptop.html',text=msg)

if __name__ == '__main__':
    app.run(host = "0.0.0.0")