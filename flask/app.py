import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request, redirect, url_for
import requests 
import pandas as pd
import tensorflow as tf
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app=Flask(__name__)

model=load_model("fruit.h5")
model1=load_model("vegetable.h5")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        img=image.load_img(file_path,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        plant=request.form['plant']
        print(plant)
        if(plant=="vegetable"):
            preds=np.argmax(model.predict(x),axis=1)
            index=['Pepper Bell Bacterial spot','Pepper bell healthy','Potato Early blight','Potato Late blight','Potato healthy','Tomato Bacterial spot','Tomato Late blight','Tomato Leaf Mold','Tomato Septoria leaf spot']
            print(index[preds[0]])
            df=pd.read_excel('precautions - veg.xlsx')
            print(df.iloc[preds[0]]['caution'])
        else:
            preds=np.argmax(model1.predict(x),axis=1)
            index=['Apple Black rot','Apple healthy','Corn (maize) Northern Leaf Blight','Corn (maize) healthy','Peach Bacterial spot',
 'Peach healthy']
            print(index[preds[0]])
            df=pd.read_excel('precautions - fruits.xlsx')
            print(df.iloc[preds[0]]['caution'])
        #text="The prediction is : " +str(index[preds[0]])
        #return text
        return df.iloc[preds[0]]['caution']
         
    
if __name__=='__main__':
    app.run(debug=False)
