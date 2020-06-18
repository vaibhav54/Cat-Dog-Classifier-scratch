import cv2
import os
import numpy as np

from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model


app = Flask(__name__)
model = load_model('models/new_catdog_scratch.h5' )


@app.route('/') 

def hello_world(): 
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        print(file_path)
  
        test_image=cv2.imread(file_path)
        test_image=cv2.resize(test_image,(150,150))
        test_image = np.reshape(test_image,(1,150,150,3))
        pred = model.predict_classes(test_image)
        os.remove(file_path)
        
        str1 = 'Dog'
        str2 = 'Cat'
        if (pred[0]) == 1:
            return str1
        else:
            return str2
    return None

if __name__ == '__main__':
        app.run(debug=True, port=8000, host='0.0.0.0')

