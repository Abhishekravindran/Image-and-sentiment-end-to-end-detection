from __future__ import division, print_function
# coding=utf-8
#import sys
import os
#import glob
#import re
import numpy as np
from PIL import Image

#from rmn import RMN
#import cv2
#import keras

# Keras
#from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


# Load your trained model
model = load_model('model_Face.h5')
#model1 = load_model('emotion_Face.h5')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#m=RMN()
#model=keras.models.load_model('model_vgg19.h5')




def model_predict(img_path, model):
    
    class_indices=['Aditya Aunt',
                   'Aditya Kumar Tripathi',
                   'Amma',
                   'Aparna Tripathi',
                   'Imli',
                   'Rupali',
                   'Satyakam',
                   'dhruv tripathi',
                   'face_extracted',
                   'police',
                   'unknown1',
                   'unknown2']
    
    
    
    img = Image.open(img_path)
    
    x = np.array(img.resize((48,48)))
    roi = x / 255
    #roi
    roi = np.expand_dims(roi, axis = 0)

    #x = x.reshape(1, 48, 48, 3)

    #result = model.predict([x])
    #results1=m.detect_emotion_for_single_face_image(img)
    probs = model.predict(roi)
     #probs
    result = np.argmax(probs)
    
    
    preds= (str("The Image is Predicted as "+class_indices[result]))#,class_indices1[result1])
    
    return preds

#----------below is the implementation of emeotion detection which has to be run seperately------     
"""def model1_predict(img_path, model1):
    class_indices1=['angry',
                     'disgust',
                     'fear',
                     'happy',
                     'neutral',
                     'sad',
                     'suprise']
    img = Image.open(img_path)
    
    

    x = np.array(img.resize((48,48)))
    roi = x / 255
    #roi
    roi = np.expand_dims(roi, axis = 0)

    #x = x.reshape(1, 48, 48, 3)

    #result = model.predict([x])
    #results1=m.detect_emotion_for_single_face_image(img)
    probs = model1.predict(roi)
     #probs
    result1 = np.argmax(probs)
    #result1
    preds1= (class_indices1[result1])
    
    return preds1"""
#----------------------------------------------------------------------#
#cont of image det

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        #preds = model1_predict(file_path, model1)#run this by commenting the above line for emotion detection from a given face

        result=preds
        
        
        
        return result  
        
        
    return None




if __name__ == '__main__':
    app.run(debug=True)
