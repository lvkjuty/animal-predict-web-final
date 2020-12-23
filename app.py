from tensorflow import keras
from keras.applications import ResNet50
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils

from PIL import Image
import numpy as np
import pandas as pd
import requests
import flask
from flask import render_template
from flask_cors import CORS
import io
import os

app = flask.Flask(__name__)
CORS(app)
model = None

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image, mode="caffe")

    # return the processed image
    return image
@app.route("/")
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return "Hello Flask"

@app.route('/predict', methods=['POST'])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(224, 224))

            #load model 
            model = keras.models.load_model("./animal_two_class.h5")
            lst = ['dog','cat']
            # lst = ['butterfly','cat','chicken','cow','dog','elephant','horse','sheep','spider','squirrel']
            pred = model.predict(image.reshape(1,224,224,3))[0]
            result = {}
            for i in range(len(lst)):
                result[lst[i]] = float(pred[i])
            
            data['success'] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(result)

if __name__ == '__main__':
    #model = keras.models.load_model(os.path.join(r"..\UseTensorflow", "model.h5"))
    model = keras.models.load_model("./animal_two_class.h5 ")
    app.run(port=8080)
