from flask import Flask

app = Flask(__name__)
from app import views

from keras.models import load_model
nn = load_model('../frozen/resXception_fer2013.hdf5')