from flask import Flask
import tensorflow as tf
from keras.models import load_model
from config import MODEL_FILE

nn = load_model(MODEL_FILE)
nn._make_predict_function()
graph = tf.get_default_graph()

app = Flask(__name__)
from app import views
