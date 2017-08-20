from flask import Flask
from keras.models import load_model
from config import MODEL_FILE

nn = load_model(MODEL_FILE)

app = Flask(__name__)
from app import views
