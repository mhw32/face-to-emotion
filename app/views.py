from app import app
from flask import request


@app.route('/')
@app.route('/index')
def index():
    return "ResXceptionNet API"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.data
