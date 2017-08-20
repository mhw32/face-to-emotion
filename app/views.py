from app import app
from flask import request, Response, jsonify

import json
from .helper import (read_base64_image, image_to_face_crop,
                     classify_image)


@app.route('/')
@app.route('/index')
def index():
    return "ResXceptionNet API"


@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data)
    base64_str = data.get('image')

    # guaranteed to have a list of faces
    faces = data.get('faces')  # trueface api

    rgb_image = read_base64_image(base64_str)

    # face_images is a list of rgb images
    face_images = image_to_face_crop(rgb_image, faces)

    emotion_maps = []
    for face_image in face_images:
        emotion_map = classify_image(face_image)
        emotion_maps.append(emotion_map)

    return Response(response=jsonify(emotion_maps),
                    mimetype="application/json")
