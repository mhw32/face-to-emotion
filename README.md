# Face2Emotion

Deep convolutional net to predict coarse emotions from image frames.

The architecture is modeled after the technical report **Real-time Convolutional Neural Networks for Emotion and Gender Classification** by Octavio Arriaga and Paul G. Ploger.

The authors designed a Xception-based model that combines residual blocks (as in ResNet [[1](https://arxiv.org/abs/1512.03385)]) and depth-wise separable convolutions (as in the traditional Xception [[2](https://arxiv.org/abs/1611.05431)]).

The model is trained on the FER2013 dataset [[3](https://arxiv.org/abs/1307.0414)] and reaches 66% accuracy. See `models/` for Keras code and see `frozen` for trained weights.

## Flask Application

There's also a folder in `app/` that stores a vanilla web application with a single endpoint that reads a base64 encoded string representing an image and outputs a map of emotion probabilities per face.

**Note**: This assumes that incoming requests will also pass coordinates for facial objects in the scene. In our instance, we used the [True Face API](http://trueface.ai/). We use the coordinates to pre-crop our image before sending it to our model.

## Additional Info

This project was entered in the TrueFace.ai hackathon. A Heroku instance is currently up at [https://face2emotionapp.herokuapp.com/](https://face2emotionapp.herokuapp.com/).
