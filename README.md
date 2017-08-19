# Face2Emotion

Deep convolutional net to predict coarse emotions from image frames.

The architecture is modeled after the technical report **Real-time Convolutional Neural Networks for Emotion and Gender Classification** by Octavio Arriaga and Paul G. Ploger.

The authors designed a Xception-based model that combines residual blocks (as in ResNet [1](https://arxiv.org/abs/1512.03385)) and depth-wise separable convolutions (as in the traditional Xception [2](https://arxiv.org/abs/1611.05431)).

The model is trained on the FER2013 dataset [3](https://arxiv.org/abs/1307.0414) and gets 66\%.
