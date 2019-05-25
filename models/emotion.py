'''Copyright (C) 2019 AS <parai@foxmail.com>'''
# https://github.com/atulapra/Emotion-detection

import os,glob,cv2
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

__all__ = ['predict']
target_classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def model():
    dir = os.path.dirname(os.path.realpath(__file__))+'/emotion'
    network = input_data(shape = [None, 48, 48, 1])
    network = conv_2d(network, 64, 5, activation = 'relu')
    network = max_pool_2d(network, 3, strides = 2)
    network = conv_2d(network, 64, 5, activation = 'relu')
    network = max_pool_2d(network, 3, strides = 2)
    network = conv_2d(network, 128, 4, activation = 'relu')
    network = dropout(network, 0.3)
    network = fully_connected(network, 3072, activation = 'relu')
    network = fully_connected(network, len(target_classes), activation = 'softmax')
    dnn = tflearn.DNN(network)
    dnn.load(glob.glob('%s/*.tflearn.index'%(dir))[0][:-6])
    return dnn

network = model()

def _predict(face):
    face = cv2.resize(face,(48,48), interpolation = cv2.INTER_CUBIC)/255.0
    face = face.reshape(1,48,48,1)
    emotions = network.predict(face)[0]
    emo,prob = target_classes[0],emotions[0]
    for id, p in enumerate(emotions):
        if(p > prob):
            emo = target_classes[id]
            prob = p
    return emo,prob

def predict(context):
    for face in context['faces']:
        x, y, w, h = face['box']
        if((w<48) or (h<48)):
            continue
        gray = context['gray'][y:y+h, x:x+w]
        emo,prob = _predict(gray)
        face['emotion'] = (emo,prob)
