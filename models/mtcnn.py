'''Copyright (C) 2019 AS <parai@foxmail.com>'''

import os
import tensorflow as tf
import numpy as np
import models.align.detect_face as detect_face

from sklearn.metrics.pairwise import euclidean_distances

__all__ = ['predict']

sess = tf.Session()

def model():
    return detect_face.create_mtcnn(sess, None)

pnet, rnet, onet = model()
minsize = 20 # minimum size of face
threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
factor = 0.709 # scale factor

def predict(context):
    frame = context['frame']
    img = frame
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
  
    if len(bounding_boxes) < 1:
        context['faces'] = []
    else:    
        boxs = []
        det=bounding_boxes

        det[:,0]=np.maximum(det[:,0], 0)
        det[:,1]=np.maximum(det[:,1], 0)
        det[:,2]=np.minimum(det[:,2], img_size[1])
        det[:,3]=np.minimum(det[:,3], img_size[0])

        det=det.astype(int)

        for i in range(len(bounding_boxes)):
            boxs.append((det[i,0],det[i,1],det[i,2]-det[i,0],det[i,3]-det[i,1]))

        context['faces'] = [{'box':(x, y, w, h), 'frame':frame[y:y+h, x:x+w]} for (x, y, w, h) in boxs]

