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

dir = os.path.dirname(os.path.realpath(__file__))
pb = '%s/align/PNet.pb'%(dir)
if(not os.path.exists(pb)):
    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, 
            ['pnet/conv4-2/BiasAdd', 'pnet/prob1'])
    with tf.gfile.FastGFile(pb, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
pb = '%s/align/RNet.pb'%(dir)
if(not os.path.exists(pb)):
    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, 
            ['rnet/conv5-2/conv5-2', 'rnet/prob1'])
    with tf.gfile.FastGFile(pb, mode='wb') as f:
        f.write(constant_graph.SerializeToString())
pb = '%s/align/ONet.pb'%(dir)
if(not os.path.exists(pb)):
    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, 
            ['onet/conv6-2/conv6-2', 'onet/conv6-3/conv6-3'])
    with tf.gfile.FastGFile(pb, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

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

        context['faces'] = [{'box':(x, y, w, h)} for (x, y, w, h) in boxs]

