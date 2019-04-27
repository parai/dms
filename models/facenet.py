'''Copyright (C) 2019 AS <parai@foxmail.com>'''

# https://www.cnblogs.com/xiaohuahua108/p/6505756.html
# https://gitee.com/PanChenGeWang/facenet_face_regonistant
# https://github.com/davidsandberg/facenet
# https://zhuanlan.zhihu.com/p/24837264
# https://github.com/WindZu/facenet_facerecognition

import os
import tensorflow as tf
import numpy as np
import cv2

from sklearn.metrics.pairwise import euclidean_distances

__all__ = ['predict']

last_embeddings = None

sess = tf.Session()

def model():
    dir = os.path.dirname(os.path.realpath(__file__))+'/facenet/20170512-110547'
    if(False):
        with tf.gfile.FastGFile('%s/20170512-110547.pb'%(dir), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    else:
        saver = tf.train.import_meta_graph('%s/model-20170512-110547.meta'%(dir))
        saver.restore(sess, '%s/model-20170512-110547.ckpt-250000'%(dir))

    input = sess.graph.get_tensor_by_name('input:0')
    embeddings = sess.graph.get_tensor_by_name('embeddings:0')
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    sess.run(tf.global_variables_initializer())
    return input,embeddings,phase_train_placeholder

input,embeddings,phase_train_placeholder = model()

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def predict(face):
    global last_embeddings

    face = cv2.resize(face,(160,160), cv2.INTER_LINEAR)
    face = prewhiten(face)
    face = face.reshape(1,160,160,3)

    feed_dict = { input: face, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict) 

    if(last_embeddings is None):
        name = 'unknown'
        last_embeddings = emb
        dis = -1
    else:
        dis = euclidean_distances(last_embeddings, emb)[0][0]
        
        if(dis < 1.0):
            name = 'same'
        else:
            name = 'face changed'

    return name,dis

