'''Copyright (C) 2019 AS <parai@foxmail.com>'''

# https://www.cnblogs.com/xiaohuahua108/p/6505756.html
# https://gitee.com/PanChenGeWang/facenet_face_regonistant
# https://github.com/davidsandberg/facenet
# https://zhuanlan.zhihu.com/p/24837264
# https://github.com/WindZu/facenet_facerecognition

import os
import glob
import tensorflow as tf
import numpy as np
import cv2
import pickle

from sklearn.metrics.pairwise import euclidean_distances

__all__ = ['predict']

sess = tf.Session()

def model():
    dir = os.path.dirname(os.path.realpath(__file__))+'/facenet/20180408-102900'
    if(True):
        pb = glob.glob('%s/*.pb'%(dir))[0]
        with tf.gfile.FastGFile(pb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')
    else:
        meta = glob.glob('%s/*.meta'%(dir))[0]
        ckpt = glob.glob('%s/*.index'%(dir))[0][:-6]
        saver = tf.train.import_meta_graph(meta)
        saver.restore(sess, ckpt)

    input = sess.graph.get_tensor_by_name('input:0')
    embeddings = sess.graph.get_tensor_by_name('embeddings:0')
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    sess.run(tf.global_variables_initializer())
    return input,embeddings,phase_train_placeholder

input_face,embeddings,phase_train_placeholder = model()

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

if(os.path.exists('facedb.pkl')):
    people = pickle.load(open('facedb.pkl','rb'))
else:
    people = {}

def _predict(face):
    global last_embeddings

    face = cv2.resize(face,(160,160), cv2.INTER_LINEAR)
    face = prewhiten(face)
    face = face.reshape(1,160,160,3)

    feed_dict = { input_face: face, phase_train_placeholder:False }
    emb = sess.run(embeddings, feed_dict=feed_dict) 

    fname = 'other'
    dis = 0
    for name, pembs in people.items():
        isMe = True
        for pemb in pembs:
            dis = euclidean_distances(pemb, emb)[0][0]
            if(dis < 1.05):
                pass
            else:
                isMe = False
                break
        if(isMe):
            fname = name
            break
    if(fname == 'other'):
        fname = input('new face detected, registering, input the name:')
        if(fname!=''):
            people[fname] = [emb]
            pickle.dump(people, open('facedb.pkl','wb'))
        else:
            fname = 'other'
    elif(len(people[fname]) < 10):
        people[fname].append(emb)
        pickle.dump(people, open('facedb.pkl','wb'))

    return fname,dis

def predict(context):
    for face in context['faces']:
        fname,dis = _predict(face['frame'])
        face['faceid'] = (fname,dis)

