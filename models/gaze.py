'''Copyright (C) 2019 AS <parai@foxmail.com>'''
# https://github.com/swook/GazeML
'''
 Below part of code add to GazeML/src/core/model.py inference_generator
to export the GazeML frozen graph:

    print('fetches:', fetches) # to get output tensors' name

    sess = self._tensorflow_session
    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph_def,
            ['hourglass/hg_2/after/hmap/conv/BiasAdd', # heatmaps
             'upscale/mul', # landmarks
             'radius/out/fc/BiasAdd', # radius
             'Webcam/fifo_queue_DequeueMany', # frame_index, eye, eye_index
            ])
    with tf.gfile.FastGFile('./gaze.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
'''

import os
import glob
import tensorflow as tf
import numpy as np

__all__ = ['predict']

sess = tf.Session()

def model():
    dir = os.path.dirname(os.path.realpath(__file__))+'/gaze'
    pb = glob.glob('%s/*.pb'%(dir))[0]
    with tf.gfile.FastGFile(pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    frame_index = sess.graph.get_tensor_by_name('Webcam/fifo_queue_DequeueMany:0')
    eye = sess.graph.get_tensor_by_name('Webcam/fifo_queue_DequeueMany:1')
    eye_index = sess.graph.get_tensor_by_name('Webcam/fifo_queue_DequeueMany:2')
    landmarks = sess.graph.get_tensor_by_name('upscale/mul:0')
    heatmaps = sess.graph.get_tensor_by_name('hourglass/hg_2/after/hmap/conv/BiasAdd:0')
    sess.run(tf.global_variables_initializer())
    return eye,eye_index,frame_index,landmarks,radius

eye,eye_index,frame_index,landmarks,radius = model()

for i,node in enumerate(sess.graph_def.node):
    try:
        me = sess.graph.get_tensor_by_name('%s:0'%(node.name))
        print(node.name, me.shape)
    except:
        print(node.name)
# tensorboard --logdir="./graphs"
#tf.summary.FileWriter('./graphs', sess.graph)

def predict(eye):
    pass