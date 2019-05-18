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
import sys
import glob
import tensorflow as tf
import numpy as np
from common import *

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
    heatmaps = sess.graph.get_tensor_by_name('hourglass/hg_2/after/hmap/conv/BiasAdd:0')
    landmarks = sess.graph.get_tensor_by_name('upscale/mul:0')
    radius = sess.graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')
    sess.run(tf.global_variables_initializer())
    return eye,heatmaps,landmarks,radius

def model2():
    dir = os.path.dirname(os.path.realpath(__file__))+'/gaze'
    GazeML = '%s/GazeML'%(dir)
    if(not os.path.exists(GazeML)):
        RunCommand('cd %s && git clone https://github.com/swook/GazeML'%(dir))
        RunCommand('cd %s && sh get_mpiigaze_hdf.bash'%(GazeML))
        RunCommand('cd %s && sh get_trained_weights.bash'%(GazeML))
        RunCommand('cd %s && touch __init__.py'%(GazeML))
        RunCommand('cd %s && touch src/__init__.py'%(GazeML))
    sys.path.append(dir)
    sys.path.append('%s/src'%(GazeML))
    from GazeML.src.models import ELG
    class DataSource():
        def __init__(self):
            # Check if GPU is available
            from tensorflow.python.client import device_lib
            session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
            gpu_available = False
            try:
                gpus = [d for d in device_lib.list_local_devices(config=session_config)
                        if d.device_type == 'GPU']
                gpu_available = len(gpus) > 0
            except:
                pass
            self.batch_size = 2
            self.data_format = 'NCHW' if gpu_available else 'NHWC'
            self.output_tensors = {
                'eye': tf.placeholder(tf.float32, [2, 36, 60, 1], name='eye')
            }
        def cleanup(self):
            pass
        def create_and_start_threads(self):
            pass

    tf.logging.set_verbosity(tf.logging.INFO)
    data_source = DataSource()
    elgmodel = ELG(
                sess, train_data={'videostream': data_source},
                first_layer_stride=1,
                num_modules=2,
                num_feature_maps=32,
                learning_schedule=[
                    {
                        'loss_terms_to_optimize': {'dummy': ['hourglass', 'radius']},
                    },
                ],
            )

    elgmodel.initialize_if_not(training=False)
    elgmodel.checkpoint.load_all()

    eye = sess.graph.get_tensor_by_name('eye:0')
    heatmaps = sess.graph.get_tensor_by_name('hourglass/hg_2/after/hmap/conv/BiasAdd:0')
    landmarks = sess.graph.get_tensor_by_name('upscale/mul:0')
    radius = sess.graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')

    return eye,heatmaps,landmarks,radius

eye,heatmaps,landmarks,radius = model2()

# tensorboard --logdir="./graphs"
#tf.summary.FileWriter('./graphs', sess.graph)

def predict(eye):
    pass