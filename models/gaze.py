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
try:
    from .common import *
except:
    from common import *
import cv2
import cv2 as cv
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

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=2, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    return image_out

def _predict(eyes, frame):
    eyes = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
    h,w = eyes.shape
    ew = int(w/3)
    eh = int(36.0*ew/60)
    ey1 = int(eh/5)
    ex1 = 0
    ey2 = int(eh/5)
    ex2 = w-ew
    eye1 = eyes[ey1:ey1+eh, ex1:ex1+ew]
    eye2 = eyes[ey2:ey2+eh, ex2:ex2+ew]
    eyeI = np.concatenate((eye1, eye2))
    eyeI = cv2.resize(eyeI, (60, 36*2), interpolation=cv2.INTER_LINEAR)
    eyeI = eyeI.reshape(2,36,60,1)

    Placeholder_1 = sess.graph.get_tensor_by_name('learning_params/Placeholder_1:0')
    feed_dict = { eye:eyeI, Placeholder_1:False }
    oheatmaps,olandmarks,oradius = sess.run((heatmaps,landmarks,radius), feed_dict=feed_dict)
    for j in range(2):
        # Decide which landmarks are usable
        heatmaps_amax = np.amax(oheatmaps[j, :].reshape(-1, 18), axis=0)
        can_use_eye = np.all(heatmaps_amax > 0.7)
        can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
        can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)
        # Embed eye image and annotate for picture-in-picture
        eye_image = eyeI[j]
        eye_landmarks = olandmarks[j, :]
        eye_radius = oradius[j][0]
        if j == 0: # left eye if 0
            eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
            eye_image = np.fliplr(eye_image)
        # Embed eye image and annotate for picture-in-picture
        eye_upscale = 2
        eye_image_raw = cv.cvtColor(cv.equalizeHist(eye_image), cv.COLOR_GRAY2BGR)
        eye_image_raw = cv.resize(eye_image_raw, (0, 0), fx=eye_upscale, fy=eye_upscale)
        eye_image_annotated = np.copy(eye_image_raw)
        if can_use_eyelid:
            cv.polylines(
                eye_image_annotated,
                [np.round(eye_upscale*eye_landmarks[0:8]).astype(np.int32)
                                                         .reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv.LINE_AA,
            )
        if can_use_iris:
            cv.polylines(
                eye_image_annotated,
                [np.round(eye_upscale*eye_landmarks[8:16]).astype(np.int32)
                                                          .reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
            )
            cv.drawMarker(
                eye_image_annotated,
                tuple(np.round(eye_upscale*eye_landmarks[16, :]).astype(np.int32)),
                color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                thickness=1, line_type=cv.LINE_AA,
            )
        cv.imshow('eye', eye_image_annotated)
        # Transform predictions
        eye_landmarks = np.concatenate([eye_landmarks,
                                        [[eye_landmarks[-1, 0] + eye_radius,
                                          eye_landmarks[-1, 1]]]])
        eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                           'constant', constant_values=1.0))
#         eye_landmarks = (eye_landmarks *
#                          eye['inv_landmarks_transform_mat'].T)[:, :2]
        eye_landmarks = np.asarray(eye_landmarks)
        eyelid_landmarks = eye_landmarks[0:8, :]
        iris_landmarks = eye_landmarks[8:16, :]
        iris_centre = eye_landmarks[16, :]
        eyeball_centre = eye_landmarks[17, :]
        eyeball_radius = np.linalg.norm(eye_landmarks[18, :] -
                                        eye_landmarks[17, :])

        # visualize gaze direction
        if can_use_eye:
            # Visualize landmarks
            cv.drawMarker(  # Eyeball centre
                frame, tuple(np.round(eyeball_centre).astype(np.int32)),
                color=(0, 255, 0), markerType=cv.MARKER_CROSS, markerSize=4,
                thickness=1, line_type=cv.LINE_AA
            )
            i_x0, i_y0 = iris_centre
            e_x0, e_y0 = eyeball_centre
            theta = -np.arcsin(np.clip((i_y0 - e_y0) / eyeball_radius, -1.0, 1.0))
            phi = np.arcsin(np.clip((i_x0 - e_x0) / (eyeball_radius * -np.cos(theta)),
                                    -1.0, 1.0))
            current_gaze = np.array([theta, phi])
            draw_gaze(frame, iris_centre, current_gaze,
                                length=120.0, thickness=1)

def predict(context):
    for face in context['faces']:
        if('drowsy' in face):
            _,_,(ex1,ey1,ex2,ey2) = face['drowsy']
            if((ex1<ex2) and (ey1<ey2)):
                _predict(face['frame'][ey1:ey2, ex1:ex2], context['frame'])
