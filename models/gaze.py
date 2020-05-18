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
import dlib
import bz2
import shutil
from urllib.request import urlopen
try:
    from .common import *
except:
    from common import *
import cv2
import cv2 as cv
__all__ = ['predict','visualize']

sess = tf.Session()

def model():
    dir = os.path.dirname(os.path.realpath(__file__))+'/gaze'
    pb = glob.glob('%s/*.pb'%(dir))[0]
    with tf.gfile.FastGFile(pb, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    #frame_index = sess.graph.get_tensor_by_name('Webcam/fifo_queue_DequeueMany:0')
    eye = sess.graph.get_tensor_by_name('eye:0')
    #eye_index = sess.graph.get_tensor_by_name('Webcam/fifo_queue_DequeueMany:2')
    heatmaps = sess.graph.get_tensor_by_name('hourglass/hg_2/after/hmap/conv/BiasAdd:0')
    landmarks = sess.graph.get_tensor_by_name('upscale/mul:0')
    radius = sess.graph.get_tensor_by_name('radius/out/fc/BiasAdd:0')
    sess.run(tf.global_variables_initializer())
    return eye,heatmaps,landmarks,radius

_data_format = 'NHWC'
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
            global _data_format
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
            _data_format = self.data_format
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

    from tensorflow.python.framework import graph_util
    constant_graph = graph_util.convert_variables_to_constants(
            sess, sess.graph_def,
            ['hourglass/hg_2/after/hmap/conv/BiasAdd', # heatmaps
             'upscale/mul', # landmarks
             'radius/out/fc/BiasAdd', # radius
             'eye',
            ])
    # Fix nodes of freezed model
    for node in constant_graph.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    with tf.gfile.FastGFile('%s/gaze.pb'%(dir), mode='wb') as f:
        f.write(constant_graph.SerializeToString())

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

def _visualize(face, context):
    frame = context['frame']
#     for x,y in face['landmarks']:
#         cv2.circle(frame, (x, y), 1, (0,255,0), -1, 8)
    oheatmaps,olandmarks,oradius = face['gaze']
    eyes = face['eyes']
    for j in range(2):
        # Decide which landmarks are usable
        heatmaps_amax = np.amax(oheatmaps[j, :].reshape(-1, 18), axis=0)
        # original GazeML demo use 0.7 instead of 0.3
        can_use_eye = np.all(heatmaps_amax > 0.3)
        can_use_eyelid = np.all(heatmaps_amax[0:8] > 0.75)
        can_use_iris = np.all(heatmaps_amax[8:16] > 0.8)
        can_use_eye,can_use_eyelid,can_use_iris = True,False,False
        # Embed eye image and annotate for picture-in-picture
        eye = eyes[j]
        eye_image = eye['image']
        eye_side = eye['side']
        eye_landmarks = olandmarks[j, :]
        eye_radius = oradius[j][0]
        if(eye_side == 'left'):
            eye_landmarks[:, 0] = eye_image.shape[1] - eye_landmarks[:, 0]
            #eye_image = np.fliplr(eye_image)
        # Transform predictions
        eye_landmarks = np.concatenate([eye_landmarks,
                                        [[eye_landmarks[-1, 0] + eye_radius,
                                          eye_landmarks[-1, 1]]]])
        eye_landmarks = np.asmatrix(np.pad(eye_landmarks, ((0, 0), (0, 1)),
                                           'constant', constant_values=1.0))
        eye_landmarks = (eye_landmarks *
                         eye['inv_landmarks_transform_mat'].T)[:, :2]
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
        if can_use_eyelid:
            cv.polylines(
                frame, [np.round(eyelid_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True, color=(255, 255, 0), thickness=1, lineType=cv.LINE_AA,
            )

        if can_use_iris:
            cv.polylines(
                frame, [np.round(iris_landmarks).astype(np.int32).reshape(-1, 1, 2)],
                isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv.LINE_AA,
            )
            cv.drawMarker(
                frame, tuple(np.round(iris_centre).astype(np.int32)),
                color=(0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=4,
                thickness=1, line_type=cv.LINE_AA,
            )

def visualize(context):
    for face in context['faces']:
        if('gaze' not in face):
            continue
        _visualize(face, context)

_landmarks_predictor = None

def _get_dlib_data_file(dat_name):
    dat_dir = os.path.dirname(os.path.realpath(__file__))+'/gaze/3rdparty'
    dat_path = '%s/%s' % (dat_dir, dat_name)
    if not os.path.isdir(dat_dir):
        os.mkdir(dat_dir)

    # Download trained shape detector
    if not os.path.isfile(dat_path):
        with urlopen('http://dlib.net/files/%s.bz2' % dat_name) as response:
            with bz2.BZ2File(response) as bzf, open(dat_path, 'wb') as f:
                shutil.copyfileobj(bzf, f)

    return dat_path

def get_landmarks_predictor():
    """Get a singleton dlib face landmark predictor."""
    global _landmarks_predictor
    if not _landmarks_predictor:
        dat_path = _get_dlib_data_file('shape_predictor_5_face_landmarks.dat')
        # dat_path = _get_dlib_data_file('shape_predictor_68_face_landmarks.dat')
        _landmarks_predictor = dlib.shape_predictor(dat_path)
    return _landmarks_predictor

def detect_landmarks(face, context):
    """Detect 5-point facial landmarks for faces in frame."""
    predictor = get_landmarks_predictor()
    l, t, w, h = face['box']
    rectangle = dlib.rectangle(left=int(l), top=int(t), right=int(l+w), bottom=int(t+h))
    landmarks_dlib = predictor(context['gray'], rectangle)

    def tuple_from_dlib_shape(index):
        p = landmarks_dlib.part(index)
        return (p.x, p.y)

    num_landmarks = landmarks_dlib.num_parts
    landmarks = np.array([tuple_from_dlib_shape(i) for i in range(num_landmarks)])
    face['landmarks'] = landmarks

def detect_eyes(face, context):
    """From found landmarks in previous steps, segment eye image."""
    eyes = []

    # Final output dimensions
    oh, ow = (36, 60)

    landmarks = face['landmarks']

    # Segment eyes
    # for corner1, corner2, is_left in [(36, 39, True), (42, 45, False)]:
    for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        if eye_width == 0.0:
            continue
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # Centre image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Rotate to be upright
        roll = 0.0 if x1 == x2 else np.arctan((y2 - y1) / (x2 - x1))
        rotate_mat = np.asmatrix(np.eye(3))
        cos = np.cos(-roll)
        sin = np.sin(-roll)
        rotate_mat[0, 0] = cos
        rotate_mat[0, 1] = -sin
        rotate_mat[1, 0] = sin
        rotate_mat[1, 1] = cos
        inv_rotate_mat = rotate_mat.T

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        # Centre image
        centre_mat = np.asmatrix(np.eye(3))
        centre_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_centre_mat = np.asmatrix(np.eye(3))
        inv_centre_mat[:2, 2] = -centre_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = centre_mat * scale_mat * rotate_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_rotate_mat * inv_scale_mat *
                             inv_centre_mat)
        eye_image = cv.warpAffine(context['gray'], transform_mat[:2, :], (ow, oh))
        if is_left:
            eye_image = np.fliplr(eye_image)
        eyes.append({
            'image': eye_image,
            'inv_landmarks_transform_mat': inv_transform_mat,
            'side': 'left' if is_left else 'right',
        })
    face['eyes'] = eyes

def eye_preprocess(eye):
    eye = cv.equalizeHist(eye)
    eye = eye.astype(np.float32)
    eye *= 2.0 / 255.0
    eye -= 1.0
    eye = np.expand_dims(eye, -1 if _data_format == 'NHWC' else 0)
    return eye

def _predict(face, context):
    detect_landmarks(face, context)
    detect_eyes(face, context)
    if(len(face['eyes']) != 2):
        return
    eye1 = eye_preprocess(face['eyes'][0]['image'])
    eye2 = eye_preprocess(face['eyes'][1]['image'])
    eyeI = np.concatenate((eye1, eye2), axis=0)
    eyeI = eyeI.reshape(2,36,60,1)
    Placeholder_1 = sess.graph.get_tensor_by_name('learning_params/Placeholder_1:0')
    feed_dict = { eye:eyeI, Placeholder_1:False }
    oheatmaps,olandmarks,oradius = sess.run((heatmaps,landmarks,radius), feed_dict=feed_dict)
    face['gaze'] = (oheatmaps,olandmarks,oradius)

def predict(context):
    for face in context['faces']:
        x, y, w, h = face['box']
        if((w<160) or (h<160)):
            continue
        _predict(face, context)
