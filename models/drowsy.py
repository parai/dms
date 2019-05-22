'''Copyright (C) 2019 AS <parai@foxmail.com>'''
# https://github.com/nishagandhi/DrowsyDriverDetection

from keras.models import load_model
import os
import cv2

__all__ = ['predict']

cv2_root = os.path.dirname(os.path.realpath(cv2.__file__))

eye_cascade = cv2.CascadeClassifier('%s/data/haarcascade_eye_tree_eyeglasses.xml'%(cv2_root))
eye_cascade_left = cv2.CascadeClassifier('%s/data/haarcascade_lefteye_2splits.xml'%(cv2_root))
eye_cascade_right = cv2.CascadeClassifier('%s/data/haarcascade_righteye_2splits.xml'%(cv2_root))
def model():
    mh5 = os.path.dirname(os.path.realpath(__file__))+'/drowsy/model.h5'
    net = load_model(mh5)
    return net

#network = model()

#from keras.utils import plot_model
#plot_model(network, to_file='model.png', show_shapes=True)

eyedb = {}
confirmed = 10

def detect_eye(face, name):
    if(True):
        eyes = eye_cascade.detectMultiScale(face)
        eye_left = eyes
        eye_right = eyes
    else:
        eye_left = eye_cascade_left.detectMultiScale(face)
        eye_right = eye_cascade_right.detectMultiScale(face)
    ex1=ex2=ey1=ey2=-1
    for (ex,ey,ew,eh) in eye_left:
        if((ex<ex1) or (ex1==-1)):
            ex1 = ex
        if((ey<ey1) or (ey1==-1)):
            ey1 = ey
    for (ex,ey,ew,eh) in eye_right:
        if((ex+ew)>ex2):
            ex2 = ex+ew
        if((ey+eh)>ey2):
            ey2 = ey+eh
    if((ex1<ex2) and (ey1<ey2) and ((ex2-ex1)/(ey2-ey1)>2.5) and (name!='other')):
        if(name not in eyedb):
            h, w = face.shape
            eyedb[name] = [ex1/w,ey1/h,ex2/w,ey2/h,1]
        elif(eyedb[name][4]<confirmed):
            h, w = face.shape
            n = eyedb[name][4]
            eyedb[name][0] = (eyedb[name][0]*n + ex1/w)/(n+1)
            eyedb[name][1] = (eyedb[name][1]*n + ey1/w)/(n+1)
            eyedb[name][2] = (eyedb[name][2]*n + ex2/w)/(n+1)
            eyedb[name][3] = (eyedb[name][3]*n + ey2/w)/(n+1)
            eyedb[name][4] += 1
            if(eyedb[name][4] >= confirmed):
                print('%s\'s eye location: %s'%(name, eyedb[name][:-1]))
    return ex1,ey1,ex2,ey2

def _predict(face, name):
    if((name!='other') and (name in eyedb) and (eyedb[name][4]>=confirmed)):
        h, w = face.shape
        ex1,ey1,ex2,ey2,_ = eyedb[name]
        ex1,ey1,ex2,ey2 = int((ex1-0.02)*w),int(ey1*h),int((ex2+0.02)*w),int(ey2*h)
    else:
        ex1,ey1,ex2,ey2 = detect_eye(face,name)
    return '?',0,(ex1,ey1,ex2,ey2)

def predict(context):
    for face in context['faces']:
        if('faceid' in face):
            name,_ = face['faceid']
        else:
            name = 'other'
        x, y, w, h = face['box']
        gray = context['gray'][y:y+h, x:x+w]
        status,prob,box = _predict(gray, name)
        face['drowsy'] = (status,prob,box)
