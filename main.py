'''Copyright (C) 2019 AS <parai@foxmail.com>'''
import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', dest='video', default=0,
                    help='specify video sorce, e.g: http://192.168.1.101:4747/video')
parser.add_argument('-d', dest='detect', default='cv2',
                    help='which method to detect location of face, cv2 or mtcnn')
parser.add_argument('-f', dest='filter', default='1',
                    help='sample rate of the input')
networks = ['facenet','emotion','drowsy','gaze']
parser.add_argument('-n', '--network',
                    default=networks,
                    help='networks to be enabled from %s, default all is enabled'%(networks), 
                    type=str, nargs='+')
args = parser.parse_args()

print('networks: %s'%(args.network))
cv2_root = os.path.dirname(os.path.realpath(cv2.__file__))
video = cv2.VideoCapture(args.video)

if('facenet' in args.network):
    from models.facenet import predict as face_recognise
if('emotion' in args.network):
    from models.emotion import predict as face_emotion
if('drowsy' in args.network):
    from models.drowsy import predict as face_drowsy
if('gaze' in args.network):
    from models.gaze import predict as gaze_direction
    from models.gaze import visualize as gaze_visualize
if(args.detect == 'cv2'):
    haar_face_cascade = cv2.CascadeClassifier('%s/data/haarcascade_frontalface_alt.xml'%(cv2_root))
    def face_detect(context):
        #https://www.superdatascience.com/blogs/opencv-face-detection
        frame = context['frame']
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxs = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        context['faces'] = [{'box':(x, y, w, h)} for (x, y, w, h) in boxs]
else:
    from models.mtcnn import predict as face_detect

def visualize(context):
    frame = context['frame']
    for face in context['faces']:
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if('faceid' in face):
            name,dis = face['faceid']
            cv2.putText(frame, '%s %.2f'%(name, dis), 
                    (x, y+h+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                    1.0, (0, 0 ,255), thickness = 1, lineType = 2)
        if('emotion' in face):
            emotion,eprob = face['emotion']
            cv2.putText(frame, '%s %.2f'%(emotion, eprob), 
                    (x, y+h), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                    1.0, (0, 0 ,255), thickness = 1, lineType = 2)
#         if('drowsy' in face):
#             drowsy,prob,(ex1,ey1,ex2,ey2) = face['drowsy']
#             if((ex1<ex2) and (ey1<ey2)):
#                 cv2.rectangle(frame, (x+ex1, y+ey1), (x+ex2, y+ey2), (0, 255, 0), 2)
    if('gaze' in args.network): gaze_visualize(context)
    cv2.imshow('frame',frame)

def main():
    ret, frame = video.read()
    filter = eval(args.filter)
    while(ret):
        h,w,_=frame.shape
        if(w>1024):
            h = int(1024*h/w)
            w = 1024
            frame = cv2.resize(frame, (w,h), cv2.INTER_LINEAR)
        context = { 'frame': frame, 'gray': cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) }
        face_detect(context)
        if('facenet' in args.network): face_recognise(context)
        if('emotion' in args.network): face_emotion(context)
        if('drowsy' in args.network): face_drowsy(context)
        if('gaze' in args.network): gaze_direction(context)
        visualize(context)
        if((cv2.waitKey(10)&0xFF) == ord('q')):
            break
        fn = 0
        while(fn < filter):
            fn += 1
            ret, frame = video.read()
    video.release()
    cv2.destroyAllWindows()

if(__name__ == '__main__'):
    main()
