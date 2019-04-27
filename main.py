'''Copyright (C) 2019 AS <parai@foxmail.com>'''
import os
import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', dest='video', default=0,
                    help='specify video sorce, e.g: http://192.168.1.101:4747/video')
parser.add_argument('-d', dest='detect', default='mtcnn',
                    help='which method to detect location of face, cv2 or mtcnn')
args = parser.parse_args()

cv2_root = os.path.dirname(os.path.realpath(cv2.__file__))
video = cv2.VideoCapture(args.video)

from models.facenet import predict as face_recognise
if(args.detect == 'cv2'):
    haar_face_cascade = cv2.CascadeClassifier('%s/data/haarcascade_frontalface_alt.xml'%(cv2_root))
else:
    from models.mtcnn import predict as face_detect

def cv2_face_detect(frame):
    #https://www.superdatascience.com/blogs/opencv-face-detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxs = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    faces = []
    for (x, y, w, h) in boxs:
        face = frame[y:y+h, x:x+w]
        faces.append(face)
    return boxs,faces

def main():
    ret, frame = video.read()
    while(ret):
        ret, frame = video.read()
        if(args.detect == 'cv2'):
            boxs, faces = cv2_face_detect(frame)
        else:
            boxs, faces = face_detect(frame)
        for i, (x, y, w, h) in enumerate(boxs):
            face = faces[i]
            name,dis = face_recognise(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, '%s %.2f'%(name, dis), (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 
                        1.0, (0, 0 ,255), thickness = 1, lineType = 2)
        cv2.imshow('frame',frame)
        if((cv2.waitKey(10)&0xFF) == ord('q')):
            break
    cap.release()
    cv2.destroyAllWindows()

if(__name__ == '__main__'):
    main()
