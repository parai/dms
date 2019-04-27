'''Copyright (C) 2019 AS <parai@foxmail.com>'''
import os
import cv2
from models.facenet import predict as face_predict
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', dest='video', default=0,
                    help='specify video sorce, e.g: http://192.168.1.101:4747/video')
args = parser.parse_args()

cv2_root = os.path.dirname(os.path.realpath(cv2.__file__))
video = cv2.VideoCapture(args.video)
haar_face_cascade = cv2.CascadeClassifier('%s/data/haarcascade_frontalface_alt.xml'%(cv2_root))

def detect_faces(frame):
    #https://www.superdatascience.com/blogs/opencv-face-detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

def main():
    ret, frame = video.read()
    while(ret):
        ret, frame = video.read()
        faces = detect_faces(frame)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face,(160,160), cv2.INTER_LINEAR)
            face = face.reshape(1,160,160,3)
            name,dis = face_predict(face)
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
