# DMS - Driver Monitor System

This DMS is based on OpenCV and Deep Learning to monitor driver's status.

The below is the list of open source Deep Larning models used by DMS:

* [facenet](https://github.com/davidsandberg/facenet): Face Recognition using Tensorflow

* [emotion](https://github.com/atulapra/Emotion-detection): Emotion(happy/angry/neturl,etc) detection using tflearn

* drowsy: TBD

* [gaze](https://github.com/swook/GazeML): Eye looking direction detection.

With facenet, who is the driver can be recognized, thus can loading the driver's preference when driving the car.

With emotion detection, can know whether the driver is in a good mood or not, do alert if the driver is too angry, angry may makes people lost mind.

With drowsy detection, do alert to wake up the driver and have a rest if the driver is too sleepy.

With gaze detection, can know whether the driver is focusing on driving or not, can know where the driver is looking at. Do alert if the driver is not looking ahead the road for example when the driver is looking down to check phone message.

![DMS](doc/demo.gif)

The pretrained weights can be downloaded from each model used by this DMS demo, the folder structure looks like as below:

```c
+ models
    +-- drowsy
    |   +-- TBD
    +-- emotion
    |   +-- model_1_atul.tflearn.data-00000-of-00001
    |   +-- model_1_atul.tflearn.index
    |   +-- model_1_atul.tflearn.meta
    +-- facenet
    |   +-- 20180402-114759
    |       +-- 20180402-114759.pb
    +-- gaze
        +-- 3rdparty
        +-- GazeML
            +-- outputs
                +-- ELG_i60x36_f60x36_n32_m2
```
