# Lightweight Head Pose Estimation
This is an official implement for [**Accurate Head Pose Estimation Using Image Rectification and Lightweight Convolutional Neural Network**](https://ieeexplore.ieee.org/abstract/document/9693249?casa_token=IEBanEdMVjIAAAAA:XJZ3g0tn6gD8FOH-0DuB3j8i9kLZn6McNf1BkJTq6yfPfi5X9jZxo5WmfJX3D-267dIWef5M)

## Abstract
Head pose estimation is an important step for many human-computer interaction applications such as face detection, facial recognition, and facial expression classification. Accurate head pose estimation benefits these applications that require face image as the input. Most head pose estimation methods suffer from perspective distortion because the users do not always align their face perfectly with the camera. This paper presents a new approach that uses image rectification to reduce the negative effect of perspective distortion and a lightweight convolutional neural network to obtain highly accurate head pose estimation. The proposed method calculates the angle between the camera optical axis and the projection vector of the face center. The face image is rectified using this estimated angle through perspective transformation. A lightweight network with the size of only 0.88 MB is designed to take the rectified face image as the input to perform head pose estimation. The output of the network, the head pose estimation of the rectified face image, is transformed back to the camera coordinate system as the final head pose estimation. Experiments on public benchmark datasets show that the proposed image rectification and the newly designed lightweight network remarkably improve the accuracy of head pose estimation. Compared with state-of-the-art methods, our approach achieves both higher accuracy and faster processing speed.

## Result
![](results/result.gif)

## Platform
+ GTX-1080Ti
+ Ubuntu

## Dependencies

+ Anaconda
+ OpenCV
+ Pytorch
+ Numpy

## How to run the code
```
python test_network.py [--input INPUT_VIDEO_PATH] [--output OUTPUT_VIDEO_PATH]
```
If you want to use your webcam, please set [--input "0"].

If you want to use mtcnn to detect face, please install [MTCNN](https://github.com/ipazc/mtcnn) and [Tensorflow](https://www.tensorflow.org/install) and run the following code.
```
python test_network_mtcnn.py [--input INPUT_VIDEO_PATH] [--output OUTPUT_VIDEO_PATH]
```


## Datasets

The model provided in this repo is trained on 300W-LP. For more datasets used in our paper, please refer to the following links.

+ [300W-LP, AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)
+ [BIWI](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html)

