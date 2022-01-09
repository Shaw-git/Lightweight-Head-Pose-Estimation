import cv2
import numpy as np
import torch
from network.network import Network
from torchvision import transforms
from PIL import Image
from utils import load_snapshot
from utils.camera_normalize import drawAxis
import time

def scale_bbox(bbox, scale):
    w = max(bbox[2], bbox[3]) * scale
    x= max(bbox[0] + bbox[2]/2 - w/2,0)
    y= max(bbox[1] + bbox[3]/2 - w/2,0)
    return np.asarray([x,y,w,w],np.int64)

def main():
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
    pose_estimator = Network(bin_train=False)
    load_snapshot(pose_estimator,"./models/model-b66.pkl")
    pose_estimator = pose_estimator.eval()

    transform_test = transforms.Compose([transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
    count = 0
    last_faces = None
    while True:
        
        ret, frame = cap.read()
        if not ret:
            break
        if count % 5 == 0:
            gray_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, 1.2)
            if len(faces)==0 and (last_faces is not None):
                faces=last_faces
            last_faces = faces

        face_images = []
        face_tensors = []
        for i, bbox in enumerate(faces):
            x,y, w,h = scale_bbox(bbox,1.5)
            frame = cv2.rectangle(frame,(x,y), (x+w, y+h),color=(0,0,255),thickness=2)
            face_img = frame[y:y+h,x:x+w]
            face_images.append(face_img)
            pil_img = Image.fromarray(cv2.cvtColor(cv2.resize(face_img,(224,224)), cv2.COLOR_BGR2RGB))
            face_tensors.append(transform_test(pil_img)[None])

        if len(face_tensors)>0:
            with torch.no_grad():
                start = time.time()
                face_tensors = torch.cat(face_tensors,dim=0)
                roll, yaw, pitch = pose_estimator(face_tensors)
                print("inference time: %.3f ms/face"%((time.time()-start)/len(roll)*1000))
                for img, r,y,p in zip(face_images, roll,yaw,pitch):
                    headpose = [r,y,p]
                    drawAxis(img, headpose,size=50)

        cv2.imshow("Result", frame)
        key = cv2.waitKey(1)
        if key==27:
            break
        count+=1

main()