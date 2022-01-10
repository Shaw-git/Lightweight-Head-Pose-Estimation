import cv2
import numpy as np
import torch
from network.network import Network
from torchvision import transforms
from PIL import Image
from utils import load_snapshot
from utils.camera_normalize import drawAxis
import time
import argparse

def parse_option():
    parser = argparse.ArgumentParser('Please set input path and output path', add_help=False)
    parser.add_argument('--input', type=str, default = "0", help="the path of input stream")
    parser.add_argument('--output', type=str, default = "", help='where to save the result')
    args = parser.parse_args()

    return args


def scale_bbox(bbox, scale):
    w = max(bbox[2], bbox[3]) * scale
    x= max(bbox[0] + bbox[2]/2 - w/2,0)
    y= max(bbox[1] + bbox[3]/2 - w/2,0)
    return np.asarray([x,y,w,w],np.int64)

def main():
    args = parse_option()
    if len(args.input)==1:
        if ord('0')<=ord(args.input) and ord(args.input)<=ord('9'):
            cap = cv2.VideoCapture(int(args.input))
        else:
            print("invalid input path")
            exit()
    else:
        cap = cv2.VideoCapture(args.input)

    
    outstream = None
    if args.output != "":
        frame_size = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        outstream = cv2.VideoWriter(args.output, 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         25, frame_size)

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
        if outstream is not None:
            outstream.write(frame)

        key = cv2.waitKey(1)
        if key==27 or key == ord("q"):
            break
        count+=1
    if outstream is not None:
        outstream.release()


if __name__ == '__main__':
    main()