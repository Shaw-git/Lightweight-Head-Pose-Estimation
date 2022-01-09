import os
import numpy as np
import cv2
import random
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from utils import tools
import copy
import imutils
from PIL import Image
from utils import euler_to_geo_coordinate, geo_coordinate_to_euler
import math



class BaseFunc():
    def load_300W_LP (self,dir, components, images, poses,  in_memory=False, orient=[],  angle_limit=[-99,99]):
        print("Loading 300W_LP dataset")
        for component in components:
            with open(os.path.join(dir,component,"label.txt")) as f:
                for line in f:
                    line=line.strip().split()
                    pose=np.array(line[1:4],np.float32)
                    if not self.is_limited(pose,angle_limit):
                        continue
                    orient.append(self.head_orient(pose))
                    img= cv2.imread(os.path.join(dir,component,line[0])) if in_memory  else os.path.join(dir,component,line[0])
                    images.append(img)
                    poses.append(pose)

    def filter_300W_LP(self, images, poses, orient, category, filter):
        new_images=[]
        new_poses = []
        new_orient = []
        assert len(images)==len(poses) and len(images) == len(category)
        for  i , c in enumerate(category):
                if c in filter:
                    new_images.append(images[i])
                    new_poses.append(poses[i])
                    new_orient.append(orient[i])
        return new_images, new_poses, new_orient

    def load_confidence(self, path= "./300W_LP_CL", T=3):
        fin = np.load(os.path.join(path, "s_confidence.npz"))
        confidence = fin [ "confidence" ]
        confidence = np.mean(confidence, axis=1)
        confidence = np.power(confidence,2)/T
        confidence = np.exp(confidence*-1)
        return confidence


    def load_AFLW2000(self,data_dir,images, poses, in_memory=False, angle_limit=[-99,99]):
        print("Loading AFLW2000 dataset")
        with open(os.path.join(data_dir,"label.txt")) as f:
            for line in f:
                line=line.strip().split()
                pose=np.array(line[1:4], np.float32)

                if not self.is_limited(pose,angle_limit):
                    continue
                img = cv2.imread(os.path.join(data_dir, line[0])) if in_memory else os.path.join(data_dir, line[0])
                images.append(img)
                poses.append(pose)


    def load_BIWI(self,data_dir, objects , in_memory, norm):
        print("Loading BIWI dataset")
        imgs=[]
        p =[]
        for object in objects:
            images, poses = self.npz_load_BIWI(os.path.join(data_dir, "%02d" % (object)), in_memory, norm)
            imgs = imgs + images
            p = p + poses
        return imgs,p

    def _load_BIWI(self, filename_path, in_memory ):
        txt_path = os.path.join(filename_path, "pose.txt")
        label = []
        images = []
        with open(txt_path) as fin:
            for i, line in enumerate(fin):
                line = line.strip().split()
                label.append([float(line[0]), float(line[1]), float(line[2])])
                img_path=os.path.join(filename_path, "%d.png" % i)
                images.append(cv2.imread(img_path) if in_memory else img_path)

        return images, label

    def npz_load_BIWI(self, filename_path, in_memory, norm ):

        fin = np.load(filename_path + ".npz")
        pose = fin["pose"]
        norm_pose = fin["norm_pose"]

        label = norm_pose.tolist() if norm else pose.tolist()
        images = []
        for i in range(len(label)):
            img_path = os.path.join(filename_path, "%d.png" % i)
            images.append(cv2.imread(img_path) if in_memory else img_path)

        return images, label


    def is_limited(self,pose, angle_limit):
        limited = True
        for i in range(3):
            limited = limited and angle_limit[0] < pose[i] and pose[i] < angle_limit[1]
        return limited

    def head_orient(self, headpose):
        headpose=np.asarray(headpose)*-1
        R = tools.EulerToMatrix(headpose[0], headpose[1], headpose[2])
        angle=np.arccos(R[2,2])*180/np.pi
        return angle

    def random_rotation(self, img, pose, rotate):
        pose = copy.copy(pose)
        for s in range(10):
            angle = random.randint(-abs(rotate), abs(rotate))
            pose_temp = tools.pose_rotate(pose, angle)
            if -99 < pose_temp[0] < 99 and -99 < pose_temp[1] < 99 and -99 < pose_temp[2] < 99:
                pose = pose_temp
                img = imutils.rotate(img, angle) # imutils.rotate is not inplace
                break
        return img,pose

    def random_scale(self,img,d_scale=0.2):
        shape=img.shape
        scale = 1+(random.random()*2-1)*d_scale
        new_shape=np.asarray(shape,dtype=np.float)*scale
        img = cv2.resize(img, (int(new_shape[1]), int(new_shape[0])))
        d=np.abs(new_shape-shape)/2
        if scale>1:
            return img[int(d[0]):int(d[0])+int(shape[0]) , int(d[1]):int(d[1])+int(shape[1])]
        else:
            new_img = np.zeros(shape,dtype=np.uint8)
            new_img[int(d[0]):int(d[0])+int(new_shape[0]) , int(d[1]):int(d[1])+int(new_shape[1])]=img
            return new_img

    def random_downsample(self, img, exp=2):
        shape = img.shape
        scale = pow(0.5, random.randint(0,exp))
        new_shape = np.asarray(shape, dtype=np.float) * scale
        img = cv2.resize(img, (int(new_shape[1]), int(new_shape[0])))
        img = cv2.resize(img, (int(shape[1]), int(shape[0])))
        return img

    def random_flip(self, img, pose, ratio):
        pose = copy.copy(pose)
        if self.flip and random.randint(0, 1) < ratio:
            img = cv2.flip(img, 1)
            pose[0] = -pose[0]
            pose[1] = -pose[1]
        return img, pose

    def data_augment(self,img,pose, flip, delta_scale, downsample, rotation):
        if flip:
            img, pose = self.random_flip(img, pose, 0.5)
        if delta_scale != 0:
            img = self.random_scale(img, self.delta_scale)
        if downsample != 0:
            img = self.random_downsample(img, exp=self.downsample)
        if rotation > 0:
            img, pose = self.random_rotation(img, pose, self.rotation)
        return img,pose

class Data300W_LP(Dataset,BaseFunc):
    def __init__(self,  data_dir ,transform=None,components = ["AFW","HELEN","IBUG","LFPW"],
                 image_mode='RGB',in_memory=False,  num_bins=67 , M= 100.5, pose_mode='euler',
                 random_flip=False,delta_scale=0, downsample=0,rotation=0, ori=False, angle_limit=[-99,99], filter=[0,1,2], T=9, step = 10):

        self.images = []
        self.poses = []
        self.orient=[]
        self.data_dir = data_dir
        self.transform = transform
        self.components=components

        self.pose_mode=pose_mode
        self.image_mode = image_mode
        self.in_memory = in_memory

        self.delta_scale=delta_scale
        self.downsample = downsample
        self.rotation = rotation
        self.flip = random_flip
        self.ori=ori
        self.step =step
        bin_interval=M*2/num_bins
        self.bins = [i*bin_interval-M for i in range(1, num_bins)]
        self.load_300W_LP(self.data_dir, self.components,self.images, self.poses, self.in_memory, orient=self.orient, angle_limit=angle_limit)
        self.confidence = self.load_confidence(T=T)
        self.sort = np.argsort(self.confidence)
        assert len(self.images) == len(self.confidence)
        self.poses=np.asarray(self.poses)
        self.length=len(self.images)

        # temps=[]
        # for i in range(self.length):
        #     orient = orient_nonlinear( self.orient[i], self.ori, self.step)
        #     temps.append(orient)
        #
        # self.sort = np.argsort(temps)

    def __getitem__(self, index):
        # index = self.sort[index]
        # print(index)

        img = self.images[index] if self.in_memory else cv2.imread(self.images[index])
        pose = copy.copy(self.poses[index])
        orient = self.orient[index]
        # conf = self.confidence[index]

        img, pose =self.data_augment(img,pose,self.flip,self.delta_scale, self.downsample, self.rotation)
        pose = pose if self.pose_mode =='euler' else euler_to_geo_coordinate(pose)
        # Bin values
        binned_pose = np.digitize(pose, self.bins)
        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor(pose)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            img = self.transform(img)

        if self.ori>0:
            orient = orient_nonlinear( orient, self.ori, self.step )
            return img, labels, cont_labels, orient
        else:
            return img, labels, cont_labels, np.float32(1)

    def __len__(self):
        return self.length

class AFLW2000(Dataset,BaseFunc):
    def __init__(self,data_dir, transform=None, angle_limit=[-99,99] ,
                 image_mode='RGB',in_memory=False,  num_bins=67 , M= 100.5, pose_mode='euler',
                 random_flip=False, delta_scale=0, downsample=0, rotation=0):
        self.images = []
        self.poses = []
        self.data_dir=data_dir
        self.transform = transform
        self.angle_limit = angle_limit

        self.pose_mode = pose_mode
        self.image_mode = image_mode
        self.in_memory = in_memory

        self.delta_scale = delta_scale
        self.downsample = downsample
        self.rotation = rotation
        self.flip = random_flip

        bin_interval = M * 2 / num_bins
        self.bins = [i * bin_interval - M for i in range(1, num_bins)]

        self.load_AFLW2000(self.data_dir, self.images, self.poses, self.in_memory,self.angle_limit)
        self.length = len(self.images)

    def __getitem__(self, index):

        img = self.images[index] if self.in_memory else cv2.imread(self.images[index])
        pose = copy.copy(self.poses[index])
        img, pose = self.data_augment(img, pose, self.flip, self.delta_scale, self.downsample, self.rotation)
        pose = pose if self.pose_mode == 'euler' else euler_to_geo_coordinate(pose)
        # Bin values
        binned_pose = np.digitize(pose, self.bins)
        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor(pose)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            img = self.transform(img)
        return img, labels, cont_labels, np.float32(1)

    def __len__(self):
        # 15,667
        return self.length


class BIWI(Dataset, BaseFunc):
    def __init__(self,data_dir, objects,transform=None, angle_limit=[-99,99] ,
                 image_mode='RGB',in_memory=False, num_bins=67 , M= 100.5, pose_mode='euler',
                 random_flip=False, delta_scale=0, downsample=0, rotation=0, ori=-1, step = 10, norm = True):
        dirname = "BIWI_Track" if norm else "BIWI_Raw"
        self.images=[]
        self.poses=[]
        self.orient = []
        self.objects= objects
        self.data_dir = os.path.join(data_dir, dirname)
        self.transform = transform
        self.angle_limit = angle_limit

        self.pose_mode = pose_mode
        self.image_mode = image_mode
        self.in_memory = in_memory

        self.delta_scale = delta_scale
        self.downsample = downsample
        self.rotation = rotation
        self.flip = random_flip
        self.ori = ori
        self.step = step

        bin_interval = M * 2 / num_bins
        self.bins = [i * bin_interval - M for i in range(1, num_bins)]

        self.images, self.poses = self.load_BIWI(self.data_dir,self.objects, self.in_memory,  norm= norm)
        self.length = len(self.images)

    def __getitem__(self, index):
        img = self.images[index] if self.in_memory else cv2.imread(self.images[index])
        pose = copy.copy(self.poses[index])
        img, pose = self.data_augment(img, pose, self.flip, self.delta_scale, self.downsample, self.rotation)
        pose = pose if self.pose_mode == 'euler' else euler_to_geo_coordinate(pose)
        # Bin values
        binned_pose = np.digitize(pose, self.bins)
        labels = torch.LongTensor(binned_pose)
        cont_labels = torch.FloatTensor(pose)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            img = self.transform(img)

        if self.ori>0:
            orient = orient_nonlinear( self.head_orient(pose), self.ori, self.step )
            return img, labels, cont_labels, orient
        else:
            return img, labels, cont_labels, np.float32(1)

    def __len__(self):
        # 15,667
        return self.length

def euler_to_axis_angle(euler_angle):
    euler_angle=-np.asarray(euler_angle)
    R = tools.EulerToMatrix(euler_angle[0], euler_angle[1], euler_angle[2])
    axes_xyz=[]
    print(R)
    for i in range(3):
        axes_xyz.append(np.arccos(R[i,i])/np.pi*180)
    return np.asarray(axes_xyz)

def orient_nonlinear(para, point ,step =10):
    if para < point:
        out = 1
    else:
        out = np.power(0.5, (para -point)/step)
    return np.float32(out)


# if __name__ == '__main__':

#     path_300W_LP="/home/ubuntu/ssd/datasets/300W_LP"
#     path_AFLW2000="/home/ubuntu/ssd/datasets/AFLW2000"
#     path_BIWI = "/home/ubuntu/ssd/datasets"

#     transform = transforms.Compose([transforms.Resize(240),
#                                                 transforms.RandomCrop(224),
#                                                ])

#     # data = AFLW2000( data_dir=path_AFLW2000, rotation=0, random_flip=False, in_memory=False,downsample=2,delta_scale=0.2)
#     data = Data300W_LP(data_dir=path_300W_LP, rotation=0, random_flip=False, in_memory=False ,downsample=0,delta_scale=0, ori=60,  filter=[ 0,1,2])
#     # print(len(data))
#     # data=BIWI(data_dir=path_BIWI,objects=[1], in_memory=False, norm=True)
#     # start = time.time()

#     k=0
#     for i in range(len(data)):

#         [image, pose, cons_pose,orient]=data[i]
#         # if cons_pose[2]>85:
#         #     k+=1
#         # if orient>60:
#         #     continue
#         print(cons_pose, orient)
#         # cons_pose = geo_coordinate_to_euler(cons_pose)
#         # print(euler_to_axis_angle(cons_pose))
#         img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#         # img = tools.drawAxis(img, cons_pose[0],cons_pose[1], cons_pose[2], landmarks=[])
#         # img = data.random_downsample(img,exp=0)
#         # img = data.random_scale(img, 0.2)
#         cv2.imshow("image", img)
#         # print(orient)
#         key = cv2.waitKey(0)&0xff
#         if key ==ord('s'):
#             cv2.imwrite("./bad_image/%d.png"%(i), img)

#     # print("%.3f s" % (time.time() - start))


