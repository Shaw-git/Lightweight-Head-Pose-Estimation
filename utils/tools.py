import numpy as np
import math
import cv2

def EulerToMatrix(roll, yaw, pitch):
    # roll - z axis
    # yaw  - y axis
    # pitch - x axis
    roll = roll / 180 * np.pi
    yaw = yaw / 180 * np.pi
    pitch = pitch / 180 * np.pi

    Rz = [[math.cos(roll), -math.sin(roll), 0],
          [math.sin(roll), math.cos(roll), 0],
          [0, 0, 1]]

    Ry = [[math.cos(yaw), 0, math.sin(yaw)],
          [0, 1, 0],
          [-math.sin(yaw), 0, math.cos(yaw)]]

    Rx = [[1, 0, 0],
          [0, math.cos(pitch), -math.sin(pitch)],
          [0, math.sin(pitch), math.cos(pitch)]]

    matrix = np.matmul(Rx, Ry)
    matrix = np.matmul(matrix, Rz)

    return matrix

def cropFace(image,landmark,scale=1.2):

    center=np.mean(landmark,axis=0)
    size=(np.max(landmark,axis=0)[0]-np.min(landmark,axis=0)[0])*scale

    x0=center[0]-size/2
    x1=center[0]+size/2
    y0=center[1]-size/2
    y1=center[1]+size/2


    face=image[int(y0):int(y1),int(x0):int(x1)]

    return face

def drawAxis(img,  roll, yaw, pitch, landmarks=[] , size=100):

    if len(landmarks)>1:
        tdx=np.mean(landmarks[42:48],axis=0)[0]
        tdy=np.mean(landmarks[42:48],axis=0)[1]
    else:
        tdx=img.shape[1]/2
        tdy=img.shape[0]/2


    matrix = EulerToMatrix(-roll, -yaw, -pitch)

    Xaxis = np.array([matrix[0, 0], matrix[1, 0], matrix[2, 0]]) * size
    Yaxis = np.array([matrix[0, 1], matrix[1, 1], matrix[2, 1]]) * size
    Zaxis = np.array([matrix[0, 2], matrix[1, 2], matrix[2, 2]]) * size


    cv2.line(img, (int(tdx), int(tdy)), (int(Xaxis[0]+tdx), int(-Xaxis[1]+tdy)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(-Yaxis[0]+tdx), int(Yaxis[1]+tdy)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(Zaxis[0]+tdx), int(-Zaxis[1]+tdy)), (255, 0, 0), 2)

    return img

def pose_rotate(headpose,angle):
    #positive anticlockwise roation image
    HeadToCr = EulerToMatrix(-headpose[0], -headpose[1], -headpose[2])  # inverse(RzRyRx)=     R(-x)R(-y)R(-z)
    CrToCn = EulerToMatrix(angle,0,0)
    HeadToCn = np.matmul(CrToCn, HeadToCr)
    headpose = -1 * MatrixToEuler(HeadToCn)
    return headpose

def MatrixToEuler(M):

    yaw=math.asin(M[0,2])
    pitch = math.atan2(-M[1, 2], M[2, 2])
    roll =math.atan2(-M[0,1],M[0,0])
    return np.array([roll ,yaw ,pitch])*180/np.pi

def matrix_to_euler(M):

    if not (M[0, 2 ] == 1 or M[0, 2] == -1):
        yaw = math.asin(M[0, 2])
        yaw2=np.pi - yaw
        yaw2 = yaw2 - 2*np.pi if yaw<0 else yaw2

        pitch = math.atan2(-M[1, 2], M[2, 2])
        roll = math.atan2(-M[0, 1], M[0, 0])

        cos_yaw=math.cos(yaw2)
        pitch2 = math.atan2(-M[1, 2]/cos_yaw, M[2, 2]/cos_yaw)
        roll2 =  math.atan2(-M[0, 1]/cos_yaw, M[0, 0]/cos_yaw)

        return np.array([roll ,yaw ,pitch])*180/np.pi , np.array([roll2 ,yaw2 ,pitch2 ])*180/np.pi
    else:
        if M[0, 2] ==1:
            yaw = 90
            roll_plus_pitch = math.atan2(M[1,0], M[1,1])*180/np.pi
            roll = roll_plus_pitch
            pitch = 0
        else:
            yaw = -90
            roll_minus_pitch = math.atan2(M[1,0], M[1,1])*180/np.pi
            roll = roll_minus_pitch
            pitch =0
        return np.array([roll ,yaw ,pitch])*180/np.pi, np.array([roll ,yaw ,pitch])*180/np.pi


if __name__ == '__main__':
    import copy
    image =  np.ones((224,224,3))
    image[10:20]=0
    headpose = [10, 80,10]
    image_1 = drawAxis(copy.copy(image), headpose[0], headpose[1], headpose[2])
    for i in range(1000):
        new_pose = pose_rotate(headpose,i)
        image_2 = drawAxis(copy.copy(image), new_pose[0], new_pose[1], new_pose[2])
        print(headpose, new_pose)
        cv2.imshow("image", image_1)
        cv2.imshow("image_2", image_2)
        cv2.waitKey(0)

    # R = EulerToMatrix(headpose[0], headpose[1], headpose[2])
    # print(pose_rotate(headpose, 10))
    # print(matrix_to_euler(R))
