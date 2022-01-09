import cv2
import numpy as np
import math

def drawAcross(color, x, y, size=20,pigment=(255, 255, 255)):
    cv2.line(color, (int(x - size), int(y)), (int(x + size), int(y)), pigment, 2)
    cv2.line(color, (int(x), int(y - size)), (int(x), int(y + size)), pigment, 2)

def normalize_headpose(headpose,intrinsic,center):
    Rt = FromPixelToRotationMatrix(center, intrinsic)
    HeadToCr = EulerToMatrix(-headpose[0], -headpose[1], -headpose[2])  # inverse(RzRyRx)=     R(-x)R(-y)R(-z)
    CrToCn = Rt
    HeadToCn = np.matmul(CrToCn, HeadToCr)
    headpose = -1 * MatrixToEuler(HeadToCn)
    return headpose

def anti_normalize_headpose(headpose,intrinsic,center):
    Rt = FromPixelToRotationMatrix(center, intrinsic)
    HeadToCr = EulerToMatrix(-headpose[0], -headpose[1], -headpose[2])  # inverse(RzRyRx)=     R(-x)R(-y)R(-z)
    # CrToCn = Rt
    CnToCr = np.mat(Rt).I
    HeadToCn = np.matmul(CnToCr, HeadToCr)
    headpose = -1 * MatrixToEuler(HeadToCn)
    return headpose


def normalize_landmarks(landmarks,intrinsic,center):
    Rt = FromPixelToRotationMatrix(center, intrinsic)
    new_marks = []
    for p in landmarks:
        p = From_src_To_dst(p, Rt, intrinsic)
        new_marks.append(p)
    return new_marks


def normalize_glabel(glabel,intrinsic, center):

    Rt = FromPixelToRotationMatrix(center, intrinsic)
    CoToCn = Rt
    normalized_glabel = np.array(np.matmul(CoToCn, glabel))
    normalized_glabel = np.reshape(normalized_glabel,-1)
    return normalized_glabel

def anti_normalize_glabel(glabel, intrinsic, center):

    Rt = FromPixelToRotationMatrix(center, intrinsic)
    #CoToCn = Rt
    CnToCo = np.mat(Rt).I
    normalized_glabel = np.array(np.matmul(CnToCo, glabel))
    normalized_glabel = np.reshape(normalized_glabel, -1)
    return normalized_glabel

def warpFace(image, headpose=[],landmark=[],intrinsic=[],center=[]):

    if len(center)==0:
        center=np.mean(landmark,axis=0)

    Rt = FromPixelToRotationMatrix(center, intrinsic)
    src, dst = generate_src_dst(Rt, intrinsic)
    Mat = cv2.getPerspectiveTransform(src, dst)
    image = cv2.warpPerspective(image, Mat, dsize=(image.shape[1], image.shape[0]))
    new_mark = []

    if len(landmark)!=0:
        new_mark=[]
        for p in landmark:
            p = From_src_To_dst(p, Rt, intrinsic)
            new_mark.append(p)

    if len(headpose)!=0:
        HeadToCr=EulerToMatrix(-headpose[0],-headpose[1],-headpose[2]) #  inverse(RzRyRx)=     R(-x)R(-y)R(-z)
        CrToCn=Rt
        HeadToCn=np.matmul(CrToCn,HeadToCr)
        headpose=-1*MatrixToEuler(HeadToCn)

    return image,np.asarray(headpose),np.asarray(new_mark)

def generate_src_dst(RM,intrinsic):

    src = [[100, 100], [100, 200], [200, 100], [200, 200]]
    CT=[[1,0,0],[0,-1,0],[0,0,-1]] #Coordinate Tranform
    dst = []
    Cr_inverse=np.mat(intrinsic).I
    Cn=intrinsic
    for d in src:
        d=[d[0],d[1],1]
        d=np.array(np.matmul(Cr_inverse,d)).reshape(3)
        d = np.matmul(CT, d)
        d=  np.matmul(RM,d)
        d = np.matmul(CT, d)
        d=np.array(np.matmul(Cn,d)).reshape(3)
        d=d/d[2]
        dst.append(d[0:2])
    return np.array(src,dtype=np.float32),np.array(dst,dtype=np.float32)

def From_src_To_dst(p,Rt,intrinsic):

    CT=[[1,0,0],[0,-1,0],[0,0,-1]] #Coordinate Tranform
    Cr_inverse=np.mat(intrinsic).I
    Cn=intrinsic
    d=p
    d=[d[0],d[1],1]
    d=np.array(np.matmul(Cr_inverse,d)).reshape(3)
    d = np.matmul(CT, d)
    d=  np.matmul(Rt,d)
    d = np.matmul(CT, d)
    d=np.array(np.matmul(Cn,d)).reshape(3)
    d=d/d[2]
    return [d[0],d[1]]

def FromPixelToRotationMatrix(center, intrinsic): 

    def RotationToMatrix(axis, angle):
        axis = axis / np.sqrt(np.sum(np.power(axis, 2), axis=0))
        a = axis[0]
        b = axis[1]
        c = axis[2]    # CnToCo = np.mat(Rt).I
        angle = -angle

        M = [
            [a ** 2 + (1 - a ** 2) * np.cos(angle), a * b * (1 - np.cos(angle)) + c * np.sin(angle),
             a * c * (1 - np.cos(angle)) - b * np.sin(angle)],
            [a * b * (1 - np.cos(angle)) - c * np.sin(angle), b ** 2 + (1 - b ** 2) * np.cos(angle),
             b * c * (1 - np.cos(angle)) + a * np.sin(angle)],
            [a * c * (1 - np.cos(angle)) + b * np.sin(angle), b * c * (1 - np.cos(angle)) - a * np.sin(angle),
             c ** 2 + (1 - c ** 2) * np.cos(angle)]
        ]

        return np.array(M)

    px = center[0] - intrinsic[0, 2]
    py = center[1] - intrinsic[1, 2]
    horizon = px / intrinsic[0, 0]
    vertical = py / intrinsic[1, 1]
    Vector = np.array([horizon, -vertical, -1])
    Vector = Vector / np.sqrt(np.sum(np.power(Vector, 2), axis=0))
    zAxis = [0, 0, -1]
    rotate_axis = np.cross(Vector, zAxis)
    rotate_axis = rotate_axis / np.sqrt(np.sum(np.power(rotate_axis, 2), axis=0))
    rotata_angel = np.sum(Vector * zAxis)
    rotata_angel = np.arccos(rotata_angel)

    return RotationToMatrix(rotate_axis,rotata_angel)

def drawAxis(img, headpose, landmarks = None,  size=100):
    roll, yaw, pitch = headpose[0], headpose[1], headpose[2]
    if landmarks!=None:
        tdx=np.mean(landmarks[42:48],axis=0)[0]
        tdy=np.mean(landmarks[42:48],axis=0)[1]
    else:
        tdx = img.shape[1]/2
        tdy = img.shape[0]/2

    matrix = EulerToMatrix(-roll, -yaw, -pitch)

    Xaxis = np.array([matrix[0, 0], matrix[1, 0], matrix[2, 0]]) * size
    Yaxis = np.array([matrix[0, 1], matrix[1, 1], matrix[2, 1]]) * size
    Zaxis = np.array([matrix[0, 2], matrix[1, 2], matrix[2, 2]]) * size

    cv2.line(img, (int(tdx), int(tdy)), (int(Xaxis[0]+tdx), int(-Xaxis[1]+tdy)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(-Yaxis[0]+tdx), int(Yaxis[1]+tdy)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(Zaxis[0]+tdx), int(-Zaxis[1]+tdy)), (255, 0, 0), 2)

    return img

def draw_axis_from_mat(img,matrix, landmark=None , center=None, size=80):


    if landmark != None:
        tdx = np.mean(landmark[42:48], axis=0)[0]
        tdy = np.mean(landmark[42:48], axis=0)[1]

    if center != None:
        tdx = center[0]
        tdy = center[1]

    if landmark == None and center == None:
        tdx = int(img.shape[1] / 2)
        tdy = int(img.shape[0] / 2)

    Xaxis = np.array([matrix[0, 0], matrix[1, 0], matrix[2, 0]]) * size
    Yaxis = np.array([matrix[0, 1], matrix[1, 1], matrix[2, 1]]) * size
    Zaxis = np.array([matrix[0, 2], matrix[1, 2], matrix[2, 2]]) * size

    # A matrix to transform 3d point from OpenGL camera coordinate to Opencv camera coordinate
    m=[[1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]]

    Xaxis = np.matmul(m, Xaxis)
    Yaxis = np.matmul(m, Yaxis)
    Zaxis = np.matmul(m, Zaxis)

    cv2.line(img, (int(tdx), int(tdy)), (int(Xaxis[0] + tdx), int(Xaxis[1] + tdy)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(Yaxis[0] + tdx), int(Yaxis[1] + tdy)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(Zaxis[0] + tdx), int(Zaxis[1] + tdy)), (255, 0, 0), 2)

def drawAxis_test(img,  roll, yaw, pitch, landmarks , size=100):

    tdx=np.mean(landmarks[42:48],axis=0)[0]
    tdy=np.mean(landmarks[42:48],axis=0)[1]

    matrix = np.linalg.inv(EulerToMatrix_XYZ(roll, yaw, pitch))

    Xaxis = np.array([matrix[0, 0], matrix[1, 0], matrix[2, 0]]) * size
    Yaxis = np.array([matrix[0, 1], matrix[1, 1], matrix[2, 1]]) * size
    Zaxis = np.array([matrix[0, 2], matrix[1, 2], matrix[2, 2]]) * size


    cv2.line(img, (int(tdx), int(tdy)), (int(Xaxis[0]+tdx), int(-Xaxis[1]+tdy)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(-Yaxis[0]+tdx), int(Yaxis[1]+tdy)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(Zaxis[0]+tdx), int(-Zaxis[1]+tdy)), (255, 0, 0), 2)

    return img

def printWords(frame,roll,yaw,pitch,pos="left"):
    if pos=="left":
        frame = cv2.putText(frame, "roll: %f" % (roll), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = cv2.putText(frame, "yaw:  %f" % (yaw), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        frame = cv2.putText(frame, "pitch:%f" % (pitch), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    elif pos=="right":
        dx=frame.shape[1]-200
        dy=50
        frame = cv2.putText(frame, "roll: %f" % (roll), (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        dy+=30
        frame = cv2.putText(frame, "yaw:  %f" % (yaw), (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        dy += 30
        frame = cv2.putText(frame, "pitch:%f" % (pitch), (dx, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return frame

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

def EulerToMatrix_XYZ(roll, yaw, pitch):
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

    matrix = np.matmul(Rz, Ry)
    matrix = np.matmul(matrix, Rx)

    return matrix

def MatrixToEuler(M):

    yaw=math.asin(M[0,2])
    pitch = math.atan2(-M[1, 2], M[2, 2])
    roll =math.atan2(-M[0,1],M[0,0])
    return np.array([roll ,yaw ,pitch])*180/np.pi

def MatrixToEuler_XYZ(M):
    M=np.mat(M).I
    return MatrixToEuler(M)*-1


def HeadToCam_Matrix(roll, yaw, pitch):

    # roll - z axis
    # yaw  - y axis
    # pitch - x axis
    #
    roll = -roll / 180 * np.pi
    yaw = -yaw / 180 * np.pi
    pitch = -pitch / 180 * np.pi

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

def CamToHead_Matrix(roll, yaw, pitch):

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

    matrix = np.matmul(Rz, Ry)
    matrix = np.matmul(matrix, Rx)

    return matrix