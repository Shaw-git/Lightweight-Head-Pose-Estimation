import numpy as np
import torch
from . import tools
from math import sin, cos, atan2
def euler_to_geo_coordinate(euler_angle):
    # rotate order :  intrinsic: x1 y2 z3 , extrinsic: z1 y2 x3 (roll, yaw, pitch)
    R = tools.EulerToMatrix(-euler_angle[0], -euler_angle[1], -euler_angle[2])
    z_axis = R[ :, 2]
    x_axis = R[ :, 0]

    latitude = np.arcsin(z_axis[1])
    longitude = atan2(z_axis[0],z_axis[2])

    norm_x=np.cross([0,1,0] ,[z_axis[0], 0 ,z_axis[2]])
    norm_x = norm_x / np.linalg.norm(norm_x, ord= 2)
    norm_y = np.cross(z_axis, norm_x)
    norm_y = norm_y / np.linalg.norm(norm_y, ord=2)

    x= np.dot(norm_x, x_axis)
    y = np.dot(norm_y, x_axis)
    roll = np.arctan2(y , x)

    coordinate = np.asarray([roll, longitude, latitude])/np.pi*180
    return coordinate

def geo_coordinate_to_euler(coordinate):
    coordinate = np.asarray(coordinate)
    coordinate = coordinate/180*np.pi
    x = cos(coordinate[2]) *sin(coordinate[1])
    y = sin(coordinate[2])
    z = cos(coordinate[2]) *cos(coordinate[1])
    z_axis=[x,y, z]
    z_axis = z_axis / np.linalg.norm(z_axis, ord=2)

    norm_x = np.cross([0, 1, 0], [z_axis[0], 0, z_axis[2]])
    norm_x = norm_x / np.linalg.norm(norm_x, ord=2)

    norm_y = np.cross(z_axis, norm_x)
    norm_y = norm_y / np.linalg.norm(norm_y, ord=2)

    x_axis = cos(coordinate[0])*norm_x + sin(coordinate[0])*norm_y
    x_axis = x_axis/np.linalg.norm(x_axis ,2)
    y_axis = np.cross(z_axis, x_axis)
    R = np.asarray([x_axis, y_axis, z_axis]).transpose()

    return -1 * tools.MatrixToEuler(R)

def train_geo_euler(pred, label):
    pred = np.asarray([ p.tolist() for p in pred]).transpose()
    label = np.asarray([ p.tolist() for p in label]).transpose()
    for i in range(len(pred)):
        pred[i] = geo_coordinate_to_euler(pred[i])
        label[i] = geo_coordinate_to_euler(label[i])
    pred = pred.transpose()
    label =label.transpose()
    return torch.FloatTensor(pred), torch.FloatTensor(label)
