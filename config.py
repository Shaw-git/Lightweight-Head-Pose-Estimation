import os
import torch
# from ExtractorList import  extractor_list
from utils import mk_train_dir, tensor_board, on_server, softCrossEntropy
from networks import Network_d9 as Network
cudaID = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = cudaID
GPU = torch.cuda.device_count()
# focused parameters
feature_extractor = "network_d9_ablation"
#data augment
flip = True
delta_scale=0.2  #0.2 or 0
downsample=2  #2 or 0
rotation = 0
padding = 20

alpha = 1
beta = 1
DSFD=True
norm = False

lr = 0.001
num_epochs = 70
milestone = [30, 50] if num_epochs>=60 else [25, 40]

angle_limit=[-99,99]
orient_restrain = -1
step = 5

num_bins = 66
M=99 if num_bins!=67 else 100.5
# only valid when num_bins = 67
data_dropout=0

sum_loss=False
resolution=1
pose_mode ="euler"
pretrain=False
T = -1
drop_rate=0
# set train dataset and test dataset
train_dataset = "300W_LP"
test_dataset=   ["AFLW2000" ,"BIWI"]
train_test = -1
train_filter = [0,1,2]
if train_test in train_filter:
    train_filter.remove(train_test)

# test = "BIWI"
subject = [i for i in range(1, 25)]
# train parameter
# cudnn.enabled = True

batch_size = 16
decay = 0.1
workers = 3
weight_decay = 5e-4
# Data Augment
min_error_BIWI = [8, 8, 8]
min_error_AFLW2000 = [8, 8, 8]
print_step=10
in_memory=on_server()
# logger path initial
logger_path=mk_train_dir(os.path.join("./log",feature_extractor))
#  Initialize tensorBoardX
is_tensorboard_available, writer = tensor_board(logger_path)
# write train message
message = "\n"

for i, name in enumerate(test_dataset):
    f = open(os.path.join(logger_path, name+ ".txt"), "w")
    f.write(message)
    f.write("Train Dataset : "+ train_dataset+"\n")
    f.write("Feature Extractor :  "+feature_extractor+"\n")
    f.write("Alpha : %f\n" % (alpha))
    f.write("Beta : %f\n" % (beta))
    f.write("DSFD : %f\n" % (DSFD))
    f.write("Bins : %f\n" % (num_bins))
    f.write("Pretrain : %f\n" %(pretrain))
    f.write("Padding : %f\n" %(padding*resolution))
    f.write("Orient : %f\n" % (orient_restrain))
    f.write("Drop Rate : %f\n" % (drop_rate))
    f.write("LR : %f\n" % (lr))
    f.write("Epochs : %d\nBatch Size : %d    \nRotation : %d    \nFlip  :  %d  \nWeight Decay : %f   \nMilestones:[" % (
        num_epochs, batch_size, rotation, flip, weight_decay))
    for i in milestone:
        f.write(" %d " % (i))
    f.write("]\n")
    f.write("Delta Scale : %f\n"%(delta_scale))
    f.write("Downsample : %f\n" % (downsample))
    f.write("Geo Mode : %d\n" % (pose_mode=='geo'))
    f.write("SUM Loss : %d\n" % (sum_loss))
    f.write("Resolution : %.3f\n" % (resolution*224))
    f.write("Angle Limit: %d\n"%(angle_limit[1]))
    f.write("300W_LP Test: %d\n" % (train_test)) if train_test in train_filter else 0
    f.write("Dpout: %d\n" % (data_dropout)) if data_dropout>0  else 0
    f.write("T: %d\n" % (T)) if T>0 else 0
    f.write("Step: %d\n" % (step))
    f.write("Norm: %d\n" % (norm))
    f.close()
