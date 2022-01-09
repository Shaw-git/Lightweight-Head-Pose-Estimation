import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# from matplotlib import pyplot as plt
class softCrossEntropy(nn.Module):
    def __init__(self,weight=[0.1, 0.2, 0.5, 0.2, 0.1], size_average=True):
        super(softCrossEntropy, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.weight = torch.FloatTensor(weight)
        self.size_average=size_average
        self.d = int(len(self.weight - 1) / 2)
        return
    def soft_cross_entropy(self, inputs, target):
        target_vector = torch.zeros(inputs.shape)
        for i, idx in enumerate(target):
            target_vector[i, idx - self.d: idx + self.d + 1] = self.weight
        if inputs.is_cuda:
            target_vector=target_vector.cuda()
        if self.size_average:
            return torch.mean(torch.sum(-target_vector * self.logsoftmax(inputs), dim=1))
        else:
            return torch.sum(torch.sum(-target * self.logsoftmax(inputs), dim=1))

    def forward(self, inputs, target):
        return self.soft_cross_entropy(inputs,target)

def orient_loss(weight,orient):
    binary_orient=[]
    for ori in orient:
        if ori>60:
            binary_orient.append(1)
        else:
            binary_orient.append(0)
    binary_orient=torch.LongTensor(binary_orient)
    weight = weight.squeeze()
    if weight.is_cuda:
        binary_orient=binary_orient.cuda()
    return F.cross_entropy(weight,binary_orient)


def get_ignored_params(model):
    # Generator function that yields ignored params.
    b = [model.conv1, model.bn1, model.fc_finetune]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param


def get_non_ignored_params(model):
    # Generator function that yields params that will be optimized.
    b = [model.layer1, model.layer2, model.layer3, model.layer4]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            if 'bn' in module_name:
                module.eval()
            for name, param in module.named_parameters():
                yield param

def get_fc_params(model):
    # Generator function that yields fc layer params.
    b = [model.fc_yaw, model.fc_pitch, model.fc_roll]
    for i in range(len(b)):
        for module_name, module in b[i].named_modules():
            for name, param in module.named_parameters():
                yield param

def count_model_params(model):
    return "Model Size:  %.4f  M " %(sum(p.numel() for p in model.parameters())/1e6)

def params_distribution(params, milestone):
    length = len(params)
    params =np.abs(params)
    for i in range(len(milestone)+1):
        if i==0:
            print("0 < value < %f  :    %.4f " %(milestone[i], sum(params<milestone[i])/length*100) )
            continue
        if i==len(milestone):
            print("%f  < value       :    %.4f " % (milestone[i-1], sum(params >milestone[i-1] )/length*100))
            continue
        print("%f  < value < %f  :    %.4f " % (milestone[i-1], milestone[i], sum((params > milestone[i-1])&(params < milestone[i]))/length*100  ))


def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


class calculate_errors():
    def __init__(self):
        self.roll = torch.empty(0)
        self.yaw = torch.empty(0)
        self.pitch = torch.empty(0)

    def clear(self):
        self.roll = torch.empty(0)
        self.yaw = torch.empty(0)
        self.pitch = torch.empty(0)

    def append(self, roll, yaw, pitch):
        self.roll = torch.cat([self.roll, roll.detach().cpu()], dim=0)
        self.yaw = torch.cat([self.yaw, yaw.detach().cpu()], dim=0)
        self.pitch = torch.cat([self.pitch, pitch.detach().cpu()], dim=0)

    def out(self):
        average_roll = torch.mean(torch.abs(torch.FloatTensor(self.roll))).item()
        average_yaw = torch.mean(torch.abs(torch.FloatTensor(self.yaw))).item()
        average_pitch = torch.mean(torch.abs(torch.FloatTensor(self.pitch))).item()
        return [average_roll, average_yaw, average_pitch]


def save_model(model, path, name):
    print('Taking snapshot...')
    name = path + "/" + name + ".pkl"
    torch.save(model.state_dict(), name)

def mk_train_dir(logger_path):
    if not os.path.exists(logger_path):
        os.mkdir(logger_path)
    for i in range(10000):
        new_path = os.path.join(logger_path, "%d" % i)
        if os.path.exists(new_path):
            continue
        else:
            os.mkdir(new_path)
            os.mkdir(os.path.join(new_path,"test_results"))
            os.mkdir(os.path.join(new_path, "models"))
            return new_path

def save_pred_label(file, pred,labels):
    for i in range(len(labels[0])):
        file.write("%.3f %.3f %.3f || %.3f %.3f %.3f \n" % (pred[0][i], pred[1][i], pred[2][i],labels[0][i], labels[1][i],labels[2][i]))

def add_test_msg(path, name ,errors, epoch, show=True):
    file=open(os.path.join(path,name+".txt"),'a')
    msg = name + "    Epoch:    %d    Roll:   %f   Yaw:  %f   Pitch:  %f     MAE:  %f    SUM:  %f  \n" % (epoch,
        errors[0], errors[1], errors[2], np.mean(errors), np.sum(errors))
    file.write(msg)
    file.close()
    if show:
        print(msg)

def tensor_board(logger_path):
    try:
        from tensorboardX import SummaryWriter
        is_tensorboard_available = True
        writer = SummaryWriter(log_dir=logger_path)
        print("--tensorBoard exit : log path - %s" % (logger_path))
        return is_tensorboard_available, writer
    except Exception:
        is_tensorboard_available = False
        return is_tensorboard_available, None

def on_server():
    if os.path.exists("/home/ubuntu/ssd/datasets"):
        return False
    return True

def get_data_path(DSFD=False):
    if os.path.exists("/home/ubuntu/ssd/datasets"):
        datafile="/home/ubuntu/ssd/datasets"
    elif os.path.exists("/home/zhangdong/Shaw/datasets"):
        datafile="/home/zhangdong/Shaw/datasets"
    path_300W_LP = os.path.join(datafile,"300W_LP")
    path_AFLW2000 = os.path.join(datafile,"AFLW2000_DSFD" if DSFD else "AFLW2000")
    path_BIWI =  datafile
    return {"300W_LP":path_300W_LP, "AFLW2000": path_AFLW2000, "BIWI": path_BIWI}

def write_tensorboard(writer, dataset_name,  errors, iter, lr=-1):
    writer.add_scalar(dataset_name + " Roll", errors[0], iter)
    writer.add_scalar(dataset_name + " Yaw", errors[1], iter)
    writer.add_scalar(dataset_name + " Pitch", errors[2], iter)
    writer.add_scalar(dataset_name + " Sum", np.sum(errors), iter)
    writer.add_scalar(dataset_name + " Average",np.mean(errors), iter)
    if lr>0:
        writer.add_scalar(dataset_name +" Learn rate", lr, iter)

def adjust_learning_rate(optimizer, epoch, schedule, gamma):
    """Sets the learning rate to the initial LR decayed by schedule"""
    if epoch in schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    return optimizer.state_dict()['param_groups'][0]['lr']

def is_minor_error(min_errors,errors):
    if errors[0] < min_errors[0] or errors[1] < min_errors[1] or errors[2] < min_errors[
        2] or np.sum(errors) < np.sum(min_errors):
        min_errors[0] = min(errors[0], min_errors[0])
        min_errors[1] = min(errors[1], min_errors[1])
        min_errors[2] = min(errors[2], min_errors[2])
        return True
    return False

def load_snapshot(model, snapshot):
    saved_state_dict = torch.load(snapshot,map_location=torch.device('cpu'))
    model.load_state_dict(saved_state_dict)
