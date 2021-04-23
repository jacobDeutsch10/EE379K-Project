"""
    MobileNet-v1 model written in PyTorch
    CIFAR10 test dataset accuracy: 77.15%
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy,time
from math import ceil

class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetv1(nn.Module):
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNetv1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)
        self.num_classes = num_classes
        self.mask_dict=None
    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def remove_channel(input_model):
    '''
    Input: model
           description: which is the model to be indeed pruned 
    Ouput: new_model
           description: which is the new model generating by removing all-zero channels
    '''
    
    new_model = copy.deepcopy(input_model)
    score_list = torch.sum(torch.abs(new_model.conv1.weight.data), dim=(1,2,3))
    next_layer_score_list = torch.sum(torch.abs(new_model.layers[0].conv1.weight.data), dim=(1,2,3))
    score_list = score_list * next_layer_score_list
    out_planes_num = int(torch.count_nonzero(score_list))
    out_planes_idx = torch.squeeze( torch.nonzero(score_list, as_tuple=False))
    conv1_wgt=copy.deepcopy(new_model.conv1.weight.data)
    new_model.conv1 = nn.Conv2d(3, out_planes_num, kernel_size=3, stride=1, padding=1, bias=False)
    new_model.bn1 = nn.BatchNorm2d(out_planes_num)
    new_model.conv1.weight.data[:,:,:,:] = conv1_wgt[out_planes_idx,:,:,:]

    in_planes_num = out_planes_num
    in_planes_idx = out_planes_idx
    for i,block in enumerate(new_model.layers):
        conv1_wgt=copy.deepcopy(block.conv1.weight.data)
        new_model.layers[i].conv1 = nn.Conv2d(in_planes_num, in_planes_num, kernel_size=3, stride=block.stride, padding=1, groups=in_planes_num, bias=False)
        new_model.layers[i].bn1 =  nn.BatchNorm2d(in_planes_num)          
        new_model.layers[i].conv1.weight.data[:,:,:,:] = conv1_wgt[in_planes_idx,:,:,:]
        score_list = torch.sum(torch.abs(block.conv2.weight.data), dim=(1,2,3))
        if i <len(new_model.layers)-1:
            next_layer_score_list = torch.sum(torch.abs(new_model.layers[i+1].conv1.weight.data), dim=(1,2,3))
            score_list = score_list * next_layer_score_list
        out_planes_num = int(torch.count_nonzero(score_list))
        out_planes_idx = torch.squeeze( torch.nonzero(score_list, as_tuple=False))
        conv2_wgt=copy.deepcopy(block.conv2.weight.data)
        new_model.layers[i].conv2 = nn.Conv2d(in_planes_num, out_planes_num, kernel_size=1, stride=1, padding=0, bias=False)
        new_model.layers[i].bn2 = nn.BatchNorm2d(out_planes_num)


        for idx_out,n in enumerate(out_planes_idx):

            new_model.layers[i].conv2.weight.data[idx_out,:,:,:] = conv2_wgt[n,in_planes_idx,:,:]
        in_planes_num = out_planes_num
        in_planes_idx = out_planes_idx
    lin_wgt=copy.deepcopy(new_model.linear.weight.data)
    lin_bias=copy.deepcopy(new_model.linear.bias.data)
    new_model.linear = nn.Linear(in_planes_num, new_model.num_classes)


    new_model.linear.weight.data = lin_wgt[:,out_planes_idx]    
    new_model.linear.bias.data = lin_bias
    return new_model
    
def calc_l1_norms(layer):
    norms = torch.abs(torch.sum(layer, dim=(1,2,3)))
    return norms

def mask_layers(model, fraction):

    norms = calc_l1_norms(model.conv1.weight.data)
    norms_idx = sorted([(i, norm) for i, norm in enumerate(norms)], key = lambda x: x[1])
    shape = model.conv1.weight.data.shape[1:]
    for i in range(int(ceil(len(norms_idx)*fraction))):
        model.conv1.weight.data[norms_idx[i][0],:,:,:] = torch.zeros(shape)

    for layer_idx, layer in enumerate(model.layers):
        """
        #This block prunes the depthwise Conv layer. adding this is the only way I can obtain a similar amount of param reduction to demo
        norms = calc_l1_norms(layer.conv1.weight.data)
        norms_idx = sorted([(i, norm) for i, norm in enumerate(norms)], key = lambda x: x[1])
        shape = layer.conv1.weight.data.shape[1:]
        for i in range(int(ceil(len(norms_idx)*fraction))):
            layer.conv1.weight.data[norms_idx[i][0],:,:,:] = torch.zeros(shape)
        """
        norms = calc_l1_norms(layer.conv2.weight.data)
        norms_idx = sorted([(i, norm) for i, norm in enumerate(norms)], key = lambda x: x[1])
        shape = layer.conv2.weight.data.shape[1:]
        for i in range(int(ceil(len(norms_idx)*fraction))):
            layer.conv2.weight.data[norms_idx[i][0],:,:,:] = torch.zeros(shape)

    return model

def channel_fraction_pruning(model, fraction=0.2):

    model = mask_layers(model, fraction)

    return model


