import torch.nn as nn
import torch
from torch import autograd
from torch.nn import *
from torchsummary import summary
# from ipdb import set_trace
class SpatialAttention(nn.Module):
    def __init__(self,  out_ch):
        super(SpatialAttention, self).__init__()
        self.conv1x1_1 = nn.Conv3d(out_ch * 2, out_ch, 1, padding=0, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.conv1x1_2 = nn.Conv3d(2, 1, 1, padding=0, bias=False)
        self.conv1x1_3 = nn.Conv3d(2, 1, 1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c1,c2):
        cat1 = torch.cat([c1, c2], dim=1)

        conv1x1_1 = self.conv1x1_1(cat1)
        avg_out = torch.mean(conv1x1_1, dim=1, keepdim=True)
        max_out, _ = torch.max(conv1x1_1, dim=1, keepdim=True)
        conv1x1_1 = torch.cat([avg_out, max_out], dim=1)
        conv1x1_2 = self.sigmoid(self.conv1x1_2(conv1x1_1))
        conv1x1_3 = self.sigmoid(self.conv1x1_3(conv1x1_1))

        cat1 = torch.cat([c1 * conv1x1_2, c2 * conv1x1_3], dim=1)
        return  cat1

class channelatt(nn.Module):
    def __init__(self,  out_ch):
        super(channelatt, self).__init__()
        self.conv = nn.Conv3d(out_ch * 2, out_ch , 1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Conv3d(out_ch, out_ch // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv3d(out_ch // 8, out_ch, 1, bias=False)
        self.fc3 = nn.Conv3d(out_ch // 8, out_ch, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input1,input2):
        cat1 = torch.cat([input1, input2], dim=1)

        c1 = self.relu1(self.conv(cat1))
        avg_pool = self.avg_pool(c1)
        c2 = self.relu1(self.fc1(avg_pool))
        c3 = self.sigmoid(self.fc2(c2))
        c4 = self.sigmoid(self.fc3(c2))

        cat1 = torch.cat([input1*c3, input2*c4], dim=1)
        return  cat1

class CBA(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CBA, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        self.batc1 = nn.BatchNorm3d(out_ch*2)
        self.act1 = nn.ReLU()
        for m in self.modules():
            if isinstance(m,nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm3d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self, input):
        c1 = self.conv1(input)
        c2 = self.conv2(input)
        cat1 = torch.cat([c1,c2],dim=1)
        bat1 = self.batc1(cat1)
        re1 = self.act1(bat1)
        return re1

class CBA_spatten(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CBA_spatten, self).__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        self.batc1 = nn.BatchNorm3d(out_ch*2)
        self.act1 = nn.ReLU()
        self.SpatialAttention = SpatialAttention(out_ch)
        self.conv3 = nn.Conv3d(out_ch * 2, out_ch * 2, 1, padding=0)

    def forward(self, input):
        # set_trace()
        c1 = self.conv1(input)
        c2 = self.conv2(input)
        #####################加空间注意力######################
        cat1 = self.SpatialAttention(c1, c2)
        bat1 = self.batc1(cat1)
        re1 = self.act1(bat1)
        return re1

class CBA_chatten(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CBA_chatten, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv3d(in_ch, out_ch, 5, padding=2)
        self.batc1 = nn.BatchNorm3d(out_ch*2)
        self.act1 = nn.ReLU()
        self.channelatt = channelatt(out_ch)
        self.SpatialAttention = SpatialAttention(out_ch)
        self.conv3 = nn.Conv3d(out_ch * 2, out_ch, 3, padding=1)
        self.conv4 = nn.Conv3d(out_ch * 2, out_ch, 5, padding=2)


    def forward(self, input):
        c1 = self.conv1(input)
        c2 = self.conv2(input)
        #####################加通道注意力#######################
        cat1 = self.channelatt(c1,c2)

        # import numpy as np
        # np.save('./c1.npy', c1.cpu().detach().numpy())
        # import os
        # os._exit(0)
        bat1 = self.batc1(cat1)
        re1 = self.act1(bat1)
        # #####################加通道、空间注意力 并行添加###############################
        # cat1 = self.channelatt(c1, c2)
        # cat2 = self.SpatialAttention(c1, c2)
        # cat1 = self.conv3(cat1+cat2)
        # bat1 = self.batc1(cat1)
        # re1 = self.act1(bat1)
        # #####################加通道、空间注意力 通空串连###############################
        # cat1 = self.channelatt(c1, c2)
        # c3 = self.conv3(cat1)
        # c4 = self.conv4(cat1)
        # cat2 = self.SpatialAttention(c3, c4)
        # bat1 = self.batc1(cat2)
        # re1 = self.act1(bat1)
        return re1


class Single16(nn.Module): #5 layers CNN branch
    def __init__(self, in_ch, out_ch, dropout=0.5):
        super(Single16, self).__init__()
        # print('in_ch', in_ch, out_ch)
        self.cba1 = CBA_chatten(in_ch, out_ch)
        self.pool1 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.cba2 = CBA_chatten(out_ch*2, out_ch)
        self.pool2 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.cba3 = CBA_chatten(out_ch*2, out_ch)
        self.pool3 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.cba4 = CBA_chatten(out_ch*2, out_ch)
        self.pool4 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.cba5 = CBA_chatten(out_ch*2, out_ch)
        self.pool5 = nn.MaxPool3d(kernel_size = (2, 2, 2))

        self.lin1 = nn.Linear(out_ch * 16, 128)
        self.lin2 = nn.Linear(128, 32)
        self.lin3 = nn.Linear(128, 1)
        self.re1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout=nn.Dropout(p=dropout)

    def forward(self, input):
        
        cb1 = self.cba1(input)
        p1 = self.pool1(cb1)
        cb2 = self.cba2(p1)
        p2 = self.pool2(cb2)
        cb3 = self.cba3(p2)
        p3 = self.pool3(cb3)
        cb4 = self.cba4(p3)
        p4 = self.pool4(cb4)
        cb5 = self.cba5(p4)
        p5 = self.pool5(cb5)

        l = p5.view(p5.size(0), -1)
        # l = self.dropout(l)
        l1 = self.lin1(l)
        r1 = self.re1(l1)
        r1 = self.dropout(r1)
        l2 = self.lin2(r1)
        r2 = self.re1(l2)
        r2 = self.dropout(r2)
        conv1 = l2
        l3 = self.lin3(r1)
        s1 = self.sigmoid(l3)
        return s1



class mymodel(torch.nn.Module):
    def __init__(self,in_channel=1):
        super(mymodel,self).__init__()
        
        self.single16 = Single16(in_channel,out_ch=32)


    def forward(self, adc):
        adc_output = self.single16(adc)
        return adc_output


if __name__ == "__main__":
    model = mymodel()
    # set_trace()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,':',param.size())
    print(model)
    summary(model,(1 ,64, 64, 64))

