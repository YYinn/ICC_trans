import torch.nn as nn
import torch
from torch import autograd
from torch.nn import *
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

        ########################## relu #######################################
        c1 = self.conv(cat1)
        # c1 = self.relu1(self.conv(cat1))
        
        avg_pool = self.avg_pool(c1)
        # c2 = self.relu1(self.fc1(avg_pool))
        c2 = self.fc1(avg_pool)
        c3 = self.fc2(c2)
        c3 = self.sigmoid(c3)
        # c3 = self.sigmoid(self.fc2(c2))
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


class Single16(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.5):
        super(Single16, self).__init__()
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
        return p1,p2,p3,p4,p5,conv1, s1


def norm_img(image): 
    image= ((image-image.min())/image.max())
    image[image<0] = 0
    image[image>1] = 1

    return image


class mymodel(torch.nn.Module):
    def __init__(self,dropout=0.3):
        super(mymodel,self).__init__()
        #
        self.single16 = Single16(1,out_ch=32)
        # self.mcf = MCF()
        # self.msf = MSF()
        # self.cba = CBA_chatten(in_ch=1*32, out_ch=16)
        # self.cba_cat = CBA_chatten(in_ch=1 , out_ch=16)
        self.cba_p5 = CBA(in_ch=32, out_ch=16)
        self.pool = nn.MaxPool3d(kernel_size = (2, 2, 2))

        # self.lin_p = nn.Linear(32, 128)
        # self.lin_1p3 = nn.Linear(512*64, 128)
        # self.lin_1p4 = nn.Linear(64 * 64, 128)
        # self.lin_1p5 = nn.Linear(8 * 64, 128)

        self.lin_f = nn.Linear(3,32)
        self.lin1 = nn.Linear(8 * 32, 128)
        self.lin1_out = nn.Linear(9, 128)
        self.lin3 = nn.Linear(128, 1)

        self.lin1_wsc = nn.Linear(128, 128)
        self.lin1_wp5 = nn.Linear(8*64, 128)
        self.lin2_wp5 = nn.Linear(128, 32)
        self.lin3_wp5 = nn.Linear(128, 1)



        self.re1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=dropout)

        # self.adc_weight = nn.Parameter(torch.ones(1, 1, 64, 64, 64), requires_grad=True)
        # self.dwi_weight = nn.Parameter(torch.ones(1, 1, 64, 64, 64), requires_grad=True)
        # self.t2_weight = nn.Parameter(torch.ones(1, 1, 64, 64, 64), requires_grad=True)
        # self.t1pre_weight = nn.Parameter(torch.ones(1, 1, 64, 64, 64), requires_grad=True)
        # self.a_weight = nn.Parameter(torch.ones(1, 1, 64, 64, 64), requires_grad=True)
        # self.d_weight = nn.Parameter(torch.ones(1, 1, 64, 64, 64), requires_grad=True)
        # self.v_weight = nn.Parameter(torch.ones(1, 1, 64, 64, 64), requires_grad=True)

        self.adc_weight3,self.adc_weight4,self.adc_weight5,self.adc_weight_c = nn.Parameter(torch.ones(1, 64, 8, 8, 8), requires_grad=True),nn.Parameter(torch.ones(1, 64, 4, 4, 4), requires_grad=True),nn.Parameter(torch.ones(1, 64, 2, 2, 2), requires_grad=True),nn.Parameter(torch.ones(128), requires_grad=True)
        self.b0_weight3,self.b0_weight4,self.b0_weight5,self.b0_weight_c = nn.Parameter(torch.ones(1, 64, 8, 8, 8), requires_grad=True),nn.Parameter(torch.ones(1, 64, 4, 4, 4), requires_grad=True),nn.Parameter(torch.ones(1, 64, 2, 2, 2), requires_grad=True),nn.Parameter(torch.ones(128), requires_grad=True)
        self.b500_weight, self.b500_weight4,self.b500_weight5,self.b500_weight_c = nn.Parameter(torch.ones(1, 64, 8, 8, 8), requires_grad=True),nn.Parameter(torch.ones(1, 64, 4, 4, 4),requires_grad=True),nn.Parameter(torch.ones(1, 64, 2, 2, 2), requires_grad=True),nn.Parameter(torch.ones(128), requires_grad=True)
        self.t2_weight3,self.t2_weight4,self.t2_weight5, self.t2_weight_c= nn.Parameter(torch.ones(1, 64, 8, 8, 8), requires_grad=True),nn.Parameter(torch.ones(1, 64, 4, 4, 4), requires_grad=True),nn.Parameter(torch.ones(1, 64, 2, 2, 2), requires_grad=True),nn.Parameter(torch.ones(128), requires_grad=True)
        self.t1pre_weight3,self.t1pre_weight4,self.t1pre_weight5,self.t1pre_weight_c = nn.Parameter(torch.ones(1, 64, 8, 8, 8), requires_grad=True),nn.Parameter(torch.ones(1, 64, 4, 4, 4), requires_grad=True),nn.Parameter(torch.ones(1, 64, 2, 2, 2), requires_grad=True),nn.Parameter(torch.ones(128), requires_grad=True)
        self.a_weight3,self.a_weight4,self.a_weight5,self.a_weight_c = nn.Parameter(torch.ones(1, 64, 8, 8, 8), requires_grad=True),nn.Parameter(torch.ones(1, 64, 4, 4, 4), requires_grad=True),nn.Parameter(torch.ones(1, 64, 2, 2, 2), requires_grad=True),nn.Parameter(torch.ones(128), requires_grad=True)
        self.d_weight3,self.d_weight4,self.d_weight5,self.d_weight_c = nn.Parameter(torch.ones(1, 64, 8, 8, 8), requires_grad=True),nn.Parameter(torch.ones(1, 64, 4, 4, 4), requires_grad=True),nn.Parameter(torch.ones(1, 64, 2, 2, 2), requires_grad=True),nn.Parameter(torch.ones(128), requires_grad=True)
        self.v_weight3,self.v_weight4,self.v_weight5,self.v_weight_c = nn.Parameter(torch.ones(1, 64, 8, 8, 8), requires_grad=True),nn.Parameter(torch.ones(1, 64, 4, 4, 4), requires_grad=True),nn.Parameter(torch.ones(1, 64, 2, 2, 2), requires_grad=True),nn.Parameter(torch.ones(128), requires_grad=True)
        self.feature_weight,self.feature_weight_c = nn.Parameter(torch.ones(3), requires_grad=True), nn.Parameter(torch.ones(128), requires_grad=True)


    def forward(self, adc, b500, t2, t1pre, a, d, v):
        # adc = norm_img(adc)
        # b500 = norm_img(b500)
        # t2 = norm_img(t2)
        # t1pre = norm_img(t1pre)
        # a = norm_img(a)
        # d = norm_img(d)
        # v = norm_img(v)

        # 得到各模态5个尺度特征图、最终输出特征、该模态预测结果
        adc_p1,adc_p2,adc_p3,adc_p4,adc_p5,adc_conv2, adc_output = self.single16(adc)
        b500_p1,b500_p2,b500_p3,b500_p4,b500_p5,b500_conv2,b500_output = self.single16(b500)
        t2_p1,t2_p2,t2_p3,t2_p4,t2_p5,t2_conv2, t2_output = self.single16(t2)
        t1pre_p1,t1pre_p2,t1pre_p3,t1pre_p4,t1pre_p5,t1pre_conv2, t1pre_output = self.single16(t1pre)
        a_p1,a_p2,a_p3,a_p4,a_p5,a_conv2, a_output = self.single16(a)
        d_p1,d_p2,d_p3,d_p4,d_p5,d_conv2, d_output = self.single16(d)
        v_p1,v_p2,v_p3,v_p4,v_p5,v_conv2, v_output = self.single16(v)
        # feature_conv = self.lin_f(feature)

        ws_p5 = self.adc_weight5 * adc_p5 + self.b500_weight5 * b500_p5 + self.t2_weight5 * t2_p5 + self.t1pre_weight5 * t1pre_p5 + self.a_weight5 * a_p5 + self.d_weight5 * d_p5 + self.v_weight5 * v_p5
        ws_p5 = ws_p5.view(ws_p5.size(0), -1)
        l1 = self.lin1_wp5(ws_p5)
        ws_conv1 = l1
        l1 = self.re1(l1)
        l1 = self.dropout(l1)
        l2 = self.lin2_wp5(l1)
        ws_conv2 = l2
        l2 = self.re1(l2)
        l2 = self.dropout(l2)
        l3 = self.lin3_wp5(l1)
        ws_output = self.sigmoid(l3)


        cat1 = torch.cat([ws_conv2,adc_conv2, b500_conv2, t2_conv2, t1pre_conv2, a_conv2, d_conv2, v_conv2], dim=1)
        cat1 = cat1.view(cat1.size(0), -1)
        l1 = self.lin1(cat1)
        conv1=l1
        l1 = self.re1(l1)
        l1 = self.dropout(l1)
        # l2 = self.lin2(l1)
        # conv1=l2
        # l2 =self.re1(l2)
        # l2 = self.dropout(l2)
        l3 = self.lin3(l1)
        output1 = self.sigmoid(l3)
        #
        cat2 = torch.cat([output1, ws_output, adc_output, b500_output, t2_output, t1pre_output, a_output, d_output, v_output], dim=1)
        cat2 = cat2.view(cat2.size(0), -1)
        l1 = self.lin1_out(cat2)
        conv2 = l1
        l1 = self.re1(l1)
        l1 = self.dropout(l1)
        # l2 = self.lin2(l1)
        # conv1=l2
        # l2 =self.re1(l2)
        # l2 = self.dropout(l2)
        l3 = self.lin3(l1)
        output2 = self.sigmoid(l3)

        return output1, ws_output, adc_output, b500_output,t2_output, t1pre_output, a_output, d_output, v_output, conv1, ws_conv1


if __name__ == "__main__":
    model = mymodel()
    # set_trace()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name,':',param.size())



# class mymodel(torch.nn.Module):
#     def __init__(self,dropout=0.5):
#         super(mymodel,self).__init__()
#         # self.single8_1 = Single8(1,out_ch=32)
#         # self.single8_2 = Single8(1,out_ch=32)
#         # self.single8_3 = Single8(1,out_ch=32)
#         # self.single8_4 = Single8(1, out_ch=32)
#         # self.single16_1 = Single16(1,out_ch=32)
#         # self.single16_2 = Single16(1,out_ch=32)
#         # self.single16_3 =ingle16(1,out_ch=32)
#         # self.single16_4 = Single16(1,out_ch=32)
#
#         self.single8_1 = Single8(1, out_ch=16)
#         self.single8_2 = Single8(1, out_ch=16)
#         self.single8_3 = Single8(1, out_ch=16)
#         self.single8_4 = Single8(1, out_ch=16)
#         self.single16_1 = Single16(1, out_ch=16)
#         self.single16_2 = Single16(1, out_ch=16)
#         self.single16_3 = Single16(1, out_ch=16)
#         self.single16_4 = Single16(1, out_ch=16)
#
#         #self.lin1 = nn.Linear(16,128)
#         #self.lin1 = nn.Linear(8,128)
#
#         self.lin1 = nn.Linear(16, 128)
#         self.re1 = nn.ReLU()
#         #self.lin2 = nn.Linear(128,2)
#         #self.sftmx1 = nn.Softmax(dim=1)
#         self.lin2 = nn.Linear(128,1)
#         self.sftmx1 = nn.Sigmoid()
#
#         #self.lin3 = nn.Linear(18,2)
#         #self.sftmx2 = nn.Softmax(dim=1)
#         # self.lin3 = nn.Linear(33,1)
#         self.lin3 = nn.Linear(9, 1)
#         self.sftmx2 = nn.Sigmoid()
#
#         #self.lin4 = nn.Linear(16,2)
#         #self.sftmx3 = nn.Softmax(dim=1)
#         self.lin4 = nn.Linear(24,1)
#         self.sftmx3 = nn.Sigmoid()
#         self.dropout = nn.Dropout(p=dropout)
#
#     def forward(self, adc,b0,b500,t2,t1,dongmai,menmai,yanchi):
#         adc_conv, adc_output = self.single8_1(adc)
#         b0_conv, b0_output = self.single8_2(b0)
#         b500_conv, b500_output = self.single8_3(b500)
#         t2_conv, t2_output = self.single8_4(t2)
#         t1_conv, t1_output = self.single16_1(t1)
#         dongmai_conv, dongmai_output = self.single16_2(dongmai)
#         menmai_conv, menmai_output = self.single16_3(menmai)
#         yanchi_conv, yanchi_output = self.single16_4(yanchi)
#
#         cat1 = torch.cat([adc_conv, b0_conv, b500_conv, t2_conv, t1_conv, dongmai_conv, menmai_conv, yanchi_conv],dim=1)
#         l1 = cat1.view(cat1.size(0),-1)
#         l1 = self.lin1(l1)
#         r1 = self.re1(l1)
#         r1 = self.dropout(r1)
#         l2 = self.lin2(r1)
#         output1 = self.sftmx1(l2)
#
#         cat2 = torch.cat([output1, adc_output,b0_output,b500_output,t2_output,t1_output,dongmai_output,menmai_output,yanchi_output],dim=1)
#         l3 = cat2.view(cat2.size(0),-1)
#         l3 = self.lin3(l3)
#         output2 = self.sftmx2(l3)
#
#         cat3 = torch.cat([adc_conv, b0_conv, b500_conv, t2_conv, t1_conv, dongmai_conv, menmai_conv, yanchi_conv,adc_output,b0_output,b500_output,t2_output,t1_output,dongmai_output,menmai_output,yanchi_output],dim=1)
#         l4 = cat3.view(cat3.size(0),-1)
#         l4 = self.lin4(l4)
#         output3 = self.sftmx2(l4)
#
#         return output1,output2,output3,adc_output,b0_output,b500_output,t2_output,t1_output,dongmai_output,menmai_output,yanchi_output
#
# if __name__ == "__main__":
#     model = mymodel()
#     # print(model)