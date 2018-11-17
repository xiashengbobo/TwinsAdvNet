#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 20:02:42 2018

@author: bobo
"""
#import os
#import sys
#import math

import torch
import torch.nn as nn

affine_par = True

def conv3x3(in_planes, out_planes, stride=1, padding=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                     padding=padding, dilation=dilation, bias=False)

class BasicBlock(nn.Module):
    """
    affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数
    inplace: 选择是否进行覆盖运算
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine = affine_par)
        
        padding = dilation 
            
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=padding, bias=False, dilation = dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine = affine_par)
        
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine = affine_par)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.downsample = downsample
        self.stride = stride



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
#########################################################################
        
class ASPP_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(ASPP_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()   # to pytorch list
        for dilation, padding in list(zip(dilation_series, padding_series)):
            self.conv2d_list.append(
                    nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, 
                              padding=padding, dilation=dilation, bias=True))
            
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)
                
    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):  # list()
            out += self.conv2d_list[i + 1](x)
            return out
        
class ASPPModule(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes=256):
        super(ASPPModule, self).__init__()
        self.conv2d_list = nn.ModuleList()   # to pytorch list
        for dilation, padding in list(zip(dilation_series, padding_series)):
            self.conv2d_list.append(
                    nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, 
                              padding=padding, dilation=dilation, bias=True))
            
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)
                
    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):  # list()
            out += self.conv2d_list[i + 1](x)
            return out

       
class PyramidPooling_Module(nn.Module):
    def __init__(self, inplanes, pool_series, num_classes):
        super(PyramidPooling_Module, self).__init__()
        self.psp = []
        
        for scale in pool_series:
            self.psp.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(inplanes, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512, momentum=0.95),
                    nn.ReLU(inplace=True)
                    ))
            
        self.psp = nn.ModuleList(self.psp)
        self.conv_last = nn.Sequential(
                nn.Conv2d(inplanes+len(pool_series)*512, 512, 
                          kernel_size=3, padding=1, dilation =1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                nn.Conv2d(512, num_classes, kernel_size=1)
                )
        
        
    def forward(self, x):
        input_size = x.size()
        psp_out = [x]
        for pool_scale in self.psp:
            psp_out.append(nn.functional.upsample(
                    pool_scale(x),
                    (input_size[2], input_size[3]),
                    mode='bilinear'))
        psp_out = torch.cat(psp_out, 1)
        
        out = self.conv_last(psp_out)
        
        return out
    
class PyramidPoolingModule(nn.Module):
    def __init__(self, inplanes, pool_series, num_classes):
        super(PyramidPoolingModule, self).__init__()
        self.psp = []
        
        for scale in pool_series:
            self.psp.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(inplanes, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512, momentum=0.95),
                    nn.ReLU(inplace=True)
                    ))
            
        self.psp = nn.ModuleList(self.psp)
        
    def forward(self, x):
        input_size = x.size()
        psp_out = [x]
        for pool_scale in self.psp:
            psp_out.append(nn.functional.upsample(
                    pool_scale(x),
                    (input_size[2], input_size[3]),
                    mode='bilinear'))
        out = torch.cat(psp_out, 1)  # 2048 + 512x4
        
        return out

#########################################################################

class ResnetDilated_PyP(nn.Module):    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResnetDilated_PyP, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        
        self.layer5 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x, x, x
    
####################################################################

class ResnetDilated_ASPP(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResnetDilated_ASPP, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        
        #self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)  
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [1, 6, 12, 18], [1, 6, 12, 18], num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0, 0.001)
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        
        return x, x, x
    
####################################################################        
class Refnet(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Refnet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18], [6, 12, 18], num_classes)
        #self.layer5 = self._make_psp_layer(PyramidPoolingModule, 2048, [1, 2, 3, 6], num_classes)
       
        
        self.conv1x1_l1 = nn.Conv2d(256, 48, kernel_size=1, stride=1, bias=False)
        self.conv1x1_l2 = nn.Conv2d(512, 48, kernel_size=1, stride=1, bias=False)
      
        self.conv_last = nn.Sequential(
                nn.BatchNorm2d(num_classes + 96),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes + 96, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
                )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0.0, 0.001)
                
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        #print(x.size())
        
        x_1 = self.layer1(x)
        #print(x_1.size())
        x_2 = self.layer2(x_1)
        #print(x_2.size())
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3) 
        
        x2 = self.layer5(x_4)
        
        #out = [x, x0, x2]
        #x_cat1 = [self.conv1x1_l2(x_2), self.conv1x1_l4(x_4), x2]
        #x_cat1 = [self.conv1x1_l2(x_2), x2]
        #x_cat1 = torch.cat(x_cat1, 1)

        x_1_size = x_1.size()
        x_2_size = x_2.size()
        #print(x_1.size())
        x2_size = x2.size()
        
        if not (x2_size[2] == x_2_size[2] and x2_size[3] == x_2_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_2_size[2], x_2_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_2.size()[2:3], "{0} vs {1}".format(x0.size(), x_2.size()) 
        x0 = [self.conv1x1_l2(x_2), x0]  
        x0 = torch.cat(x0, 1)
        
        x0_size = x0.size()
        if not (x0_size[2] == x_1_size[2] and x0_size[3] == x_1_size[3]):
            x0 = nn.functional.upsample(x0, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_1.size()[2:3], "{0} vs {1}".format(x0.size(), x_1.size()) 
        
        #print(x0.size())
        #print(x_1.size())
        x3 = [self.conv1x1_l1(x_1), x0]
        x3 = torch.cat(x3, 1)
        #print(x3.size())
        
        x3 = self.conv_last(x3)
            
        return x3, x3, x3


##################################################################### 
class Refnet_up(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Refnet_up, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        #self.conv2 = conv3x3(64, 64, stride=1, padding=2, dilation=2)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        #self.conv3 = conv3x3(64, 64, stride=1, padding=2, dilation=2)
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18], [6, 12, 18], num_classes)
        
        self.conv1x1_l1 = nn.Conv2d(256, 48, kernel_size=1, stride=1, bias=False)
        #self.conv1x1_l2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False) 
        
        self.conv1x1_l3 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=False)
        self.conv1x1_l4 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=False)
        
        self.conv_oneplus = nn.Sequential(
                nn.BatchNorm2d(num_classes * 3),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes * 3, num_classes, kernel_size=1, stride=1, bias=False)
                )

        self.conv_last = nn.Sequential(
                nn.BatchNorm2d(num_classes + 48),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes + 48, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                #nn.Conv2d(512, num_classes, kernel_size=1)
                nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
                )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0.0, 0.001)
                
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        #print(x.size())
        
        x_1 = self.layer1(x)
        #print(x_1.size())
        x_2 = self.layer2(x_1)
        #print(x_2.size())
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3) 
        
        x_5 = self.layer5(x_4)
        
        #out = [x, x0, x2]
        #x_cat1 = [self.conv1x1_l2(x_2), self.conv1x1_l4(x_4), x2]
        #x_cat1 = [self.conv1x1_l2(x_2), x2]
        #x_cat1 = torch.cat(x_cat1, 1)
        x2 = [self.conv1x1_l3(x_3), self.conv1x1_l4(x_4), x_5]
        x2 = torch.cat(x2, 1)
    
        x2= self.conv_oneplus(x2)
        
        x_1_size = x_1.size()
        #x_2_size = x_2.size()
        #print(x_1.size())
        x2_size = x2.size()
        
        if not (x2_size[2] == x_1_size[2] and x2_size[3] == x_1_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_1.size()[2:3], "{0} vs {1}".format(x0.size(), x_1.size())
        """
        if not (x2_size[2] == x_2_size[2] and x2_size[3] == x_2_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_2_size[2], x_2_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_2.size()[2:3], "{0} vs {1}".format(x0.size(), x_2.size()) 
        """
        
        #print(x0.size())
        #print(x_1.size())
        x3 = [x0, self.conv1x1_l1(x_1)]
        #x3 = [x0, self.conv1x1_l2(x_2)]
        
        x3 = torch.cat(x3, 1)
        #print(x3.size())
        
        x3 = self.conv_last(x3)
            
        return x2, x2, x3

##################################################################### 

class Refnet_ASPP(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Refnet_ASPP, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=1, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18], [6, 12, 18], num_classes)
        
        self.conv1x1_l1 = nn.Conv2d(256, 48, kernel_size=1, stride=1, bias=False) 
        
        self.conv1x1_l2 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=False)
        self.conv1x1_l4 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=False)
        
        self.conv_oneplus = nn.Sequential(
                nn.BatchNorm2d(num_classes * 3),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes * 3, num_classes, kernel_size=1, stride=1, bias=True)
                )

        self.conv_last = nn.Sequential(
                nn.BatchNorm2d(num_classes + 48),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes + 48, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
                )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0.0, 0.001)
                
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        #print(x.size())
        
        x_1 = self.layer1(x)
        #print(x_1.size())
        x_2 = self.layer2(x_1)
        #print(x_2.size())
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3) 
        
        x_5 = self.layer5(x_4)
        
        #out = [x, x0, x2]
        #x_cat1 = [self.conv1x1_l2(x_2), self.conv1x1_l4(x_4), x2]
        #x_cat1 = [self.conv1x1_l2(x_2), x2]
        #x_cat1 = torch.cat(x_cat1, 1)
        x2 = [self.conv1x1_l2(x_2), self.conv1x1_l4(x_4), x_5]
        x2 = torch.cat(x2, 1)
    
        x2= self.conv_oneplus(x2)
        
        x_1_size = x_1.size()
        #print(x_1.size())
        x2_size = x2.size()
        
        if not (x2_size[2] == x_1_size[2] and x2_size[3] == x_1_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
        
        assert x0.size()[2:3] == x_1.size()[2:3], "{0} vs {1}".format(x0.size(), x_1.size()) 
        
        #print(x0.size())
        #print(x_1.size())
        x3 = [x0, self.conv1x1_l1(x_1)]
        x3 = torch.cat(x3, 1)
        #print(x3.size())
        
        x3 = self.conv_last(x3)
            
        return x2, x2, x3


##################################################################### 

class MulRefnet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(MulRefnet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18], [6, 12, 18], num_classes)
        
        self.layer6 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
       
        self.conv1x1_l1 = nn.Conv2d(256, 48, kernel_size=1, stride=1, bias=False)
        self.conv1x1_l2 = nn.Conv2d(512, 48, kernel_size=1, stride=1, bias=False)
        
        self.conv_last = nn.Sequential(
                nn.BatchNorm2d(num_classes + 96),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes + 96, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
                )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0.0, 0.001)
                
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        #print(x.size())
        
        x_1 = self.layer1(x)
        #print(x_1.size())
        x_2 = self.layer2(x_1)
        #print(x_2.size())
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3) 
        
        x1 = self.layer6(x_4)
        #x1 = self.layer6(x_4.detach())
        
        x2 = self.layer5(x_4)
        
        #out = [x, x0, x2]
        #x_cat1 = [self.conv1x1_l2(x_2), self.conv1x1_l4(x_4), x2]
        #x_cat1 = [self.conv1x1_l2(x_2), x2]
        #x_cat1 = torch.cat(x_cat1, 1)

        x_1_size = x_1.size()
        x_2_size = x_2.size()
        #print(x_1.size())
        x2_size = x2.size()
        
        if not (x2_size[2] == x_2_size[2] and x2_size[3] == x_2_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_2_size[2], x_2_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_2.size()[2:3], "{0} vs {1}".format(x0.size(), x_2.size()) 
        x0 = [self.conv1x1_l2(x_2), x0]  
        x0 = torch.cat(x0, 1)
        
        x0_size = x0.size()
        if not (x0_size[2] == x_1_size[2] and x0_size[3] == x_1_size[3]):
            x0 = nn.functional.upsample(x0, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_1.size()[2:3], "{0} vs {1}".format(x0.size(), x_1.size()) 
        
        #print(x0.size())
        #print(x_1.size())
        x3 = [self.conv1x1_l1(x_1), x0]
        x3 = torch.cat(x3, 1)
        #print(x3.size())
        
        x3 = self.conv_last(x3)
            
        return x1, x2, x3

        
##################################################################### 
class MulRefnet_PL(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(MulRefnet_PL, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer5 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        self.layer6 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer5 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l1 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(48),
                                        nn.ReLU(inplace=True))
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        """
        self.conv1x1_l4 = nn.Sequential(nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        """
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(inplace=True),
                )

        self.conv_last = nn.Sequential(
                nn.Conv2d(num_classes + 48, 1024, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(1024, num_classes, kernel_size=1)
                )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0.0, 0.001)
                
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3) 
        
        x2 = self.layer5(x_4)
        x1 = self.layer6(x_4.detach())
        
        x_cat1 = [self.conv1x1_l2(x_2), x2]
        x_cat1 = torch.cat(x_cat1, 1)
    
        x0= self.conv_oneplus(x_cat1)
        
        x0_size = x0.size()
        x_1_size = x_1.size()
        
        if not (x0_size[2] == x_1_size[2] and x0_size[3] == x_1_size[3]):
            x0 = nn.functional.upsample(x0, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
        
        assert x0.size()[2:3] == x_1.size()[2:3], "{0} vs {1}".format(x0.size(), x_1.size()) 
        x3 = [x0, self.conv1x1_l1(x_1)]
        x3 = torch.cat(x3, 1)
        x3 = self.conv_last(x3)
    
        return x1, x2, x3

        
##################################################################### 
class MulRefnet_up(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(MulRefnet_up, self).__init__()
        
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        #self.conv2 = conv3x3(64, 64, stride=1, padding=2, dilation=2)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        #self.conv3 = conv3x3(64, 64, stride=1, padding=2, dilation=2)
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        """
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18], [6, 12, 18], num_classes)
        self.layer6 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l1 = nn.Conv2d(256, 48, kernel_size=1, stride=1, bias=False)
        #self.conv1x1_l2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False) 
        
        self.conv1x1_l3 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=False)
        self.conv1x1_l4 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=False)
        
        self.conv_oneplus = nn.Sequential(
                nn.BatchNorm2d(num_classes * 3),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes * 3, num_classes, kernel_size=1, stride=1, bias=False)
                )

        self.conv_last = nn.Sequential(
                nn.BatchNorm2d(num_classes + 48),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes + 48, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                #nn.Conv2d(512, num_classes, kernel_size=1)
                nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
                )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0.0, 0.001)
                
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        #print(x.size())
        
        x_1 = self.layer1(x)
        #print(x_1.size())
        x_2 = self.layer2(x_1)
        #print(x_2.size())
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3) 
        
        x_5 = self.layer5(x_4)
        
        x1 = self.layer6(x_4.detach())
        #x1 = self.layer6(x_4)
        
        #out = [x, x0, x2]
        #x_cat1 = [self.conv1x1_l2(x_2), self.conv1x1_l4(x_4), x2]
        #x_cat1 = [self.conv1x1_l2(x_2), x2]
        #x_cat1 = torch.cat(x_cat1, 1)
        x2 = [self.conv1x1_l3(x_3), self.conv1x1_l4(x_4), x_5]
        x2 = torch.cat(x2, 1)
    
        x2= self.conv_oneplus(x2)
        
        x_1_size = x_1.size()
        #x_2_size = x_2.size()
        #print(x_1.size())
        x2_size = x2.size()
        
        if not (x2_size[2] == x_1_size[2] and x2_size[3] == x_1_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_1.size()[2:3], "{0} vs {1}".format(x0.size(), x_1.size())
        """
        if not (x2_size[2] == x_2_size[2] and x2_size[3] == x_2_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_2_size[2], x_2_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_2.size()[2:3], "{0} vs {1}".format(x0.size(), x_2.size()) 
        """
        
        #print(x0.size())
        #print(x_1.size())
        x3 = [x0, self.conv1x1_l1(x_1)]
        #x3 = [x0, self.conv1x1_l2(x_2)]
        
        x3 = torch.cat(x3, 1)
        #print(x3.size())
        
        x3 = self.conv_last(x3)
            
        return x1, x2, x3


        
#####################################################################
class MulRefnet_SELU(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(MulRefnet_SELU, self).__init__()
        
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ELU(inplace=True)
        
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ELU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ELU(inplace=True)
        
        #self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        #self.relu1 = nn.ELU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer5 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        self.layer6 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        
        self.conv1x1_l1 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
                                        nn.BatchNorm2d(48),
                                        nn.ELU(inplace=True))
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=6, dilation=6, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ELU(inplace=True))
        
        """
        self.conv1x1_l4 = nn.Sequential(nn.Conv2d(2048, 48, kernel_size=1, stride=1, bias=False),
                                        nn.BatchNorm2d(48),
                                        nn.ELU(inplace=True))
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(num_classes),
                nn.ELU(inplace=True),
                )
        """

        self.conv_last = nn.Sequential(
                nn.Conv2d(num_classes * 2 + 48, 1024, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(1024),
                nn.ELU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(1024, num_classes, kernel_size=1)
                )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0.0, 0.001)
                
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        #x = self.maxpool(x)
        
        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3) 
        
        x2 = self.layer5(x_4)
        x1 = self.layer6(x_4.detach())
        
        #out = [x, x0, x2]
        #x_cat1 = [self.conv1x1_l2(x_2), self.conv1x1_l4(x_4), x2]
        #x_cat1 = [self.conv1x1_l2(x_2), x2]
        #x_cat1 = torch.cat(x_cat1, 1)
    
        #x0= self.conv_oneplus(x_cat1)
        
        x2_size = x2.size()
        #print(x2_size)
        x_1_size = x_1.size()
        #print(x_1_size)
        x_2_size = x_2.size()
        #print(x_2_size)
        
        if not (x2_size[2] == x_1_size[2] and x2_size[3] == x_1_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
            
        if not (x_2_size[2] == x_1_size[2] and x_2_size[3] == x_1_size[3]):
            x_2 = nn.functional.upsample(x_2, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
        
        #x0 = nn.functional.upsample(x0, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
        
        assert x0.size()[2:3] == x_1.size()[2:3], "{0} vs {1}".format(x0.size(), x_1.size()) 
        assert x_2.size()[2:3] == x_1.size()[2:3], "{0} vs {1}".format(x_2.size(), x_1.size())
        
        x3 = [x0, self.conv1x1_l2(x_2), self.conv1x1_l1(x_1)]
        x3 = torch.cat(x3, 1)
        x3 = self.conv_last(x3)
    
        return x1, x2, x3

#####################################################################        
##################################################################### 
class MulRefnet_self(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(MulRefnet_self, self).__init__()
        
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        #self.conv2 = conv3x3(64, 64, stride=1, padding=2, dilation=2)
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        #self.conv3 = conv3x3(64, 64, stride=1, padding=2, dilation=2)
        self.conv3 = conv3x3(64, 64)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        """
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=2)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18], [6, 12, 18], num_classes)
        self.layer6 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l1 = nn.Conv2d(256, 48, kernel_size=1, stride=1, bias=False)
        #self.conv1x1_l2 = nn.Conv2d(512, 64, kernel_size=1, stride=1, bias=False) 
        
        self.conv1x1_l3 = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, bias=False)
        self.conv1x1_l4 = nn.Conv2d(2048, num_classes, kernel_size=1, stride=1, bias=False)
        
        self.conv_oneplus = nn.Sequential(
                nn.BatchNorm2d(num_classes * 3),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes * 3, num_classes, kernel_size=1, stride=1, bias=False)
                )

        self.conv_last = nn.Sequential(
                nn.BatchNorm2d(num_classes + 48),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_classes + 48, 512, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.2),
                #nn.Conv2d(512, num_classes, kernel_size=1)
                nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=True)
                )
        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data.normal_(0.0, 0.001)
                
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)  # 1e-4
                
    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                    )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
            
        return nn.Sequential(*layers) 
    
    def _make_aspp_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)   
        
    def _make_psp_layer(self, block, inplanes, pool_series, num_classes):
        return block(inplanes, pool_series, num_classes)
    
    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        #print(x.size())
        
        x_1 = self.layer1(x)
        #print(x_1.size())
        x_2 = self.layer2(x_1)
        #print(x_2.size())
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3) 
        
        x_5 = self.layer5(x_4)
        
        #x1 = self.layer6(x_4.detach())
        #x1 = self.layer6(x_4)
        
        #out = [x, x0, x2]
        #x_cat1 = [self.conv1x1_l2(x_2), self.conv1x1_l4(x_4), x2]
        #x_cat1 = [self.conv1x1_l2(x_2), x2]
        #x_cat1 = torch.cat(x_cat1, 1)
        x2 = [self.conv1x1_l3(x_3), self.conv1x1_l4(x_4), x_5]
        x2 = torch.cat(x2, 1)
    
        x2= self.conv_oneplus(x2)
        
        x_1_size = x_1.size()
        #x_2_size = x_2.size()
        #print(x_1.size())
        x2_size = x2.size()
        
        if not (x2_size[2] == x_1_size[2] and x2_size[3] == x_1_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_1_size[2], x_1_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_1.size()[2:3], "{0} vs {1}".format(x0.size(), x_1.size())
        """
        if not (x2_size[2] == x_2_size[2] and x2_size[3] == x_2_size[3]):
            x0 = nn.functional.upsample(x2, size=(x_2_size[2], x_2_size[3]), mode='bilinear')
        assert x0.size()[2:3] == x_2.size()[2:3], "{0} vs {1}".format(x0.size(), x_2.size()) 
        """
        
        #print(x0.size())
        #print(x_1.size())
        x3 = [x0, self.conv1x1_l1(x_1)]
        #x3 = [x0, self.conv1x1_l2(x_2)]
        
        x3 = torch.cat(x3, 1)
        #print(x3.size())
        
        x3 = self.conv_last(x3)
            
        return x3, x2, x3


        
#####################################################################

def ResNet34_PyP(num_classes=150):
    model = ResnetDilated_PyP(BasicBlock, [3, 4, 6, 3], num_classes)
    return model

def ResNet50_PyP(num_classes=150):
    model = ResnetDilated_PyP(Bottleneck, [3, 4, 6, 3], num_classes)
    return model

def ResNet101_PyP(num_classes=150):  # ? 151
    model = ResnetDilated_PyP(Bottleneck, [3, 4, 23, 3], num_classes)   # 101
    return model
#####################################################################
    
def ResNet34_ASPP(num_classes=150):
    model = ResnetDilated_ASPP(BasicBlock, [3, 4, 6, 3], num_classes)
    return model

def ResNet50_ASPP(num_classes=150):
    model = ResnetDilated_ASPP(Bottleneck, [3, 4, 6, 3], num_classes)
    return model

def ResNet101_ASPP(num_classes=150):  # ? 151
    model = ResnetDilated_ASPP(Bottleneck, [3, 4, 23, 3], num_classes)   # 101
    return model
################################################################

def RefNet(num_classes=150):  # ? 151
    model = Refnet(Bottleneck, [3, 4, 6, 3], num_classes)  # 101 [3, 4, 23, 3]
    return model

def RefNet_UP(num_classes=150):  # ? 151
    model = Refnet_up(Bottleneck, [3, 4, 6, 3], num_classes)  # 101 [3, 4, 23, 3]
    return model

def MulRefNet(num_classes=150):  # ? 151
    model = MulRefnet(Bottleneck, [3, 4, 6, 3], num_classes)   # 101
    return model

def MulRefNet_PL(num_classes=150):  # ? 151
    model = MulRefnet_PL(Bottleneck, [3, 4, 6, 3], num_classes)   # 101
    return model

def MulRefNet_UP(num_classes=150):  # ? 151
    model = MulRefnet_up(Bottleneck, [3, 4, 6, 3], num_classes)   # 101
    return model

def MulRefNet_SELU(num_classes=150):  # ? 151
    model = MulRefnet_SELU(Bottleneck, [3, 4, 6, 3], num_classes)   # 101
    return model
###################################################################
    
if __name__ == '__main__':
    model = RefNet(num_classes=150)
    #model = TwinsAdvNet_DL(num_classes=150)
    
    #print(model)
    
    for name, param in model.named_parameters():
        print(name, param.size())
    print('*#*'*20)
    
    
    
    
    
    
    
    
    
    
    
    