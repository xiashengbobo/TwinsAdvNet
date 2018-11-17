#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 09:57:48 2018

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
 
       
class PyramidPooling_Module(nn.Module):
    def __init__(self, inplanes, pool_series, num_classes):
        super(PyramidPooling_Module, self).__init__()
        self.psp = []
        
        for scale in pool_series:
            self.psp.append(nn.Sequential(
                    nn.AdaptiveAvgPool2d(scale),
                    nn.Conv2d(inplanes, 512, kernel_size=1, bias=False),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=True)
                    ))
            
        self.psp = nn.ModuleList(self.psp)
        
        self.conv_last = nn.Sequential(
                nn.Conv2d(inplanes+len(pool_series)*512, 512, 
                          kernel_size=3, padding=1, dilation =1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                #nn.Dropout2d(0.1),
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
        
        """
        if not (input_size[2] == segSize[0] and input_size[3] == segSize[1]):
            x = nn.functional.upsample(x, size=segSize, mode='bilinear')
        """
        
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
        
        self.conv3 = conv3x3(64, 64, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
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
        
        self.conv3 = conv3x3(64, 64, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)    
        
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
        
class Refnet_PyP(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Refnet_PyP, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        #self.layer5 = self._make_aspp_layer(ASPP_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer5 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        self.conv1x1_l3 = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 3, 2048, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(2048, num_classes, kernel_size=1)
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
        
        x = self.layer1(x)
        x = self.layer2(x)
        x0 = self.layer3(x)
        
        x1 = self.layer4(x0)  
        x1 = self.layer5(x1)
        
        #out = [x, x0, x2]
        out = [self.conv1x1_l2(x), self.conv1x1_l3(x0), x1]
        out = torch.cat(out, 1)
        
        x2= self.conv_oneplus(out)
        
        
        return x1, x1, x2
###################################################################
        
class Refnet_ASPP(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Refnet_ASPP, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer5 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        self.conv1x1_l3 = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 3, 2048, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(2048, num_classes, kernel_size=1)
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
        
        x = self.layer1(x)
        x = self.layer2(x)
        x0 = self.layer3(x)
        
        x1 = self.layer4(x0)  
        x1 = self.layer5(x1)
        
        #out = [x, x0, x2]
        out = [self.conv1x1_l2(x), self.conv1x1_l3(x0), x1]
        out = torch.cat(out, 1)
        
        x2= self.conv_oneplus(out)
        
        
        return x1, x1, x2
    
########################################################################
    
########################################################################

class Refnet_rp(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Refnet_rp, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=2, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        #self.layer5 = self._make_aspp_layer(ASPP_Module, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer5 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        self.conv1x1_l3 = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 3, 2048, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv2d(2048, num_classes, kernel_size=1)
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
        
        x = self.layer1(x)
        x = self.layer2(x)
        x0 = self.layer3(x)
        
        x1 = self.layer4(x0)  
        x1 = self.layer5(x1)
        
        #out = [x, x0, x2]
        out = [self.conv1x1_l2(x), self.conv1x1_l3(x0), x1]
        out = torch.cat(out, 1)
        
        x2= self.conv_oneplus(out)
        
        
        return x1, x1, x2        
    
    
#########################################################################
class Mulrefnet(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Mulrefnet, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer5 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        self.layer6 = self._make_aspp_layer(ASPP_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer5 = self._make_psp_layer(PyramidPooling_Module, 1024, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        self.conv1x1_l3 = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 3, 1024, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(1024, num_classes, kernel_size=1)
                )
        
        
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
        #x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x0 = self.layer3(x)
        x2 = self.layer4(x0)        
        x2 = self.layer5(x2) 
        
        x1 = self.layer6(x0.detach())
        
        #out = [x, x0, x2]
        out = [self.conv1x1_l2(x), self.conv1x1_l3(x0), x2]
        out = torch.cat(out, 1)
        
        x3= self.conv_oneplus(out)
        
        
        return x1, x2, x3    

#####################################################################
class Mulrefnet_DL(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Mulrefnet_DL, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer5 = self._make_psp_layer(PyramidPooling_Module, 1024, [1, 2, 3, 6], num_classes)
        self.layer6 = self._make_psp_layer(PyramidPooling_Module, 1024, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        self.conv1x1_l3 = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 3, 2048, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(2048, num_classes, kernel_size=1)
                )
        
        
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
        x0 = self.layer3(x)
        x0_= self.layer4(x0)
               
        x2 = self.layer5(x0)
        x1 = self.layer6(x0_.detach())
        
        #out = [x, x0, x1]
        out = [self.conv1x1_l2(x), self.conv1x1_l3(x0), x2]
        out = torch.cat(out, 1)
        
        x3= self.conv_oneplus(out)
        
        
        return x1, x2, x3    

#####################################################################        
    
        
class Mulrefnet_SL(nn.Module):
    
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(Mulrefnet_SL, self).__init__()
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = conv3x3(64, 64, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=2, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer5 = self._make_psp_layer(PyramidPooling_Module, 1024, [1, 2, 3, 6], num_classes)
        self.layer6 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)      
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        self.conv1x1_l3 = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=True),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ReLU(inplace=True))
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 3, 2048, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(2048, num_classes, kernel_size=1)
                )
        
        
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
        x0 = self.layer3(x)
        x0_ = self.layer4(x0)
               
        x2 = self.layer5(x0_)
        x1 = self.layer6(x0_.detach())
        
        #out = [x, x0, x1]
        out = [self.conv1x1_l2(x), self.conv1x1_l3(x0), x2 ]
        out = torch.cat(out, 1)
        
        x3= self.conv_oneplus(out)
        
        
        return x1, x2, x3    

#####################################################################
        
class Mulrefnet_DELU(nn.Module):
    
    def __init__(self, block, layers, num_classes, use_softmax=False):
        self.inplanes = 64
        super(Mulrefnet_DELU, self).__init__()
        self.use_softmax = use_softmax
        
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        #self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.ELU(inplace=True)
        
        self.conv2 = conv3x3(64, 64, stride=2)
        #self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ELU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=2, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ELU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer5 = self._make_aspp_layer(ASPP_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer5 = self._make_psp_layer(PyramidPooling_Module, 1024, [1, 2, 3, 6], num_classes)
        self.layer6 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ELU(alpha=1.0, inplace=True))
        self.conv1x1_l3 = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ELU(alpha=1.0, inplace=True))
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 3, 2048, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(2048),
                nn.ELU(alpha=1.0, inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(2048, num_classes, kernel_size=1)
                )
        
        
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
        x0 = self.layer3(x)
        
        x1 = self.layer5(x0.detach())

        x2 = self.layer4(x0)      
        x2 = self.layer6(x2)
        
        #out = [x, x0, x2]
        out = [self.conv1x1_l2(x), self.conv1x1_l3(x0), x2]
        out = torch.cat(out, 1)
        
        x3= self.conv_oneplus(out)
        
        return x1, x2, x3    

#####################################################################    
class Mulrefnet_SELU(nn.Module):
    
    def __init__(self, block, layers, num_classes, use_softmax=False):
        self.inplanes = 64
        super(Mulrefnet_SELU, self).__init__()
        self.use_softmax = use_softmax
        
        self.conv1 = conv3x3(3, 64, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        #self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.ELU(inplace=True)
        
        self.conv2 = conv3x3(64, 64, stride=2)
        #self.conv2 = conv3x3(64, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ELU(inplace=True)
        
        self.conv3 = conv3x3(64, 64, stride=2, padding=2, dilation=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ELU(inplace=True)
        
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        
        self.layer6 = self._make_aspp_layer(ASPP_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        #self.layer5 = self._make_psp_layer(PyramidPooling_Module, 1024, [1, 2, 3, 6], num_classes)
        self.layer5 = self._make_psp_layer(PyramidPooling_Module, 2048, [1, 2, 3, 6], num_classes)
        
        self.conv1x1_l2 = nn.Sequential(nn.Conv2d(512, num_classes, kernel_size=3, stride=1, padding=4, dilation=4, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ELU(alpha=1.0, inplace=True))
        self.conv1x1_l3 = nn.Sequential(nn.Conv2d(1024, num_classes, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
                                        nn.BatchNorm2d(num_classes),
                                        nn.ELU(alpha=1.0, inplace=True))
        
        self.conv_oneplus = nn.Sequential(
                nn.Conv2d(num_classes * 3, 2048, kernel_size=3, padding=4, dilation=4, bias=False),
                nn.BatchNorm2d(2048),
                nn.ELU(alpha=1.0, inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(2048, num_classes, kernel_size=1)
                )
        
        
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
        #x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x0 = self.layer3(x)
        x0_ = self.layer4(x0)
               
        x1 = self.layer5(x0_.detach())
        x2 = self.layer6(x0_)
        
        #out = [x, x0, x2]
        out = [self.conv1x1_l2(x), self.conv1x1_l3(x0), x2]
        out = torch.cat(out, 1)
        
        x3= self.conv_oneplus(out)
        
        return x1, x2, x3    

#####################################################################         
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

def RefNet_PyP(num_classes=150):  # ? 151
    model = Refnet_PyP(Bottleneck, [3, 4, 6, 3], num_classes)  # 101 [3, 4, 23, 3]
    return model

def RefNet_ASPP(num_classes=150):  # ? 151
    model = Refnet_ASPP(Bottleneck, [3, 4, 6, 3], num_classes)  # 101 [3, 4, 23, 3]
    return model

def RefNet_RP(num_classes=150):  # ? 151
    model = Refnet_rp(Bottleneck, [3, 4, 6, 3], num_classes)  # 101 [3, 4, 23, 3]
    return model
#################################################################

def MulRefNet(num_classes=150):  # ? 151
    model = Mulrefnet(Bottleneck, [3, 4, 6, 3], num_classes)   # 101
    return model

def MulRefNet_DL(num_classes=150):  # ? 151
    model = Mulrefnet_DL(Bottleneck, [3, 4, 6, 3], num_classes)   # 101
    return model

def MulRefNet_SL(num_classes=150):  # ? 151
    model = Mulrefnet_SL(Bottleneck, [3, 4, 6, 3], num_classes)   # 101
    return model
###############################################################

def MulRefNet_DELU(num_classes=150):  # ? 151
    model = Mulrefnet_DELU(Bottleneck, [3, 4, 6, 3], num_classes)   # 101
    return model

def MulRefNet_SELU(num_classes=150):  # ? 151
    model = Mulrefnet_SELU(Bottleneck, [3, 4, 6, 3], num_classes)   # 101
    return model
###################################################################
    
if __name__ == '__main__':
    model = MulRefNet_DELU(num_classes=150)
    #model = TwinsAdvNet_DL(num_classes=150)
    
    #print(model)
    
    for name, param in model.named_parameters():
        print(name, param.size())
    print('*#*'*20)
    
    
    
    
    
    
    
    
    
    
    
    