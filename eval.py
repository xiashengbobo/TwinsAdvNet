#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:57:33 2018

@author: bobo
"""


# System libs
import os
#import sys
import datetime
import argparse
#from collections import OrderecdDict

# Numerical libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data, model_zoo
#import torchvision.models as models

import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave
from scipy.ndimage import zoom

#from PIL import Image
#import matplotlib.pyplot as plt

# Our libs
from utils.utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion

#from genertors.refnet import RefNet_Baseline, RefNet_BS, RefNet_MulDL, RefNet_PA
#from genertors.MulRefNet import ResNet34, ResNet50, ResNet101, RefNet, MulRefNet_D2d
#from genertors.MulRefNet import MulRefNet, MulRefNet_D, MulRefNet_D2d
#from genertors.RefNet import ResNet34_PyP, ResNet50_PyP, ResNet101_PyP, ResNet34_ASPP, ResNet50_ASPP, ResNet101_ASPP, RefNet_PyP, RefNet_ASPP, RefNet_RP, MulRefNet, MulRefNet_DL, MulRefNet_SL, MulRefNet_DELU, MulRefNet_SELU
from genertors.MulRefNet import ResNet34_PyP, ResNet50_PyP, ResNet101_PyP, ResNet34_ASPP, ResNet50_ASPP, ResNet101_ASPP, RefNet, RefNet_UP, MulRefNet, MulRefNet_UP, MulRefNet_SELU
#from discriminators.discriminator import Discriminator
#from utils.loss import CrossEntropy2d
from utils.loss import CrossEntropyLoss2d
from datasets.mit_dataset import MITSceneParsingDataset

#start_time = timeit.default_timer()

#RESTORE_FROM = './pretrained/MS_DeepLab_resnet_pretrained_COCO_init_modified.pth'
#RESTORE_FROM = './pretrained/resnet101COCO-41f33a49.pth'
##RESTORE_FROM = './pretrained/resnet34-places365.pth'

#RESTORE_FROM = './checkpoints/baseline_ADE2016_ResNet50_ASPP_Ngpu1_batchSize8_epochs100_imgSize354_segSize354_nclasses150_S1_1_S2_1_La_0.05_adv_0.01/ResNet50_ASPP_100epoch_2526_latest.pth'
#RESTORE_FROM = './checkpoints/baseline_ADE2016_RefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1_S2_1_La_0.1_adv_0.01/RefNet_100epoch_2526_latest.pth'
#RESTORE_FROM = './checkpoints/baseline_ADE2016_MulRefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1_S2_1_La_0.1_adv_0.01/MulRefNet_100epoch_2526_latest.pth'
#RESTORE_FROM = './checkpoints/baseline_ADE2016_MulRefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1_S2_1_La_0.2_adv_0.005/MulRefNet_100epoch_2526_latest.pth'
#RESTORE_FROM = './checkpoints/baseline_ADE2016_MulRefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1_S2_1_La_1_adv_0.005/MulRefNet_100epoch_2526_latest.pth'
#RESTORE_FROM ='./checkpoints/baseline_ADE2016_MulRefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1.0_S2_1.0_La_1.0_adv_0.005/MulRefNet_100epoch_2526_latest.pth'
#RESTORE_FROM ='./checkpoints/baseline_ADE2016_MulRefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1.0_S2_1.0_La_1.0_adv_0.01/MulRefNet_100epoch_2526_latest.pth'
#RESTORE_FROM ='./checkpoints/baseline_ADE2016_MulRefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1.0_S2_1.0_La_1.0_adv_0.05/MulRefNet_100epoch_2526_latest.pth'
#RESTORE_FROM ='./checkpoints/baseline_ADE2016_MulRefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1.0_S2_1.0_La_1.0_adv_0.015/MulRefNet_100epoch_2526_latest.pth'
#RESTORE_FROM ='./checkpoints/baseline_ADE2016_MulRefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1.0_S2_1.0_La_1.0_adv_0.02/MulRefNet_100epoch_2526_latest.pth'

RESTORE_FROM ='./checkpoints/baseline_ADE2016_MulRefNet_Ngpu1_batchSize8_epochs100_imgSize348_segSize348_nclasses150_S1_1_S2_1_La_1_adv_0.015/MulRefNet_100epoch_2526_latest.pth'

def _get_model_instance(name):
    try:
        return{
                'ResNet34_PyP' : ResNet34_PyP,
                'ResNet50_PyP' : ResNet50_PyP,
                'ResNet101_PyP' : ResNet101_PyP,
                'ResNet34_ASPP' : ResNet34_ASPP,
                'ResNet50_ASPP' : ResNet50_ASPP,
                'ResNet101_ASPP' : ResNet101_ASPP,
                'RefNet' : RefNet,
                'RefNet_UP' : RefNet_UP,
                'MulRefNet' : MulRefNet,
                'MulRefNet_UP' : MulRefNet_UP,
                'MulRefNet_SELU' : MulRefNet_SELU
                }[name]
    except:
        print('Model {} not available'.format(name))
           
def get_model(name, num_classes):
    model = _get_model_instance(name)
    
    if name == 'ResNet34_PyP':
        model = model(num_classes = num_classes)
    elif name == 'ResNet50_PyP':
        model = model(num_classes = num_classes)
    elif name == 'ResNet101_PyP':
        model = model(num_classes = num_classes)
    elif name == 'ResNet34_ASPP':
        model = model(num_classes = num_classes)
    elif name == 'ResNet50_ASPP':
        model = model(num_classes = num_classes)
    elif name == 'ResNet101_ASPP':
        model = model(num_classes = num_classes)
    elif name == 'RefNet':
        model = model(num_classes = num_classes)
    elif name == 'RefNet_UP':
        model = model(num_classes = num_classes)
    elif name == 'MulRefNet':
        model = model(num_classes = num_classes)
    elif name == 'MulRefNet_UP':
        model = model(num_classes = num_classes)
    elif name == 'MulRefNet_SELU':
        model = model(num_classes = num_classes)
    else:
        model = model(num_classes = num_classes)
    return model

####################################################################
# forward func for evalution
def forward_multiscale(nets, batch_data, args):
    (model, interp, crit) = nets
    (imgs, segs, infos) = batch_data
    
    #segSize = (segs.size(1), segs.size(2))
    
    pred1 = torch.zeros(imgs.size(0), args.num_classes, segs.size(1), segs.size(2))
    pred1 = Variable(pred1, volatile=True).cuda()
    
    pred2 = torch.zeros(imgs.size(0), args.num_classes, segs.size(1), segs.size(2))
    pred2 = Variable(pred2, volatile=True).cuda()
    
    #pred_outputs = torch.zeros(imgs.size(0), args.num_classes, segs.size(1), segs.size(2))
    #pred_outputs = Variable(pred_outputs, volatile=True).cuda()
    
    for scale in args.scales:
        imgs_scale = zoom(imgs.numpy(), 
                          (1., 1., scale, scale),
                          order=1,
                          prefilter=False,
                          mode='nearest')
        
        # feed input data
        input_img = Variable(torch.from_numpy(imgs_scale), volatile=True).cuda()
        
        # forward
        #pred1_scale , pred2_scale, pred_outputs_scale= model(input_img)
        pred1_scale , _, pred2_scale = model(input_img)
        
        pred1_scale = interp(pred1_scale)
        pred1_scale = F.softmax(pred1_scale)
        
        pred2_scale = interp(pred2_scale)
        pred2_scale = F.softmax(pred2_scale)
        
        #pred_outputs_scale = interp(pred_outputs_scale)
        #pred_outputs_scale = F.softmax(pred_outputs_scale)
        
        pred1 = pred1 + pred1_scale / len(args.scales)     # --> [ B x 150 x 321 x 321 ]
        pred2 = pred2 + pred2_scale / len(args.scales)
        #pred_outputs = pred_outputs + pred_outputs_scale / len(args.scales)
        
        
    pred1 = torch.log(pred1)
    pred2 = torch.log(pred2)
    #pred_outputs = torch.log(pred_outputs)
    
    label_seg = Variable(segs.long(), volatile=True).cuda()
    
    loss_pred1 = crit(pred1, label_seg)
    loss_pred2 = crit(pred2, label_seg)
    #loss_pred_outputs = crit(pred_outputs, label_seg)
    
    #return pred1, pred2, pred_outputs, loss_pred1, loss_pred2, loss_pred_outputs
    return pred1, pred2, loss_pred1, loss_pred2
    

def visualize_result(batch_data, pred1, pred2, args):
    colors = loadmat('datasets/mit_list/color150.mat')['colors']
    (imgs, segs, infos) = batch_data
    for j in range(len(infos)):
        # get/recover image
        # img = imread(os.path.join(args.root_img, infos[j]))
        img = imgs[j].clone()
        for t, m, s in zip(img, 
                           [0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225]):
            t.mul_(s).add_(m)
        img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        #img = imresize(img, (args.imgSize, args.imgSize), interp='bilinear')
        
        # segmentation
        lab = segs[j].numpy()
        lab_color = colorEncode(lab, colors)
        #lab_color = imresize(lab_color, (args.imgSize, args.imgSize), interp='nearest')
        
        # prediction
        #print('#############')
        #print(pred1)
        pred1_ = np.argmax(pred1.data.cpu()[j].numpy(), axis=0)
        #print('**************')
        #print(pred1_)
        #print(pred1_.size())
        pred1_color = colorEncode(pred1_, colors)
        #print('&&&&&&&&&&&&&&&&&')
        #print(pred1_color)
        #pred1_color = imresize(pred1_color, (args.imgSize, args.imgSize), interp='nearest')
        
        pred2_ = np.argmax(pred2.data.cpu()[j].numpy(), axis=0)
        pred2_color = colorEncode(pred2_, colors)
        #pred2_color = imresize(pred2_color, (args.imgSize, args.imgSize), interp='nearest')
        
        #pred_outputs_ = np.argmax(pred_outputs.data.cpu()[j].numpy(), axis=0)
        #pred_outputs_color = colorEncode(pred_outputs_, colors)
        #pred2_color = imresize(pred2_color, (args.imgSize, args.imgSize), interp='nearest')
        
        
        # aggregate images and save
        im_vis = np.concatenate((img, lab_color, pred1_color, pred2_color), axis=1).astype(np.uint8)
        imsave(os.path.join(args.result, 
                            infos[j].replace('/', '_')
                            .replace('.jpg', '.png')), im_vis)



def evaluate(nets, loader, args):
    loss_pred1_meter = AverageMeter()
    loss_pred2_meter = AverageMeter()
    #loss_pred_outputs_meter = AverageMeter()
    
    acc_pred1_meter = AverageMeter()
    acc_pred2_meter = AverageMeter()
    #acc_pred_outputs_meter = AverageMeter()
    
    intersection_pred1_meter = AverageMeter()
    intersection_pred2_meter = AverageMeter()
    #intersection_pred_outputs_meter = AverageMeter()
    
    union_pred1_meter = AverageMeter()
    union_pred2_meter = AverageMeter()
    #union_pred_outputs_meter = AverageMeter()
    
    for model in nets:
        model.eval()
        
    for i, batch_data in enumerate(loader):
        # forward pass
        if i % 100 == 0:
            print('{:d} processd'.format(i))
           
        #pred1, pred2, pred_outputs, loss_pred1, loss_pred2, loss_pred_outputs = forward_multiscale(nets, batch_data, args)
        pred1, pred2, loss_pred1, loss_pred2 = forward_multiscale(nets, batch_data, args)
        loss_pred1_meter.update(loss_pred1.data[0])
        loss_pred2_meter.update(loss_pred2.data[0])
        #loss_pred_outputs_meter.update(loss_pred_outputs.data[0])
        
        # calculate accuracy
        acc_pred1, pix_pred1 = accuracy(batch_data, pred1)
        intersection_pred1, union_pred1 = intersectionAndUnion(batch_data, pred1, args.num_classes)
        
        acc_pred2, pix_pred2 = accuracy(batch_data, pred2)
        intersection_pred2, union_pred2 = intersectionAndUnion(batch_data, pred2, args.num_classes)
        
        #acc_pred_outputs, pix_pred_outputs = accuracy(batch_data, pred_outputs)
        #intersection_pred_outputs, union_pred_outputs = intersectionAndUnion(batch_data, pred_outputs, args.num_classes)
        
        acc_pred1_meter.update(acc_pred1, pix_pred1)
        intersection_pred1_meter.update(intersection_pred1)
        union_pred1_meter.update(union_pred1)
        
        acc_pred2_meter.update(acc_pred2, pix_pred2)
        intersection_pred2_meter.update(intersection_pred2)
        union_pred2_meter.update(union_pred2)
        
        #acc_pred_outputs_meter.update(acc_pred_outputs, pix_pred_outputs)
        #intersection_pred_outputs_meter.update(intersection_pred_outputs)
        #union_pred_outputs_meter.update(union_pred_outputs)
        
        print('[{}] iter {}, loss_pred1: {} loss_pred2: {}, Accurarcy_pred1: {} Accurarcy_pred2: {}'
              .format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), i, loss_pred1.data[0], loss_pred2.data[0], acc_pred1, acc_pred2))
       
        
        # visualization
        if args.visualize:
            visualize_result(batch_data, pred1, pred2, args)
           
    iou_pred1 = intersection_pred1_meter.sum / (union_pred1_meter.sum + 1e-10)
    iou_pred2 = intersection_pred2_meter.sum / (union_pred2_meter.sum + 1e-10)
    #iou_pred_outputs = intersection_pred_outputs_meter.sum / (union_pred_outputs_meter.sum + 1e-10)
    '''
    for i , _iou_pred1 in enumerate(iou_pred1):
        for j, _iou_pred2 in enumerate(iou_pred2):
                for k, _iou_pred_outputs in enumerate(iou_pred_outputs):
                    if k == (j == i):
                    #print('class [{}], IoU_pred1: {}, IoU_pred2: {}, IoU_pred_outputs: {}'.format(i, _iou_pred1, _iou_pred2, _iou_pred_outputs) )
                    
                    print('class [{}], IoU_pred1: {}, IoU_pred2: {}'.format(i, _iou_pred1, _iou_pred2)) 
                    break
    
    for i, _iou_pred1, _iou_pred2, _iou_pred_outputs in list(zip(iou_pred1, iou_pred2, iou_pred_outputs )):
        print('class [{}], IoU_pred1: {}, IoU_pred2: {}'.format(i, _iou_pred1, _iou_pred2))
    '''
    iou = list(zip(iou_pred1, iou_pred2))
    for i, (_iou_pred1, _iou_pred2) in enumerate(iou):
        print('class [{}],\n IoU_pred1: {},\n IoU_pred2: {}\n'.format(i, _iou_pred1, _iou_pred2))
        #print('class [{}],\n IoU_pred1: {},\n IoU_pred2: {},\n IoU_pred_outputs: {}\n'.format(i, _iou_pred1, _iou_pred2, _iou_pred_outputs))
    
    
    print('[Eval Summary]:')
    print('Loss_pred1: {},\n Loss_pred2: {},\n Mean IoU_pred1: {:.2f}%,\n Mean IoU_pred2: {:.2f}%,\n  Accurarcy_pred1: {:.2f}%,\n Accurarcy_pred2: {:.2f}%,\n'
          .format(loss_pred1_meter.average(), loss_pred2_meter.average(), iou_pred1.mean()*100, iou_pred2.mean()*100, acc_pred1_meter.average()*100, acc_pred2_meter.average()*100))
    
    
    
def main(args):
    # args = get_arguments()
    
    if not os.path.exists(args.result):
        os.makedirs(args.result)
        
    # create network
    model = get_model(name=args.generatormodel, num_classes = args.num_classes)
    
    if args.pretrained_model != None:
            args.restore_from = pretrianed_models_dict[args.pretrainned_model]

            
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
        
    model.load_state_dict(saved_state_dict)
    
    """
    if args.model == 'DeepLab':
        model = TwinsAdvNet_DL(num_classes = args.num_classes)
        if args.pretrained_model != None:
            args.restore_from = pretrained_models_dict[args.pretrained_model]
            
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
            
    model.load_state_dict(saved_state_dict)
    """
    
    """
    # load nets into gpu
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=range(args.num_gpus))
    model.cuda()
    """
    # crit = nn.NLLLoss2d(ignore_index=-1)  # ade20k
    #crit = nn.CrossEntropyLoss(ignore_index = -1)
    #crit = CrossEntropyLoss2d()
    crit = CrossEntropyLoss2d(ignore_index = -1)
    
    # interp = nn.Upsample(size=(384, 384), mode='bilinear')
    interp = nn.Upsample(size=(args.segSize, args.segSize), mode='bilinear')
    
    #train_dataset = MITSceneParsingDataset(args.list_train, args, is_train=1)
    val_dataset = MITSceneParsingDataset(args.list_val, args, max_sample=args.num_val, is_train=0)
    
    #val_dataset_size = len(val_dataset)
    #args.epoch_iters = int(train_dataset_size / (args.batch_size * args.num_gpus))
    #print('train_dataset_size = {} | 1 Epoch = {} iters'.format(train_dataset_size, args.epoch_iters))
  
    val_loader = data.DataLoader(val_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=int(args.workers),
                                  pin_memory=True,
                                  drop_last=True)
    
    nets = (model, interp, crit)
    
    """
    for model in nets:
        # load nets into gpu
        if args.num_gpus > 1:
            model = torch.nn.DataParallel(model, device_ids=range(args.num_gpus))
        model.cuda()
    """
    for model in nets:
        model.cuda()
    # Main loop
    evaluate(nets, val_loader, args)
    
    print('Evaluation Done!')
    

    
    




if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    # Model related arguments
    parser.add_argument("--generatormodel", type=str, default='MulRefNet',
                        help="available options : ResNet50_ASPP, RefNet, MulRefNet...")
    
    # optimization related arguments
    
    
    # Data related arguments
    parser.add_argument("--num_val", type=int, default= -1, # -1
                        help="Number of images to evalutate.")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="number of gpus to use.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=1,
                        help="input batch size.")
    parser.add_argument("--num_classes", type=int, default=150,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--workers", type=int, default=1,
                        help="number of data loading workers.")
    parser.add_argument("--imgSize", type=int, default=348, # 384 -1
                        help="input image size. -1 = keep original")
    parser.add_argument("--segSize", type=int, default=348, # 384 -1
                        help="output image size. -1 = keep original")
    """
    parser.add_argument("--max_iters", type=int, default=5052*100,  # 20210
                        help="args.max_iters = args.epoch_iters * args.num_epoches.")
    """
    # path related arguments
    parser.add_argument("--restore-from", type=str, default= RESTORE_FROM ,
                        help="Where restore model parameters from.")
    parser.add_argument("--pretrained-model", type=str, default=None,
                        help="Where restore model parameters from.")
    
    parser.add_argument("--id", type=str, default='ADE2016_3epoch_5052_4BN',
                        help="a name for identifying the model to load.")
    
    
    parser.add_argument("--list_val",
                        default='./datasets/mit_list/ADE20K_object150_val.txt')
    parser.add_argument("--root_img",
                        default='./data/ADEChallengeData2016/images')
    parser.add_argument("--root_seg",
                        default='./data/ADEChallengeData2016/annotations')
    
    # Misc arguments
    parser.add_argument("--visualize", default=1,
                        help="output visualation? 0 or 1")
    parser.add_argument("--result", default='./result/val0.02',
                        help="folder to output visualization results.")
    
    args = parser.parse_args()
    print(args)
    
    # scale for evaluation
    # args.scales = (1, )
    #args.scales = (0.5, 0.75, 1, 1.25, 1.5)
    args.scales = (0.5, 0.75, 1, 1.25 ,1.5)
    
    args.result = os.path.join(args.result, args.generatormodel)
    args.batch_size = args.batch_size_per_gpu * args.num_gpus
    main(args)
   