#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:20:55 2018

@author: bobo
"""
# System libs
import os
import argparse

# Numerical libs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from scipy.io import loadmat
from scipy.misc import imread, imresize, imsave
from scipy.ndimage import zoom
from torch.utils import model_zoo

# our libs
from utils.utils import colorEncode
from genertors.RefNet import ResNet34_PyP, ResNet50_PyP, ResNet101_PyP, ResNet34_ASPP, ResNet50_ASPP, ResNet101_ASPP, RefNet_PyP, RefNet_ASPP, RefNet_RP, MulRefNet, MulRefNet_DL, MulRefNet_SL, MulRefNet_DELU, MulRefNet_SELU

RESTORE_FROM = './checkpoints//baseline_ADE2016_ResNet50_ASPP_Ngpu1_batchSize8_epochs100_imgSize275_segSize275_nclasses150_G1_1_G2_0_G12_0.2_Outputs_1_adv1_0.005_adv2_0.005/ResNet50_ASPP_2526_train_best.pth'


def _get_model_instance(name):
    try:
        return{
                'ResNet34_PyP' : ResNet34_PyP,
                'ResNet50_PyP' : ResNet50_PyP,
                'ResNet101_PyP' : ResNet101_PyP,
                'ResNet34_ASPP' : ResNet34_ASPP,
                'ResNet50_ASPP' : ResNet50_ASPP,
                'ResNet101_ASPP' : ResNet101_ASPP,
                'RefNet_PyP' : RefNet_PyP,
                'RefNet_ASPP' : RefNet_ASPP,
                'RefNet_RP' : RefNet_RP,
                'MulRefNet' : MulRefNet,
                'MulRefNet_DL' : MulRefNet_DL,
                'MulRefNet_SL' : MulRefNet_SL,        
                'MulRefNet_DELU' : MulRefNet_DELU,
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
    elif name == 'RefNet_PyP':
        model = model(num_classes = num_classes)
    elif name == 'RefNet_ASPP':
        model = model(num_classes = num_classes)
    elif name == 'RefNet_RP':
        model = model(num_classes = num_classes)
    elif name == 'MulRefNet':
        model = model(num_classes = num_classes)
    elif name == 'MulRefNet_DL':
        model = model(num_classes = num_classes)
    elif name == 'MulRefNet_SL':
        model = model(num_classes = num_classes)
    elif name == 'MulRefNet_DELU':
        model = model(num_classes = num_classes)
    elif name == 'MulRefNet_SELU':
        model = model(num_classes = num_classes)
    else:
        model = model(num_classes = num_classes)
    return model

####################################################################
# forward func for testing
def forward_test_multiscale(nets, img, args):
    (model, interp) = nets
    #segSize = (segs.size(1), segs.size(2))
    #imgsize : (n, c, h, w))
    pred1 = torch.zeros(1, args.num_classes, img.size(2), img.size(3))
    pred1 = Variable(pred1, volatile=True).cuda()
    
    pred2 = torch.zeros(1, args.num_classes, img.size(2), img.size(3))
    pred2 = Variable(pred2, volatile=True).cuda()
    
    pred_outputs = torch.zeros(1, args.num_classes, img.size(2), img.size(3))
    pred_outputs = Variable(pred_outputs, volatile=True).cuda()
    
    for scale in args.scales:
        imgs_scale = zoom(img.numpy(), 
                          (1., 1., scale, scale),
                          order=1,
                          prefilter=False,
                          mode='nearest')
        
        # feed input data
        input_img = Variable(torch.from_numpy(imgs_scale), volatile=True).cuda()
        
        # forward
        pred1_scale , pred2_scale, pred_outputs_scale= model(input_img)
        
        #pred1_scale = interp(pred1_scale)
        pred1_scale = nn.functional.upsample(pred1_scale, size=(img.size(2), img.size(3)), mode='bilinear' )
        pred1_scale = F.softmax(pred1_scale)
        
        #pred2_scale = interp(pred2_scale)
        pred2_scale = nn.functional.upsample(pred2_scale, size=(img.size(2), img.size(3)), mode='bilinear')
        pred2_scale = F.softmax(pred2_scale)
        
        #pred_outputs_scale = interp(pred_outputs_scale)
        pred_outputs_scale = nn.functional.upsample(pred_outputs_scale, size=(img.size(2), img.size(3)), mode='bilinear')
        pred_outputs_scale = F.softmax(pred_outputs_scale)
        
        pred1 = pred1 + pred1_scale / len(args.scales)     # --> [ B x 150 x 321 x 321 ]
        pred2 = pred2 + pred2_scale / len(args.scales)
        pred_outputs = pred_outputs + pred_outputs_scale / len(args.scales)  
    
    return pred1, pred2, pred_outputs
    

def visualize_test_result(img, pred1, pred2, pred_outputs, args):
    colors = loadmat('datasets/mit_list/color150.mat')['colors']
    # recover image
    img = img[0]
    #pred1 = pred1.data.cpu()[0]
    #pred2 = pred2.data.cpu()[0]
    #pred_outputs = pred_outputs.data.cpu()[0]

    for t, m, s in zip(img, 
                       [0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    #img = imresize(img, (args.imgSize, args.imgSize), interp='bilinear')
    
    # prediction
    #print('#############')
    #print(pred1)
    pred1_ = np.argmax(pred1.data.cpu()[0].numpy(), axis=0) + 1
    #print('**************')
    #print(pred1_)
    #print(pred1_.size())
    pred1_color = colorEncode(pred1_, colors)
    #print('&&&&&&&&&&&&&&&&&')
    #print(pred1_color)
    #pred1_color = imresize(pred1_color, (args.imgSize, args.imgSize), interp='nearest')
    
    pred2_ = np.argmax(pred2.data.cpu()[0].numpy(), axis=0) + 1
    pred2_color = colorEncode(pred2_, colors)
    #pred2_color = imresize(pred2_color, (args.imgSize, args.imgSize), interp='nearest')
    
    pred_outputs_ = np.argmax(pred_outputs.data.cpu()[0].numpy(), axis=0) + 1
    pred_outputs_color = colorEncode(pred_outputs_, colors)
    #pred2_color = imresize(pred2_color, (args.imgSize, args.imgSize), interp='nearest')
    
    
    # aggregate images and save
    im_vis = np.concatenate((img, pred1_color, pred2_color, pred_outputs_color), axis=1).astype(np.uint8)
    imsave(os.path.join(args.result, 
                        os.path.basename(args.test_img) + '.png'), 
        im_vis)



def test(nets, args):
    #  Switch to eval mode
    for model in nets:
        model.eval()
        
    # loading image, resize, convert to tensor
    img = imread(args.test_img, mode='RGB')
    h, w = img.shape[0], img.shape[1]
    s = 1. * args.imgSize / min(h, w)
    #img = imresize(img, s, interp='nearest')
    img = imresize(img, s)
    img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    img = img_transform(img)
    print(img.size)
    img = img.view(1, img.size(0), img.size(1), img.size(2))
    
    # foward pass
    pred1, pred2, pred_outputs = forward_test_multiscale(nets, img, args)
    
    # visualization
    visualize_test_result(img, pred1, pred2, pred_outputs, args)
    
    
    
    
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
    val_loader = data.DataLoader(val_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=int(args.workers),
                                  pin_memory=True,
                                  drop_last=True)
    
    """
    interp = nn.Upsample(size=(args.segSize, args.segSize), mode='bilinear')
    
    nets = (model, interp)
    for model in nets:
        model.cuda()
    # Main loop
    test(nets, args)
    
    print('Done! Output is saved in {}'.format(args.result))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MulRefNet Network")
    # Model related arguments
    parser.add_argument("--generatormodel", type=str, default='ResNet50_ASPP',
                        help="available options : , MulRefNet_SL, MulRefNet_DL")
    
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
    parser.add_argument("--imgSize", type=int, default=384, # 384 -1
                        help="input image size. -1 = keep original")
    parser.add_argument("--segSize", type=int, default=-1, # 384 -1
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
    
    parser.add_argument("--test_img", type=str, default='./data/release_test/testing/ADE_test_00003273.jpg',
                        help="a name for identifying the image to load.")
    
    
    parser.add_argument("--list_val",
                        default='./datasets/mit_list/ADE20K_object150_val.txt')
    parser.add_argument("--root_img",
                        default='./data/ADEChallengeData2016/images')
    parser.add_argument("--root_seg",
                        default='./data/ADEChallengeData2016/annotations')
    
    # Misc arguments
    parser.add_argument("--visualize", default=1,
                        help="output visualation? 0 or 1")
    parser.add_argument("--result", default='./result/test',
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
   