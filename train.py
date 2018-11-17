#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 13:20:19 2018

@author: hi
"""

import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
#import pickle
from torch.autograd import Variable
import torch.optim as optim
#import scipy.misc
from scipy.io import loadmat
from scipy.misc import imsave, imresize
#from scipy.ndimage import zoom
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

#import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
import random
import time
import timeit
#import datetime


from utils.utils import AverageMeter, colorEncode, accuracy
from genertors.MulRefNet import ResNet34_PyP, ResNet50_PyP, ResNet101_PyP, ResNet34_ASPP, ResNet50_ASPP, ResNet101_ASPP, RefNet, RefNet_UP, MulRefNet, MulRefNet_UP, MulRefNet_SELU
from discriminators.discriminator import Discriminator
#from utils.loss import CrossEntropyLoss2d, BCEWithLogitsLoss2d
from utils.loss import CrossEntropyLoss2d
from datasets.mit_dataset import MITSceneParsingDataset

start_time = timeit.default_timer()

#RESTORE_FROM = './pretrained/MS_DeepLab_resnet_pretrained_COCO_init_modified.pth'
#RESTORE_FROM = './pretrained/resnet101COCO-41f33a49.pth'
RESTORE_FROM = './pretrained/resnet50-places365.pth'
#RESTORE_FROM = './pretrained/resnet34-places365.pth'
#RESTORE_FROM = './pretrained/resnet101-imagenet.pth'

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
        
#########################################################################
        
def create_optimizers(genertor, discriminator, criterion, args):
    optimizer_genertor = optim.SGD(genertor.parameters(), 
                                   lr = args.learning_rate, 
                                   momentum=args.momentum,
                                   weight_decay=args.weight_decay)
    optimizer_genertor.zero_grad()
    
    optimizer_discriminator = optim.Adam(discriminator.parameters(), 
                                   lr = args.learning_rate_D,
                                   betas=(0.9, 0.99))
    optimizer_discriminator.zero_grad()
    
    return optimizer_genertor, optimizer_discriminator
 
    
def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1. - float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    args.running_lr = lr_poly(args.learning_rate, i_iter, args.max_iters, args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr

def adjust_learning_rate_D(optimizer, i_iter):
    args.running_lr_D = lr_poly(args.learning_rate_D, i_iter, args.max_iters, args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.running_lr_D
        
   
#######################################################################

def visualize_tv(batch_data, pred1, pred2, args):
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
        img = imresize(img, (args.imgSize, args.imgSize), interp='bilinear')
        
        # segmentation
        lab = segs[j].numpy()
        lab_color = colorEncode(lab, colors)
        lab_color = imresize(lab_color, (args.imgSize, args.imgSize), interp='nearest')
        
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
        pred1_color = imresize(pred1_color, (args.imgSize, args.imgSize), interp='nearest')
        
        pred2_ = np.argmax(pred2.data.cpu()[j].numpy(), axis=0)
        pred2_color = colorEncode(pred2_, colors)
        pred2_color = imresize(pred2_color, (args.imgSize, args.imgSize), interp='nearest')
        
        #pred_out_ = np.argmax(pred_out.data.cpu()[j].numpy(), axis=0)
        #pred_out_color = colorEncode(pred_out_, colors)
        #pred_out_color = imresize(pred_out_color, (args.imgSize, args.imgSize), interp='nearest')
        # aggregate images and save
        im_vis = np.concatenate((img, lab_color, pred1_color, pred2_color), axis=1).astype(np.uint8)
        imsave(os.path.join(args.result, 
                            infos[j].replace('/', '_')
                            .replace('.jpg', '.png')), im_vis)
        
# train one epoch    
def train(genertor, discriminator, iterator, interp, optimizer, optimizer_D, criterion, criterion_bce, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    # laber for adversarial training
    S1_label = 0
    S2_label = 1
    
    genertor.train()
    discriminator.train()
        
    # main loop
    tic = time.time()
    for i_iter in range(args.epoch_iters):
        loss_seg_value_S1 = 0
        loss_seg_value_S2 = 0
        loss_seg_value_La = 0
        
        loss_adv_pred_value = 0  
        loss_D_value = 0
           
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, i_iter)
        
        optimizer_D.zero_grad()
        adjust_learning_rate_D(optimizer_D, i_iter)
            
        for param in discriminator.parameters():
            param.requires_grad = False
            
        _, batch_data = next(iterator)  # use  enumerate()
        data_time.update(time.time() - tic)
        # batch_data = next(trainloader_iter)   # use  iter()
        images, labels, infos = batch_data
        
        # images, labels, _ = batch_data
        # print(images, labels)
       
        # feed input data
        input_img = Variable(images, volatile = False) # train:False , val: True
        label_seg = Variable(labels.long(), volatile = False) # long() ???
        input_img = input_img.cuda()
        label_seg = label_seg.cuda()
        #print(label_seg)
        #print('input_img_size: {}, label_seg_size: {}'.format(input_img.size(), label_seg.size()))
        
        pred_S2, _, pred_S1 = genertor(input_img)
        
        pred_S1 = interp(pred_S1)     # --> [ B x 150 x 321 x 321 ]
        pred_S2 = interp(pred_S2)
        #print(pred_G2.size())
        #print(pred_G2.type())
        
        # input size (torch.Size([4, 150, 321, 321])) Target size (torch.Size([4, 321, 321]) 
        loss_seg_S1 = criterion(pred_S1, label_seg)
        loss_seg_S2 = criterion(pred_S2, label_seg)
    
        # produce mask
        #pred_label = pred_S2.data.cpu().numpy().argmax(axis=1)
        pred_label = pred_S1.data.cpu().numpy().argmax(axis=1)
        pred_label = torch.from_numpy(pred_label)
        pred_label = Variable(pred_label.long()).cuda()
        
        #loss_seg_La = criterion(pred_S2, pred_label)  # / 1.65
        loss_seg_La = criterion(pred_S2, label_seg)  # / 1.65

        D_out_S1 = interp(discriminator(F.softmax(pred_S1)))  # --> [B x 1 x 321 x 321]
        D_out_S2 = interp(discriminator(F.softmax(pred_S2)))
       
        #loss_adv_pred = criterion_bce(D_out_S1, Variable(torch.FloatTensor(D_out_S1.data.size()).fill_(S2_label)).cuda())       
        loss_adv_pred = criterion_bce(D_out_S2, Variable(torch.FloatTensor(D_out_S2.data.size()).fill_(S1_label)).cuda())
        
        loss_weakly = args.lambda_seg_La * loss_seg_La
        #loss_weakly = args.lambda_seg_La * (1 - (loss_seg_La / loss_seg_S2))**2
        
        #loss = args.lambda_seg_S1 * loss_seg_S1
        loss = args.lambda_seg_S1 * loss_seg_S1 + args.lambda_adv_pred * loss_adv_pred
        
        #loss = args.lambda_seg_S1 * loss_seg_S1 + args.lambda_adv_pred * loss_adv_pred + args.lambda_seg_La * loss_seg_La
        #loss = args.lambda_seg_S1 * loss_seg_S1 + args.lambda_adv_pred * loss_adv_pred + args.lambda_seg_La *  (1 - (loss_seg_La / loss_seg_S2))**2
        # proper normalization
        #loss_1.backward()  # detach()
        
        loss_weakly.backward(retain_graph=True)
        loss.backward()
        
        loss_seg_value_S1 += loss_seg_S1.data.cpu().numpy()[0] 
        loss_seg_value_S2 += loss_seg_S2.data.cpu().numpy()[0] 
        loss_seg_value_La += loss_seg_La.data.cpu().numpy()[0]        
        loss_adv_pred_value += loss_adv_pred.data.cpu().numpy()[0]   
        
        # train D
        # model_D.train()
        # optimizer_D.zero_grad()
        
        # bring back requires_grad
        for param in discriminator.parameters():
            param.requires_grad = True
        
        # train S1
        pred_S1 = pred_S1.detach()
        D_out_S1 = interp(discriminator(F.softmax(pred_S1)))        
        loss_D = criterion_bce(D_out_S1, Variable(torch.FloatTensor(D_out_S1.data.size()).fill_(S1_label)).cuda())      
        
        loss_D = loss_D / 2.0      
        loss_D.backward()
        loss_D_value += loss_D.data.cpu().numpy()[0]
        
        # train S2
        pred_S2 = pred_S2.detach()
        D_out_S2 = interp(discriminator(F.softmax(pred_S2)))
        loss_D = criterion_bce(D_out_S2, Variable(torch.FloatTensor(D_out_S2.data.size()).fill_(S2_label)).cuda())
        
        loss_D = loss_D / 2.0
        loss_D.backward()
        loss_D_value += loss_D.data.cpu().numpy()[0]
        
        optimizer.step()
        optimizer_D.step()
        
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        
        # calculate accuracy , mIOU, and display
        if i_iter % args.disp_iter == 0 :  # can not change
            acc_pred_outputs, pix_pred_outputs = accuracy(batch_data, pred_S1)               
            
            #print('exp = {}'.format(args.checkpoints_dir))
            print('iter =[{0:d}]/[{1:d}/{2:d}], Time: {3:.2f}, Data: {4:.2f}, loss_seg_S1 = {5:.4f} loss_seg_S2 = {6:.4f} loss_seg_La = {7:.4f}, loss_adv_pred = {8:.4f}, loss_D = {9:.4f}, Accurarcy: {10:4.2f}%'
                  .format(epoch, i_iter, args.epoch_iters, batch_time.average(), data_time.average(),
                          loss_seg_value_S1, loss_seg_value_S2, loss_seg_value_La, 
                          loss_adv_pred_value, loss_D_value,
                          acc_pred_outputs *100))
            
            fractional_epoch = epoch - 1 + 1. * i_iter / args.epoch_iters        
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss_pred_outputs'].append(loss_seg_S1.data[0])
            history['train']['acc_pred_outputs'].append(acc_pred_outputs)
                
        #  checkpoint          
        if epoch  == args.num_epoches and i_iter >= args.epoch_iters - 1:
            print ('taking checkpoints latest ...')
            torch.save(genertor.state_dict(), osp.join(args.checkpoints_dir,  str(args.generatormodel) + '_' + str(epoch) + 'epoch_' + str(args.epoch_iters)  + '_latest.pth'))
            torch.save(discriminator.state_dict(), osp.join(args.checkpoints_dir, str(args.generatormodel) + '_' + str(epoch) + 'epoch_' + str(args.epoch_iters) + '_D_latest.pth'))
        
        loss_seg_S1 = history['train']['loss_pred_outputs'][-1]
        if loss_seg_S1 < args.best_loss:
            args.best_loss = loss_seg_S1
            print ('taking checkpoints best ...')
            torch.save(genertor.state_dict(), osp.join(args.checkpoints_dir,  str(args.generatormodel) + '_' +  str(args.epoch_iters)  + '_train_best.pth'))
            torch.save(discriminator.state_dict(), osp.join(args.checkpoints_dir, str(args.generatormodel) + '_' + str(args.epoch_iters) + '_D_train_best.pth'))
        
        
def evaluate(genertor, val_loader, interp, criterion, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    loss_pred_outputs_meter = AverageMeter()
    acc_pred_outputs_meter = AverageMeter()
    
    # switch to eval mode
    genertor.eval()
    
    for i, batch_data in enumerate(val_loader):
        # forward pass
        #_, batch_data = next(iterator)  # use  enumerate()
        #data_time.update(time.time() - tic)
        # batch_data = next(trainloader_iter)   # use  iter()
        images, labels, infos = batch_data
        
        # images, labels, _ = batch_data
        # print(images, labels)
       
        # feed input data
        input_img = Variable(images, volatile = True) # train:False , val: True
        label_seg = Variable(labels.long(), volatile = True) # long() ???
        input_img = input_img.cuda()
        label_seg = label_seg.cuda()
        #print(label_seg)
        #print('input_img_size: {}, label_seg_size: {}'.format(input_img.size(), label_seg.size()))
        
        pred1, _, pred2 = genertor(input_img)
        
        pred1 = interp(pred1)     # --> [ B x 150 x 321 x 321 ]
        pred2 = interp(pred2)
        
        #pred1 = nn.functional.log_softmax(pred1)
        #pred2 = nn.functional.log_softmax(pred2)
        #pred_outputs = nn.functional.log_softmax(pred_outputs)
        
        loss_pred_outputs = criterion(pred2, label_seg)
        loss_pred_outputs_meter.update(loss_pred_outputs.data[0])
        print('[Eval] iter {}, loss_pred_outputs:{}'.format(i, loss_pred_outputs.data[0]))
        
        acc_pred_outputs, pix_pred_outputs = accuracy(batch_data, pred2)
        acc_pred_outputs_meter.update(acc_pred_outputs, pix_pred_outputs)
        
        if args.visualize:
            visualize_tv(batch_data, pred1, pred2, args)
        
    history['val']['epoch'].append(epoch)
    history['val']['loss_pred_outputs'].append(loss_pred_outputs_meter.average())
    history['val']['acc_pred_outputs'].append(acc_pred_outputs_meter.average())
  
    print('[Eval Summary] Epoch: {}, Loss: {}, Accurarcy: {:4.2f}%'
          .format(epoch, loss_pred_outputs_meter.average(), acc_pred_outputs_meter.average()*100))
    
    # plot figure
    if epoch > 0:
        print('Plotting loss figure...')
        fig = plt.figure()
        plt.plot(np.asarray(history['train']['epoch']),
                 np.log(np.asarray(history['train']['loss_pred_outputs'])),
                 color='b', label='training')
        
        plt.plot(np.asarray(history['val']['epoch']),
                 np.log(np.asarray(history['val']['loss_pred_outputs'])),
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Log(loss)')
        fig.savefig('{}/loss.png'.format(args.checkpoints_dir), dpi=200)
        plt.close('all')
        
        fig = plt.figure()
        plt.plot(history['train']['epoch'], 
                 history['train']['acc_pred_outputs'],
                 color='b', label='training')
        plt.plot(history['val']['epoch'], 
                 history['val']['acc_pred_outputs'],
                 color='c', label='validation')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        fig.savefig('{}/accuracy.png'.format(args.checkpoints_dir), dpi=200)
        plt.close('all')
    
    """
    # checkpoint val
    loss_seg_outputs = history['val']['loss_pred_outputs'][-1]
    if loss_seg_outputs < args.best_loss:
        args.best_loss = loss_seg_outputs
        print ('taking checkpoints ...')
        torch.save(genertor.state_dict(), osp.join(args.checkpoints_dir,  str(args.generatormodel) + '_' + str(args.epoch_iters)  + '_val_best.pth'))
        #torch.save(discriminator.state_dict(), osp.join(args.checkpoints_dir, str(args.generatormodel) + '_' + str(epoch) + 'epoch_' + str(args.epoch_iters) + '_D_best.pth'))
    """


def main(args):
    # random.seed(args.random_seed)
    # print(args)
    cudnn.enabled = True
    
    # create network
    """
    if args.generatormodel == 'TwinsAdvNet_D':
        model = get_model(name=args.generatormodel, num_classes = args.num_classes)
        #model = TwinsAdvNet_DL(num_classes = args.num_classes)
        if args.restore_from[:4] == 'http' :
            saved_state_dict = model_zoo.load_url(args.restore_from)
        else:
            saved_state_dict = torch.load(args.restore_from)
    """
            
    model = get_model(name=args.generatormodel, num_classes = args.num_classes)
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from)
        
    # only copy the params that exist in current model
    new_params = model.state_dict().copy()
    for name, param in new_params.items(): 
        #print(name)
        if name in saved_state_dict and param.size() == saved_state_dict[name].size():
            new_params[name].copy_(saved_state_dict[name])
            # print('copy {}'.format(name))
    model.load_state_dict(new_params)
      
    model.train()
    
    # load nets into gpu
    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=range(args.num_gpus))
    model.cuda()
    
    cudnn.benchmark = True  # acceleration
    
    # init D
    model_D = Discriminator(num_classes = args.num_classes)
    
    model_D.train()
    
    if args.num_gpus > 1:
        model_D = torch.nn.DataParallel(model_D, device_ids=range(args.num_gpus))
    model_D.cuda()
    
    
        
    #train_dataset = MITSceneParsingDataset(args.list_train, args, is_train=1)
    train_dataset = MITSceneParsingDataset(args.list_train, args, max_iters=args.max_iters, is_train=1)
    
    #train_dataset_size = len(train_dataset)
    #args.epoch_iters = int(train_dataset_size / (args.batch_size * args.num_gpus))
    #print('train_dataset_size = {} | 1 Epoch = {} iters'.format(train_dataset_size, args.epoch_iters))
      
    trainloader = data.DataLoader(train_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=int(args.workers),
                                  pin_memory=True,
                                  drop_last=True)
    
    val_dataset = MITSceneParsingDataset(args.list_val, args, max_sample=args.num_val, is_train=0)
    
    val_loader = data.DataLoader(val_dataset, 
                                  batch_size=args.batch_size,
                                  shuffle=False, # False
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=True)
    
    trainloader_iter = enumerate(trainloader)
    
    # loss / bilinear upsampling
    #bce_loss = BCEWithLogitsLoss2d()
    bce_loss = torch.nn.BCEWithLogitsLoss()  # only 0, 1
    
    #crit = nn.NLLLoss2d(ignore_index=-1)  # ade20k
    crit = CrossEntropyLoss2d(ignore_index=-1)

    
    # trainloader_iter = iter(trainloader)
    
    # implement model.optim_parameters(args) to handle different models' lr setting
       
    # optimizer for segmentation networks
    """
    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    """
    #optimizer.zero_grad()  #
    """
    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
    # optimizer for discriminator network
    optimizer_D = optim.Adam(model_D.parameters(),
                             lr=args.learning_rate_D,
                             betas=(0.9, 0.99))
    #optimizer_D.zero_grad()
    """
    optimizer, optimizer_D = create_optimizers(model, model_D, crit, args)
    #optimizer.zero_gard()
    #optimizer_D.zero_gard()
    
    # interp = nn.Upsample(size=(384, 384), mode='bilinear')
    interp = nn.Upsample(size=(args.imgSize, args.imgSize), mode='bilinear')
    # interp_x1 = nn.Upsample(size=(384, 384), mode='bilinear')  # G1
    # interp_x2 = nn.Upsample(size=(384, 384), mode='bilinear')  # G2
    
    # main loop
    history = { split: {'epoch': [], 'loss_pred_outputs': [], 'acc_pred_outputs': []} for split in ('train', 'val')}
    
    # initial eval
    evaluate(model,val_loader, interp, crit, history, 0, args)
    for epoch in range(args.start_epoch, args.num_epoches + 1):
        train(model, model_D, trainloader_iter, interp, optimizer, optimizer_D, crit, bce_loss, history, epoch, args)
        
        if epoch % args.eval_epoch == 0:
            evaluate(model,val_loader, interp, crit, history, epoch, args)
        
    end_time = timeit.default_timer()
    print(' running time(s): [{0:.4f} seconds]'.format((end_time - start_time)))
 
if __name__ == '__main__':
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    # Model related arguments
    parser.add_argument("--generatormodel", type=str, default='MulRefNet',
                        help="available options : ResNet50_ASPP, RefNet, MulRefNet...")
    parser.add_argument("--id", type=str, default='baseline',
                        help="a name for identifying the model")
        
    # optimization related arguments
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="number of gpus to use.")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8,
                        help="input batch size.")
    parser.add_argument("--num_epoches", type=int, default=100, # 100
                        help="epochs to train for.")
    parser.add_argument("--start_epoch", type=int, default=1, # 1
                        help="epochs to start training. useful if continue from a checkpoint .")
    parser.add_argument("--save_pred_every", type=int, default=100000, # 
                        help="Save summaries and checkpoint every often.")
    
    """
    parser.add_argument("--epoch_iters", type=int, default=10105,  # 20210 10105 5052
                        help="args.epoch_iters = int(train_dataset_size / (args.batch_size * args.num_gpus))") 
    parser.add_argument("--max_iters", type=int, default=10105*100,  # 20210
                        help="args.max_iters = args.epoch_iters * args.num_epoches.")
    parser.add_argument("--num_steps", type=int, default=50,  # 20210
                        help="Number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=10000,
                        help="Number of training steps for early stopping")
    """
    
    parser.add_argument("--learning-rate", type=float, default=3e-3, # 2.5e-4
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-D", type=float, default=1e-4,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    """
    parser.add_argument("--mask_T", type=float, default=0.1,
                        help="mask T for adversarial training, 0.1 - 0.3")
    """
    parser.add_argument("--best_loss", type=float, default= 1,
                        help="initialize with a big number")    
    
    parser.add_argument("--lambda_seg_S1", type=float, default= 1,
                        help="lambda_seg_S1.")
    parser.add_argument("--lambda_seg_S2", type=float, default= 1,
                        help="lambda_seg_S2.")
    parser.add_argument("--lambda_seg_La", type=float, default= 1,
                        help="lambda_seg_La.")
   
    parser.add_argument("--lambda_adv_pred", type=float, default=0.015,   # 0.008 0.01
                        help="lambda_adv for adversarial training.")    
    
    # Data related arguments
    parser.add_argument("--num_val", type=int, default= 128, # -1 128
                        help="Number of images to evalutate.")
    parser.add_argument("--num_classes", type=int, default=150,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--workers", type=int, default=16,
                        help="number of data loading workers.")
    parser.add_argument("--imgSize", type=int, default=348, # 384 321
                        help="input image size.")
    parser.add_argument("--segSize", type=int, default=348, # 384
                        help="output image size.") 
    
    # path related arguments
    parser.add_argument("--restore-from", type=str, default= RESTORE_FROM ,
                        help="Where restore model parameters from.")
    parser.add_argument("--checkpoints_dir", type=str, default='./checkpoints/',
                        help="Where to save checkpoints(ckpt) of the model.")
    
    parser.add_argument("--list_train",
                        default='./datasets/mit_list/ADE20K_object150_train.txt')
    parser.add_argument("--list_val",
                        default='./datasets/mit_list/ADE20K_object150_val.txt')
    parser.add_argument("--root_img",
                        default='./data/ADEChallengeData2016/images')
    parser.add_argument("--root_seg",
                        default='./data/ADEChallengeData2016/annotations')
       
    # Misc arguments
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible result")
    parser.add_argument("--disp_iter", type=int, default=20,
                        help="frequency to display.")
    parser.add_argument("--eval_epoch", type=int, default=1,
                        help="frequency to evaluate.")
    parser.add_argument("--visualize", default=1,
                        help="output visualation? 0 or 1")
    parser.add_argument("--result", default='./result/train',
                        help="folder to output visualization results.")
    
    args = parser.parse_args()
    
    args.batch_size = args.batch_size_per_gpu * args.num_gpus
    
    #train_dataset_size = len(train_dataset) = 20210
    #args.epoch_iters = int( 20210 / args.batch_size)  # 20210
    args.samples_size = 20210  # 20210
    args.epoch_iters = int( args.samples_size / args.batch_size)  # 20210
    
    #args.max_iters = args.epoch_iters * args.batch_size * args.num_epoches + 20
    args.max_iters = args.epoch_iters * args.num_epoches 
    #args = get_arguments()
    #args.max_iters = args.epoch_iters * args.num_epoches
    #random.seed(args.random_seed)
    args.id += '_ADE2016_' + str(args.generatormodel)
    args.id += '_Ngpu' + str(args.num_gpus)
    args.id += '_batchSize' + str(args.batch_size)
    args.id += '_epochs' + str(args.num_epoches)
    args.id += '_imgSize' + str(args.imgSize)
    args.id += '_segSize' + str(args.imgSize)
    args.id += '_nclasses' + str(args.num_classes)
    args.id += '_S1_' + str(args.lambda_seg_S1)
    args.id += '_S2_' + str(args.lambda_seg_S2)
    args.id += '_La_' + str(args.lambda_seg_La)
    args.id += '_adv_' + str(args.lambda_adv_pred)
    #args.id += '_' + str(args.)
    print('Model ID: {}'.format(args.id))
    args.checkpoints_dir = osp.join(args.checkpoints_dir, args.id)   
    args.result = os.path.join(args.result, args.generatormodel)
    
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
        
    if not os.path.exists(args.result):
        os.makedirs(args.result) 
    
    #args.best_loss_outputs = 2.e10  # initialize with a big number
    #print(args.best_loss_outputs)
    
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    main(args)  