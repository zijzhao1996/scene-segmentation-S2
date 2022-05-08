#!/user/bin/env python
# -*- coding:utf-8 -*-
__Author__ = 'Zijie Zhao'
__Create__ = '04/26/2022'


import os
import time
import random
import json
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import functools
from torchvision import transforms
from torchvision import models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from scene_seg.dataset import TrainDataset
from scene_seg.models import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large
from scene_seg.utils import AverageMeter, parse_devices, user_scattered_collate, UserScatteredDataParallel, patch_replication_callback



def pixel_acc(pred, label):
        _, preds = torch.max(pred, dim=1)
        valid = (label >= 0).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
        return acc
    

def checkpoint(model, history, epoch):
    print('Saving checkpoints...')
    dict_model = model.state_dict()
    torch.save(
        history,
        '{}/history_epoch_{}.pth'.format(DIR, epoch))
    torch.save(
        dict_model,
        '{}/model_epoch_{}.pth'.format(DIR, epoch))
    

def train(segmentation_module, iterator, optimizer, criterion, history, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()

    segmentation_module.train()

    # main loop
    tic = time.time()
    for i in range(5000):
        # load a batch of data
        batch_data = next(iterator)[0]
        batch_data['img_data'] = batch_data['img_data'].cuda()
        batch_data['seg_label'] = batch_data['seg_label'].cuda()
        data_time.update(time.time() - tic)
        segmentation_module.zero_grad()

        # adjust learning rate
        cur_iter = i + (epoch - 1) * 5000

        # # forward pass
        pred = segmentation_module(batch_data['img_data'])['out']
        loss = criterion(pred, batch_data['seg_label'])
        acc = pixel_acc(pred, batch_data['seg_label'])

        loss = loss.mean()
        acc = acc.mean()

        # Backward
        loss.backward()
        optimizer.step()
#         scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % 20 == 0:
            print('Epoch: [{}][{}/{}], Avg Train Time: {:.2f}, Avg Data time: {:.2f}, LR: {:.4f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, 5000,
                          batch_time.average(), data_time.average(), optimizer.param_groups[0]['lr'],
                          ave_acc.average(), ave_total_loss.average()))

            fractional_epoch = epoch - 1 + 1. * i / 5000
            history['train']['epoch'].append(fractional_epoch)
            history['train']['loss'].append(loss.data.item())
            history['train']['acc'].append(acc.data.item())
          
            writer.add_scalar('training loss',
                ave_total_loss.average(),
                cur_iter)
            writer.add_scalar('training accuracy',
                ave_acc.average(),
                cur_iter)



def main(gpus, start_epoch, model):
    # build model
    # start from checkpoint
    assert start_epoch >= 0, "Invalid start epoch."
    # start from scratch
    if start_epoch == 0:
        if model == 'deeplabv3_resnet50':    
            segmentation_module = deeplabv3_resnet50(outputchannels=150, keep_feature_extract=False, use_pretrained=True)
        elif model == 'deeplabv3_mobilenet_v3_large':
            segmentation_module = deeplabv3_mobilenet_v3_large(outputchannels=150, keep_feature_extract=False, use_pretrained=True)
        elif model == 'lraspp_mobilenet_v3_large':
            segmentation_module = lraspp_mobilenet_v3_large(outputchannels=150, keep_feature_extract=False, use_pretrained=True)
        else:
            raise NameError
    else:
        path = os.path.join(
            DIR, 'model_epoch_{}.pth'.format(start_epoch))
        print('Load checkpoint model {}'.format(start_epoch))
        print(path)
        assert os.path.exists(path), "checkpoint does not exitst!"
        model = deeplabv3_resnet50(outputchannels=150, keep_feature_extract=False, use_pretrained=True)
        model.load_state_dict(torch.load(path))
        segmentation_module = model
        
    DATASET = {'root_dataset': "./data/", 
              'list_train': "./data/training.odgt",
              'list_val': "./data/validation.odgt", 
              'num_class': 150, 
              'imgSizes': (300, 375, 450, 525, 600), 
              'imgMaxSize': 1000, 
              'padding_constant': 8, 
              'segm_downsampling_rate': 8, 
              'random_flip': True}
    dataset_train = TrainDataset(
            "./data/",
            "./data/training.odgt",
            DATASET,
            batch_per_gpu=2)
    loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=len(gpus),  # we have modified data_parallel
            shuffle=False,  # we do not use this param
            collate_fn=user_scattered_collate,
            num_workers=2,
            drop_last=True,
            pin_memory=True)
    print('1 Epoch = {} iters'.format(5000))

    # create loader iterator
    iterator_train = iter(loader_train)

#     # load nets into gpu
#     if len(gpus) > 1:
#         segmentation_module = UserScatteredDataParallel(
#             segmentation_module,
#             device_ids=gpus)
#         # For sync bn
#         patch_replication_callback(segmentation_module)

    # Set up optimizer
    optimizer = torch.optim.Adam(segmentation_module.parameters(), lr=1e-4, weight_decay=1e-4)
#     scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15, 20], gamma=1)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    segmentation_module.cuda()

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(start_epoch, num_epoch):
        print('============= Epoch {} ============='.format(epoch))
        train(segmentation_module, iterator_train, optimizer, criterion, history, epoch+1)
        print('============== already train ==============')
        # checkpointing
        checkpoint(segmentation_module, history, epoch+1)

    print('Training Done!')
    

if __name__ == '__main__':
    
    print('''
#############################################################################

               _____                            _____            
              / ____|                          / ____|           
             | (___   ___ ___ _ __   ___ _____| (___   ___  __ _ 
              \___ \ / __/ _ \ '_ \ / _ \______\___ \ / _ \/ _` |
              ____) | (_|  __/ | | |  __/      ____) |  __/ (_| |
             |_____/ \___\___|_| |_|\___|     |_____/ \___|\__, |
                                                            __/ |
                                                           |___/                                                                    

##############################################################################
Welcome to use Scene-Seg (S2)! This is a PyTorch implementation of semantic 
segmentation models on MIT ADE20K scene parsing dataset.
(http://sceneparsing.csail.mit.edu/). 

This tool is also the main part of MIT 6.869 project, which is under active 
development. 

-- Author: Zijie Zhao
-- Date: Apr 26 2022
            ''')
    print('Current use model: DeepLabV3 model with a ResNet-101 backbone.')
    
    
    random.seed(869)
    torch.manual_seed(869)
    writer = SummaryWriter('runs/deeplabv3_resnet101_experiment_1')
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Training"
    )
    parser.add_argument(
        "--gpus",
        default="0",
        help="gpus to use, e.g. 0-3 or 0,1,2,3"
    )
    parser.add_argument(
        "--dir",
        default="./ckpt/deeplabv3_resnet101",
        help="folder to save ckpt"
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=int,
        help="start epoch to train the models"
    )
    parser.add_argument(
        "--model",
        default="deeplabv3_resnet50",
        help="which models to train"
    )
    args = parser.parse_args()

    # Parse gpu ids
    gpus = parse_devices(args.gpus)
    gpus = [x.replace('gpu', '') for x in gpus]
    gpus = [int(x) for x in gpus]
    num_gpus = len(gpus)
    num_epoch = 10
    start_epoch = args.start_epoch
    DIR = args.dir
    if not os.path.isdir(DIR):
        os.makedirs(DIR)
    model = args.model
    main(gpus, start_epoch, model)

