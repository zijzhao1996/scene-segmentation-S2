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
import numpy as np
from torchvision import transforms
from torchvision import models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from scene_seg.dataset import TrainDataset
from scene_seg.models import createDeepLabv3
from scene_seg.utils import AverageMeter



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
    

def train(segmentation_module, iterator, optimizers, criterion, history, epoch):
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
        optimizers.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item()*100)

        # calculate accuracy, and display
        if i % 20 == 0:
            print('Epoch: [{}][{}/{}], Avg Train Time: {:.2f}, Avg Data time: {:.2f}, '
                  'Accuracy: {:4.2f}, Loss: {:.6f}'
                  .format(epoch, i, 5000,
                          batch_time.average(), data_time.average(),
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

    
# class UserScatteredDataParallel(DictGatherDataParallel):
#     def scatter(self, inputs, kwargs, device_ids):
#         assert len(inputs) == 1
#         inputs = inputs[0]
#         inputs = _async_copy_stream(inputs, device_ids)
#         inputs = [[i] for i in inputs]
#         assert len(kwargs) == 0
#         kwargs = [{} for _ in range(len(inputs))]

#         return inputs, kwargs

def user_scattered_collate(batch):
        return batch

# def patch_replication_callback(data_parallel):
#     """
#     Monkey-patch an existing `DataParallel` object. Add the replication callback.
#     Useful when you have customized `DataParallel` implementation.

#     Examples:
#         > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
#         > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
#         > patch_replication_callback(sync_bn)
#         # this is equivalent to
#         > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
#         > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
#     """

#     assert isinstance(data_parallel, DataParallel)

#     old_replicate = data_parallel.replicate

#     @functools.wraps(old_replicate)
#     def new_replicate(module, device_ids):
#         modules = old_replicate(module, device_ids)
#         execute_replication_callbacks(modules)
#         return modules

#     data_parallel.replicate = new_replicate


def main():
    gpus = [0]
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

    # load nets into gpu
    # if len(gpus) > 1:
    #     segmentation_module = UserScatteredDataParallel(
    #         segmentation_module,
    #         device_ids=gpus)
    #     # For sync bn
    #     patch_replication_callback(segmentation_module)

    # Set up optimizers
    segmentation_module = createDeepLabv3(outputchannels=150)
    optimizers = torch.optim.Adam(segmentation_module.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    segmentation_module.cuda()

    # Main loop
    history = {'train': {'epoch': [], 'loss': [], 'acc': []}}

    for epoch in range(0, 1):
        train(segmentation_module, iterator_train, optimizers, criterion, history, epoch+1)

        # checkpointing
        checkpoint(segmentation_module, history, epoch+1)

    print('Training Done!')
    

if __name__ == '__main__':
    
    print('''
            ###############################################################

                   _____                            _____            
                  / ____|                          / ____|           
                 | (___   ___ ___ _ __   ___ _____| (___   ___  __ _ 
                  \___ \ / __/ _ \ '_ \ / _ \______\___ \ / _ \/ _` |
                  ____) | (_|  __/ | | |  __/      ____) |  __/ (_| |
                 |_____/ \___\___|_| |_|\___|     |_____/ \___|\__, |
                                                                __/ |
                                                               |___/                                                                    

            ###############################################################
            Welcome to use Scene-Seg (S2)! This is a PyTorch implementation 
            of semantic segmentation models on MIT ADE20K scene parsing 
            dataset (http://sceneparsing.csail.mit.edu/). 

            Developing this tool is also the main part of MIT 6.869 project, 
            which is under active development.

            -- Author: Zijie Zhao
            -- Date: Apr 26 2022

            ''')
    print('Current use model: DeepLabV3 model with a ResNet-101 backbone.')
    # writer = SummaryWriter('runs/deeplabv3_resnet101_experiment_1')
    DIR = "./ckpt/deeplabv3_resnet101"
    if not os.path.isdir(DIR):
        os.makedirs(DIR)
    # parser = argparse.ArgumentParser(
    #     description="PyTorch Semantic Segmentation Training"
    # )

    # parser.add_argument(
    #     "--gpus",
    #     default="0-3",
    #     help="gpus to use, e.g. 0-3 or 0,1,2,3"
    # )
    random.seed(869)
    torch.manual_seed(869)
    writer = SummaryWriter('runs/deeplabv3_resnet101_experiment_1')
    main()

    # Start from checkpoint
    # start_epoch = 0
    # if start_epoch > 0:
    #     cfg.MODEL.weights_encoder = os.path.join(
    #         cfg.DIR, 'encoder_epoch_{}.pth'.format(start_epoch))
    #     cfg.MODEL.weights_decoder = os.path.join(
    #         cfg.DIR, 'decoder_epoch_{}.pth'.format(start_epoch))
    #     assert os.path.exists(cfg.MODEL.weights_encoder) and \
    #         os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # # Parse gpu ids
    # gpus = parse_devices(args.gpus)
    # gpus = [x.replace('gpu', '') for x in gpus]
    # gpus = [int(x) for x in gpus]
    # num_gpus = len(gpus)
    # batch_size_per_gpu = 2