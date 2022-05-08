#!/user/bin/env python
# -*- coding:utf-8 -*-
__Author__ = 'Zijie Zhao'
__Create__ = '05/08/2022'


import os
import time
import argparse
import collections
from distutils.version import LooseVersion
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from scene_seg.dataset import ValDataset
from scene_seg.models import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large
from scene_seg.utils import AverageMeter, user_scattered_collate, colorEncode, accuracy, intersectionAndUnion, async_copy_to


colors = loadmat('data/color150.mat')['colors']


def as_numpy(obj):
    if isinstance(obj, collections.Sequence):
        return [as_numpy(v) for v in obj]
    elif isinstance(obj, collections.Mapping):
        return {k: as_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, Variable):
        return obj.data.cpu().numpy()
    elif torch.is_tensor(obj):
        return obj.cpu().numpy()
    else:
        return np.array(obj)
    
    
def visualize_result(data, pred, dir_result):
    (img, seg, info) = data

    # segmentation
    seg_color = colorEncode(seg, colors)

    # prediction
    pred_color = colorEncode(pred, colors)

    # aggregate images and save
    im_vis = np.concatenate((img, seg_color, pred_color),
                            axis=1).astype(np.uint8)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(os.path.join(dir_result, img_name.replace('.jpg', '.png')))


def evaluate(segmentation_module, loader, gpu):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['seg_label'][0])
        img_resized_list = batch_data['img_data']

        torch.cuda.synchronize()
        tic = time.perf_counter()
        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, 150, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                print('shape', feed_dict['img_data'].shape, feed_dict['seg_label'].shape)
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)
                
                # forward pass
                scores_tmp = segmentation_module(feed_dict['img_data'])['out']
#                 scores = scores + scores_tmp / len([300, 375, 450, 525, 600])
                scores = scores_tmp

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())
#             print(pred.shape)
#             print(seg_label.shape)

        torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)

        # calculate accuracy
        acc, pix = accuracy(pred, seg_label)
        intersection, union = intersectionAndUnion(pred, seg_label, 150)
        acc_meter.update(acc, pix)
        intersection_meter.update(intersection)
        union_meter.update(union)
       
        pbar.update(1)

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))


def main(segmentation_module, gpu):
#     torch.cuda.set_device(gpu)

    criterion = nn.NLLLoss(ignore_index=-1)

    # Dataset and Loader
    DATASET = {'root_dataset': "./data/", 
              'list_train': "./data/training.odgt",
              'list_val': "./data/validation.odgt", 
              'num_class': 150, 
              'imgSizes': (300, 375, 450, 525, 600), 
              'imgMaxSize': 1000, 
              'padding_constant': 8, 
              'segm_downsampling_rate': 8, 
              'random_flip': True}
    dataset_val = ValDataset(
        "./data/",
        "./data/validation.odgt",
        DATASET)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=8,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=2,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, gpu)

    print('Evaluation Done!')


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
segmentation models on MIT SceneParse150 benchmark dataset. This tool can be 
used for both image and video scence segmenation. This tool is also the main 
part of MIT 6.869 project, which is under active development.

Currently, we provide three model options:
- deeplabv3_resnet50
- deeplabv3_mobilenet_v3_large
- lraspp_mobilenet_v3_large
 
            ''')
    print('Now you are entering the [training] mode...')

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--gpus",
        default=0,
        help="gpu to use"
    )
    parser.add_argument(
        "--model",
        default=0,
        help="which model to load"
    )
    parser.add_argument(
        "--ckpt",
        default=0,
        help="checkpoint path to load"
    )
    args = parser.parse_args()
    model = args.model
    ckpt_path = args.ckpt
    if model == 'deeplabv3_resnet50':    
            segmentation_module = deeplabv3_resnet50(outputchannels=150, keep_feature_extract=False, use_pretrained=True)
    elif model == 'deeplabv3_mobilenet_v3_large':
        segmentation_module = deeplabv3_mobilenet_v3_large(outputchannels=150, keep_feature_extract=False, use_pretrained=True)
    elif model == 'lraspp_mobilenet_v3_large':
        segmentation_module = lraspp_mobilenet_v3_large(outputchannels=150, keep_feature_extract=False, use_pretrained=True)
    else:
        raise NameError
    segmentation_module.load_state_dict(torch.load(ckpt_path))


    main(segmentation_module, args.gpus)