#!/user/bin/env python
# -*- coding:utf-8 -*-
__Author__ = 'Zijie Zhao'
__Create__ = '05/08/2022'


import os
import time
import csv
import torch
import scipy.io
import PIL.Image
import torchvision.transforms
import argparse
import cv2
import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt
from scene_seg.utils import colorEncode
from scene_seg.models import deeplabv3_resnet50, deeplabv3_mobilenet_v3_large, lraspp_mobilenet_v3_large


colors = scipy.io.loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

def visualize_result(img, pred, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')
        
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)
#     display(PIL.Image.fromarray(im_vis))

def image_overlay(image, segmented_image):
    alpha = 1 # transparency for the original image
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(image, alpha, segmented_image, beta, gamma, image)
    return image

def video_seg(segmentation_module, gpu):
    segmentation_module.eval()
    segmentation_module.cuda()
    cap = cv2.VideoCapture('example_video.mp4')
    # get the frame width and height
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(f"example_video_output.mp4", 
                          cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                          (frame_width, frame_height))

    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second

    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])


    # read until end of video
    while(cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret:
            img_original = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_data = pil_to_tensor(img_original)
#             singleton_batch = {'img_data': img_data[None].cuda()}
            output_size = img_data.shape[1:]
            # get the start time
            start_time = time.time()
            with torch.no_grad():
                # get predictions for the current frame
                scores = segmentation_module(img_data[None].cuda())['out']
            _, pred = torch.max(scores, dim=1)
            pred = pred.cpu()[0].numpy()
            top_5_classes = np.bincount(pred.flatten()).argsort()[::-1][:5]
            top_5_classes_name = [names[index+1] for index in top_5_classes]
            # colorize prediction
            pred_color = colorEncode(pred, colors).astype(np.uint8)
            end_time = time.time()
            fps = 1 / (end_time - start_time)
            im_vis = image_overlay(img_original, pred_color)
            visualize_result(img_original, pred)
            total_fps += fps
            frame_count += 1
            print(f"Frame: {frame_count}, FPS:{fps:.3f} FPS")
            print(f"Top 5 classes: {top_5_classes_name[:5]}")
            cv2.putText(im_vis, f"Top 5 classes: {top_5_classes_name[:5]}", (20, 35),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            cv2.putText(im_vis, f"{fps:.3f} FPS", (20, 70),
                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
            out.write(im_vis) 
        else:
            break

    out.release()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    


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
    print('Now you are entering the [video segmentation] mode...')

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


    video_seg(segmentation_module, args.gpus)