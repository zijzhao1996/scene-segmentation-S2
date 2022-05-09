# scene-segmentation

<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">
 <a href="https://github.com/zijzhao1996/scene-segmentation">
    <img src="images/logo.jpg" alt="Logo" width="640" height="240">
  </a>

<h3 align="center">Scene Segmentation (S2)</h3>

  <p align="center">
    Transfer Learning for Scene Segmentation in SceneParse150 Benchmark  
    <br />
    <a href="https://github.com/zijzhao1996/scene-segmentation"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/zijzhao1996/scene-segmentation">View Demo</a>
    ·
    <a href="https://github.com/zijzhao1996/scene-segmentation">Report Bug</a>
    ·
    <a href="https://github.com/zijzhao1996/scene-segmentation">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#video segmentation">Usage</a></li>
    <li><a href="#outputs">Output demo</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Welcome to use Scene-Seg (S2)! This is a PyTorch implementation of semantic scene segmentation models on the [MIT SceneParse150 benchmark dataset](http://sceneparsing.csail.mit.edu/). This tool can be used for both image and video scene segmentation. This tool is also the main part of the MIT 6.869 project, which is under active development.

Currently, we provide three model options:
- deeplabv3_resnet50
- deeplabv3_mobilenet_v3_large
- lraspp_mobilenet_v3_large

Below is a segmentation video predicted by our deeplabv3_resnet50 model, a common situation used by self-driving cars. 

![gif](./images/video_seg.gif "Video segmentation")

We provide the following features:
- A full ML pipeline of training and inference with PyTroch.
- Image/video segmentation.
- Finetune/Feature extraction of 3 models.
- Dynamic scale of inputs across GPUs.
- Start training from existing checkpoints.
- Tensorboard with accuracy and loss monitoring.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

The code is developed under the following hardward and softward.

Hardware: 
- Google Colab with NVIDIA P100 GPU. 
- [Satori](https://mit-satori.github.io/), a GPU dense, high-performance Power 9 system developed as a collaboration between MIT and IBM. The GPU we use is NVIDIA Tesla V100.
Software: CUDA>=8.0, Python>=3.5, PyTorch>=0.4.0
Dependencies: numpy, scipy, opencv

Please make sure you computing environment has at least one available GPU.

<!-- USAGE EXAMPLES -->
## Usage

First, download the ADE20K dataset, which will download in your `./data ` folder.

```sh
chmod +x download_dataset.sh
./download_dataset.sh

```
For training, please excute following commands:

```python
python train.py  --gpus 0 --dir ./ckpt/$CHEKPOINT --start_epoch 0 --model $MODELNAME
```

For evaluation on the validation set (2,000 images) of SceneParse150 benchmark dataset, please excute following commands:

```python
python evaluation.py  --gpus 0 ./ckpt/$CHEKPOINT --model $MODELNAME
```

For video segmentation, please excute following commands:
```python
python video_seg.py  --gpus 0 ./ckpt/$CHEKPOINT --model $MODELNAME
```
<p align="right">(<a href="#top">back to top</a>)</p>


<!-- OUTPUTS DeMO -->
## Outputs

You can check the time, pixel-wise accuracy, and training loss for each 20 iterations during each epoch.

```
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
of semantic scene segmentation models on the MIT SceneParse150 
benchmark dataset. This tool can be used for both image and 
video scene segmentation. This tool is also the main part of 
the MIT 6.869 project, which is under active development.

Currently, we provide three model options:
- deeplabv3_resnet50
- deeplabv3_mobilenet_v3_large
- lraspp_mobilenet_v3_large

Now you are entering the [training] mode.      
Current use model: DeepLabV3 model with a ResNet-50 backbone.
2022-04-26 18:05:02.704803: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.1
# samples: 20210
1 Epoch = 5000 iters
Epoch: [1][0/5000], Avg Train Time: 1.10, Avg Data time: 0.00, Accuracy: 6.80, Loss: 4.704109
Epoch: [1][20/5000], Avg Train Time: 0.66, Avg Data time: 0.00, Accuracy: 18.12, Loss: 4.045720
Epoch: [1][40/5000], Avg Train Time: 0.59, Avg Data time: 0.00, Accuracy: 20.84, Loss: 3.658643
Epoch: [1][60/5000], Avg Train Time: 0.57, Avg Data time: 0.00, Accuracy: 22.26, Loss: 3.457493
Epoch: [1][80/5000], Avg Train Time: 0.55, Avg Data time: 0.00, Accuracy: 22.58, Loss: 3.395919
Epoch: [1][100/5000], Avg Train Time: 0.55, Avg Data time: 0.00, Accuracy: 23.51, Loss: 3.370975
Epoch: [1][120/5000], Avg Train Time: 0.55, Avg Data time: 0.00, Accuracy: 24.47, Loss: 3.315614
Epoch: [1][140/5000], Avg Train Time: 0.55, Avg Data time: 0.00, Accuracy: 25.27, Loss: 3.263218
......
```
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact
For any question, feel free to contact:

Zijie Zhao - zijiezha@mit.edu

<p align="right">(<a href="#top">back to top</a>)</p>
