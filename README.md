# Cross-Modality Asymmetric Mutual Learning
Asymmetric Mutual Learning for Unsupervised Transferable Visible-Infrared Re-Identification (IEEE TCSVT 2024)  
[[paper](https://ieeexplore.ieee.org/document/10538304)].

This repository contains the code for training and evaluation on
[SYSU-MM01](https://github.com/wuancong/SYSU-MM01),
[RegDB](https://www.mdpi.com/1424-8220/17/3/605)
and
**SYSU-MM02** (new dataset introduced in this paper).

## Dataset description

The SYSU-MM02 dataset is an RGB-Infrared (IR) multi-modality pedestrian dataset for unsupervised cross-modality person re-identification.
The training set is constructed by pedestrian detection on untrimmed videos, which contains more noises than the training sets of existing manually labeled visible-infrared pedestrian benchmark datasets.
The training set contains 7,440 visible images and 13,614 near infrared images. The testing set contains 2,360 visible images and 2,360 near infrared images of 118 identities.


### SYSU-MM02 dataset download  
Please send a signed [dataset release agreement](agreement/agreement.pdf) copy to wuanc@mail.sysu.edu.cn.
If your application is passed, we will send the download link of the dataset.
The pedestrian captured in "examples" folder of SYSU-MM02 has signed a privacy license to allow the images to be used for scientific research and shown in research papers. The other pedestrians are not allowed to be shown publicly, whose faces are masked to avoid privacy problem.

## Dependencies

The implementation is based on pytorch 1.13.1+cu117 and 1 NVIDIA RTX 3090.
The required packages are listed in `requirements.txt`.
```sh
pip install -r requirements.txt
```

## Data preparation
* SYSU-MM02  
Apply for the dataset following the instruction above.
Download the dataset and unzip it in `./datasets/SYSU-MM02`.

* market1501_gen  
`market1501_gen.zip` [[Baiduyun]](https://pan.baidu.com/s/1sdNNrDgiNaOwg9b72TufcA?pwd=e33e) [[Google Drive]](https://drive.google.com/file/d/1BJePq0rNSI44RaiJE09rhsJZ13SJz8pH/view?usp=sharing) contains fake infrared images of Market-1501 transformed by diffusion model, which is required for training in our method.
Download and unzip it in `./datasets/market1501_gen`. 

* SYSU-MM01  
Apply for the dataset following the instruction in https://github.com/wuancong/SYSU-MM01.
Download the dataset and put the image folders in `./datasets/SYSU-MM01`.

* RegDB  
Apply for the dataset following the instruction in [Nguyen et al., Sensors 2017](https://www.mdpi.com/1424-8220/17/3/605).  
Download and unzip the dataset in `./datasets/RegDB`.

## Model training and testing

We provide examples of training scripts for SYSU-MM01, SYSU-MM02 and RegDB in `./scripts`.
Evaluation results are saved in `./logs`.

## Acknowledgement
We based our codes on [JDAI-CV/fast-reid](https://github.com/JDAI-CV/fast-reid). 

## Citation

If you find the SYSU-MM02 dataset or this code useful, please cite

```
Ancong Wu, Chengzhi Lin, Wei-Shi Zheng. Asymmetric Mutual Learning for Unsupervised Transferable Visible-Infrared Re-Identification. 
IEEE Transactions on Circuits and Systems for Video Technology (IEEE TCSVT), 2024.
```
