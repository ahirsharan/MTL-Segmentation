### Note: Currently in Development
## Meta Transfer Learning for Few Shot Semantic Segmentation using U-Net
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

<!-- TOC -->

- [Meta Transfer Learning for Few Shot Semantic Segmentation using U-Net](#Meta-Transfer-Learning-for-Few-Shot-Semantic-Segmentation-using-U-Net)
  - [Requirements](#requirements)
  - [Characteristics](#Characteristics)
    - [Model and Technique](#Model-and-Technique)
    - [Datasets](#datasets)
    - [Losses](#losses)
  - [Code structure](#code-structure)
  - [Running Experiments](#Running-Experiments)
  - [Hyperparameters and Options](#Hyperparameters-and-Options)
  - [Training](#Training)
  - [Acknowledgement](#acknowledgement)

<!-- /TOC -->

## Requirements
PyTorch and Torchvision needs to be installed before running the scripts, together with `PIL` for data-preprocessing and `tqdm` for showing the training progress.

To run this repository, kindly install python 3.5 and PyTorch 0.4.0 with Anaconda.

You may download Anaconda and read the installation instruction on their official website:
<https://www.anaconda.com/download/>

Create a new environment and install PyTorch and torchvision on it:

```bash
conda create --name segfew python=3.5
conda activate segfew
conda install pytorch=0.4.0 
conda install torchvision -c pytorch
```

Clone this repository:

```bash
git clone https://github.com/ahirsharan/MTL_Segmentation.git
```
## Characteristics: 

### Model and Technique
- (**U-Net**) Convolutional Networks for Biomedical Image Segmentation (2015): [[Paper]](https://arxiv.org/abs/1505.04597)
- (**Meta Tranfer Learning**) Meta-Transfer Learning for Few-Shot Learning: [[Paper]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Meta-Transfer_Learning_for_Few-Shot_Learning_CVPR_2019_paper.pdf)

### Datasets

- **COCO Stuff:** For COCO, there is two partitions, CocoStuff10k with only 10k that are used for training the evaluation, note that this dataset is outdated, can be used for small scale testing and training, and can be downloaded [here](https://github.com/nightrome/cocostuff10k). For the official dataset with all of the training 164k examples, it can be downloaded from the official [website](http://cocodataset.org/#download).

- **Few-Shot:** For Few Shot, there are 1000 object classes folder each with 10 images with ground truth mask for segmentation. This dataset can be used for few shot learning and can be downloaded [here](https://drive.google.com/open?id=1oG5aOw4H4IyrbrBb0eWMCBOdDLRf9P1Y).

### Losses
In addition to the Cross-Entropy loss:

- **Dice-Loss**, which measures of overlap between two samples and can be more reflective of the training objective (maximizing the mIoU), but is highly non-convexe and can be hard to optimize.
- **CE Dice loss**, the sum of the Dice loss and CE, CE gives smooth optimization while Dice loss is a good indicator of the quality of the segmentation results.
- **Focal Loss**, an alternative version of the CE, used to avoid class imbalance where the confident predictions are scaled down.
- **Lovasz Softmax** lends it self as a good alternative to the Dice loss, where we can directly optimization for the mean intersection-over-union based on the convex Lovász extension of submodular losses (for more details, check the paper: [The Lovász-Softmax loss](https://arxiv.org/abs/1705.08790)).


## Code Structure
The code structure is based on [MTL-template](https://github.com/yaoyao-liu/meta-transfer-learning) and [Pytorch-Segmentation](https://github.com/yassouali/pytorch_segmentation). 

```
.
|
├── FewShotPreprocessing.py     # utility to organise the Few-shot data into train,test and val set
|
|  
├── dataloader              
|   ├── dataset_loader.py       # data loader for pre datasets
|   ├── mdataset_loader.py      # data loader for meta task dataset
|   └── samplers.py             # samplers for meta task dataset(Few-Shot) 
|
|
├── models                      
|   ├── mtl.py                  # meta-transfer class
|   ├── unet_mtl.py             # unet class
|   └── conv2d_mtl.py           # meta-transfer convolution class
|
├── trainer                     
|   ├── pre.py                  # pre-train trainer class
|   └── meta.py                 # meta-train trainer class
|
|
├── utils                       
|   ├── gpu_tools.py            # GPU tool functions
|   ├── metrics.py              # Metrics functions
|   ├── losses.py               # Loss functions
|   ├── lovasz_losses.py        # Lovasz Loss function
|   └── misc.py                 # miscellaneous tool functions
|
├── main.py                     # the python file with main function and parameter settings
├── run_pre.py                  # the script to run pre-train phase
└── run_meta.py                 # the script to run meta-train and meta-test phases
```
## Running Experiments

Run pretrain phase:
```bash
python run_pre.py
```
Run meta-train and meta-test phase:
```bash
python run_meta.py
```

## Hyperparameters and Options
Hyperparameters and options in `main.py`.

- `model_type` The network architecture
- `dataset` Meta dataset
- `phase` pre-train, meta-train or meta-eval
- `seed` Manual seed for PyTorch, "0" means using random seed
- `gpu` GPU id
- `dataset_dir` Directory for the images
- `max_epoch` Epoch number for meta-train phase
- `num_batch` The number for different tasks used for meta-train
- `shot` Shot number, how many samples for one class in a task
- `teshot` Test-Shot number, how many samples for one class in a meta test task
- `way` Way number, how many classes in a task
- `train_query` The number of training samples for each class in a task 
- `val_query` The number of test samples for each class in a task
- `meta_lr1` Learning rate for SS weights
- `meta_lr2` Learning rate for Base learner weights (meta task)
- `base_lr` Learning rate for the inner loop
- `update_step` The number of updates for the inner loop
- `step_size` The number of epochs to reduce the meta learning rates
- `gamma` Gamma for the meta-train learning rate decay
- `init_weights` The pretained weights for meta-train phase
- `pre_init_weights` The pretained weights for pre-train phase
- `eval_weights` The meta-trained weights for meta-eval phase
- `meta_label` Additional label for meta-train
- `pre_max_epoch` Epoch number for pre-train psase
- `pre_batch_size` Batch size for pre-train phase
- `pre_lr` Learning rate for pre-train pahse
- `pre_gamma` Gamma for the preteain learning rate decay
- `pre_step_size` The number of epochs to reduce the pre-train learning rate
- `pre_custom_weight_decay` Weight decay for the optimizer during pre-train

## Training
### Pre-Train Phase
- Mean IoU
 - ![text alt](https://i.ibb.co/Jr0Hx60/Pre-Train-Io-U.png)

- CE Loss
  - ![text alt](https://i.ibb.co/yBLMtLL/Pre-Train-Loss.png)

### Meta-Train Phase
- Mean IoU
  - ![text alt](https://i.ibb.co/JjjbrMC/Meta-Train-Io-U.png)

- CE Loss
  - ![text alt](https://i.ibb.co/M88HH4v/Meta-Train-Loss.png)

### Meta-Val Phase
- Mean IoU
  - ![text alt](https://i.ibb.co/8KXMZ6j/Meta-Val-Io-U.png)

- CE Loss
  - ![text alt](https://i.ibb.co/FVRZyBC/Meta-Val-Loss.png)

## Acknowledgement
- [Pytorch-Segmentation](https://github.com/yassouali/pytorch_segmentation)
- [Meta-Transfer Learning for Few-Shot Learning](https://github.com/yaoyao-liu/meta-transfer-learning)
