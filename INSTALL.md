# Installation

We provide installation instructions for pre-training and fine-tuning experiments here.

## Dependency Setup

Install OpenMixup>=0.2.7 for A2MIM experiments. Here are installation steps with a new conda virtual environment. You can modify the PyTorch version according to your own environment.
```shell
conda create -n a2mim python=3.8 -y
conda activate a2mim
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
mim install mmcv-full
git clone https://github.com/Westlake-AI/openmixup.git
cd openmixup
python setup.py install
cd ..
rm -r openmixup  # you can keep the source code to view implementation details
```

Then, you can setup [MMDetection](https://github.com/open-mmlab/mmdetection/) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/) for downstream tasks.
```shell
pip install mmdet
pip install mmseg
```

## Dataset Preparation

It is recommended to symlink your dataset root (assuming `$DATA_ROOT`) to `$A2MIM/data` by `ln -s $DATA_ROOT ./data`. If your folder structure is different, you may need to change the corresponding paths in config files.

### ImageNet

Prepare the meta files of ImageNet from [OpenMixup](https://github.com/Westlake-AI/openmixup) with following scripts:
```shell
mkdir data/meta
cd data/meta
wget https://github.com/Westlake-AI/openmixup/releases/download/dataset/meta.zip
unzip meta.zip
rm meta.zip
```

Download the [ImageNet-1K](http://image-net.org/) classification dataset ([train](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar) and [val](https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar)) and structure the data as follows.

### Downstream Task Datasets

Download [COCO2017](https://cocodataset.org/#download) and prepare COCO experiments according to the guidelines in [MMDetection](https://github.com/open-mmlab/mmdetection/).

Prepare [ADE20K](https://arxiv.org/abs/1608.05442) according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets) in MMSegmentation. Please use the 2016 version of ADE20K dataset, which can be downloaded from [ADEChallengeData2016](data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip) or [**Baidu Cloud**](https://pan.baidu.com/s/1EIrXVTOxX-cdhYVfqd9Vng?pwd=7ycz) (7ycz).

At last, the folder looks like:

```
root
├── configs
├── data
│   ├── ade
│   ├── coco
│   ├── meta [used for 'ImageList' dataset]
│   ├── ImageNet
│   │   ├── train
│   │   |   ├── n01440764
│   │   |   ├── n01443537
│   │   |   ...
│   │   |   ├── n15075141
│   │   ├── val
│   │   |   ├── ILSVRC2012_val_00000001.JPEG
│   │   |   ...
```
