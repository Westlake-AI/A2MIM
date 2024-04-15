# PyTorch Implementation of Representation Bottleneck

## Overview

This repository evaluates multi-order interaction strengths of [timm](https://github.com/rwightman/pytorch-image-models) backbones for visual representation learning, which is based on **Discovering and Explaining the Representation Bottleneck of DNNs** ([ICLR'2022](https://arxiv.org/abs/2111.06236)) and the [official implementation](https://github.com/nebularaid2000/bottleneck). Please refer to the [official implementation](https://github.com/nebularaid2000/bottleneck) for details.

## Installation

This repository works with **PyTorch 1.8** or higher and timm. There are installation steps with the latest PyTorch:
```shell
conda create -n bottleneck python=3.8 pytorch=1.12 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate bottleneck
pip install -r requirements.txt
```

Then, please download datasets and place them under './datasets'. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) will be automatically downloaded, while [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/) should be downloaded and unziped manually.

## Usage

We only support the evaluation of pre-trained models. Please download released pre-trained models from [timm](https://github.com/rwightman/pytorch-image-models) and place them in './timm_hub'. We provide examples on ImageNet in 'interaction_in1k.sh':

```
bash interaction.sh
```
You can uncomment the setting (the model name and ckeckpoints) you want to run on top of the script. The results will be saved in the `results` directory by default.

## Citation

Please cite the relevant papers if you find this repository useful in your research.

```
@article{Deng2022Discovering,
  title={Discovering and Explaining the Representation Bottleneck of DNNs},
  author={Huiqi Deng and Qihan Ren and Xu Chen and Hao Zhang and Jie Ren and Quanshi Zhang},
  journal={ArXiv},
  year={2022},
  volume={abs/2111.06236}
}

@inproceedings{icml2023a2mim,
  title={Architecture-Agnostic Masked Image Modeling -- From ViT back to CNN},
  author={Li, Siyuan and Wu, Di and Wu, Fang and Zang, Zelin and Li, Stan. Z.},
  booktitle={International Conference on Machine Learning},
  year={2023},
}
```
