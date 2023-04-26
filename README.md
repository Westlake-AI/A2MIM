<div align="center">
<h2><a href="https://arxiv.org/abs/2205.13943">Architecture-Agnostic Masked Image Modeling - From ViT back to CNN</a></h2>

[Siyuan Li](https://lupin1998.github.io/)<sup>\*,1,2</sup>, [Di Wu](https://scholar.google.com/citations?user=egz8bGQAAAAJ&hl=zh-CN)<sup>\*,1,2</sup>, [Zelin Zang](https://scholar.google.com/citations?user=foERjnQAAAAJ&hl=en)<sup>1,2</sup>, [Fang Wu](https://smiles724.github.io/)<sup>1,2</sup>, [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl=zh-CN)<sup>†,1</sup>

<sup>1</sup>[Westlake University](https://westlake.edu.cn/), <sup>2</sup>[Zhejiang University](https://www.zju.edu.cn/english/), <sup>2</sup>[Tsinghua University](https://air.tsinghua.edu.cn/en/)
</div>

<p align="center">
<a href="https://arxiv.org/abs/2205.13943" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2205.13943-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/Westlake-AI/A2MIM/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" /></a>
<!-- <a href="https://colab.research.google.com/github/Westlake-AI/MogaNet/blob/main/demo.ipynb" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a> -->
<!-- <a href="https://huggingface.co/MogaNet" alt="Huggingface">
    <img src="https://img.shields.io/badge/huggingface-MogaNet-blueviolet" /></a> -->
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/44519745/174272666-30bc3177-e61f-4331-9f32-91f47aad6578.png" width=100% height=100% 
class="center">
</p>

Masked image modeling (MIM), an emerging self-supervised pre-training method, has shown impressive success across numerous downstream vision tasks with Vision transformers (ViT). Its underlying idea is simple: a portion of the input image is randomly masked out and then reconstructed via the pre-text task. However, why MIM works well is not well explained, and previous studies insist that MIM primarily works for the Transformer family but is incompatible with CNNs. In this paper, we first study interactions among patches to understand what knowledge is learned and how it is acquired via the MIM task. We observe that MIM essentially teaches the model to learn better middle-level interactions among patches and extract more generalized features. Based on this fact, we propose an Architecture-Agnostic Masked Image Modeling framework (A2MIM), which is compatible with not only Transformers but also CNNs in a unified way. Extensive experiments on popular benchmarks show that our A2MIM learns better representations and endows the backbone model with the stronger capability to transfer to various downstream tasks for both Transformers and CNNs.

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#catalog">Catalog</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgement">Acknowledgement</a></li>
    <li><a href="#citation">Citation</a></li>
  </ol>
</details>

## Catalog

We have released implementations of A2MIM based on [OpenMixup](https://github.com/Westlake-AI/openmixup). In the future, we plan to add A2MIM implementations to [MMPretrain](https://github.com/open-mmlab/mmpretrain). Pre-trained and fine-tuned models are released in [GitHub](https://github.com/Westlake-AI/openmixup/releases/tag/a2mim-in1k-weights) / [Baidu Cloud](https://pan.baidu.com/s/1aj3Lbj_wvyV_1BRzFhPcwQ?pwd=3q5i).

- [ ] Update camera-ready version of A2MIM or arXiv.
- [x] **ImageNet** pre-training and fine-tuning with [OpenMixup](https://github.com/Westlake-AI/openmixup) [[config_pretrain](configs/openmixup/pretrain)] [[config_finetune](configs/openmixup/finetune)]
- [ ] **ImageNet** pre-training and fine-tuning with [MMPretrain](https://github.com/open-mmlab/mmpretrain)
- [x] Downstream Transfer to Object Detection on **COCO** with [MMDetection](https://github.com/open-mmlab/mmdetection) [[config](det_mmdetection/configs)]
- [x] Downstream Transfer to Semantic Segmentation on **ADE20K** [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) [[config](seg_mmsegmentation/configs)]
- [ ] Analysis tools and results
- [ ] Visualization of pre-training on Google Colab and Notebook Demo

## Pre-training on ImageNet

### 1. Installation

Please refer to [INSTALL.md](INSTALL.md) for installation instructions.

### 2. Pre-training and fine-tuning

We provide scripts for multiple GPUs pre-training and the specified `CONFIG_FILE`. 
```shell
bash tools/dist_train.sh ${CONFIG_FILE} ${GPUS} [optional arguments]
```

For example, you can run the script below to pre-train ResNet-50 with A2MIM on ImageNet with 8 GPUs:
```shell
PORT=29500 bash tools/dist_train.sh configs/openmixup/pretrain/a2mim/imagenet/r50_l3_sz224_init_8xb256_cos_ep300.py 8
```
After pre-trianing, you can fine-tune and evaluate the models with the corresponding script:
```shell
python tools/model_converters/extract_backbone_weights.py work_dirs/openmixup/pretrain/a2mim/imagenet/r50_l3_sz224_init_8xb256_cos_ep300/latest.pth ${PATH_TO_CHECKPOINT}
PORT=29500 bash tools/dist_train_ft_8gpu.sh configs/openmixup/finetune/imagenet/r50_rsb_a3_ft_sz160_4xb512_cos_fp16_ep100.py ${PATH_TO_CHECKPOINT}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

- [OpenMixup](https://github.com/Westlake-AI/openmixup): Open-source toolbox for supervised and self-supervised visual representation learning.
- [pytorch-image-models](https://github.com/rwightman/pytorch-image-models): PyTorch image models, scripts, pretrained weights.
- [SimMIM](https://github.com/microsoft/simmim): Official PyTorch implementation of SimMIM.
- [MMPretrain](https://github.com/open-mmlab/mmpretrain): OpenMMLab Pre-training Toolbox and Benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab Detection Toolbox and Benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab Semantic Segmentation Toolbox and Benchmark.

## Citation

If you find this repository helpful, please consider citing our paper:
```
@inproceedings{zbontar2021barlow,
  title={Architecture-Agnostic Masked Image Modeling -- From ViT back to CNN},
  author={Li, Siyuan and Wu, Di and Wu, Fang and Zang, Zelin and Li, Stan. Z.},
  booktitle={International Conference on Machine Learning},
  year={2023},
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
