<div align="center">
<h2><a href="https://arxiv.org/abs/2205.13943">Architecture-Agnostic Masked Image Modeling - From ViT back to CNN</a></h2>

[Siyuan Li](https://lupin1998.github.io/)<sup>\*,1,2</sup>, [Di Wu](https://scholar.google.com/citations?user=egz8bGQAAAAJ&hl=zh-CN)<sup>\*,1,2</sup>, [Fang Wu](https://smiles724.github.io/)<sup>1,3</sup>, [Zelin Zang](https://scholar.google.com/citations?user=foERjnQAAAAJ&hl=en)<sup>1,2</sup>, [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl=zh-CN)<sup>†,1</sup>

<sup>1</sup>[Westlake University](https://westlake.edu.cn/), <sup>2</sup>[Zhejiang University](https://www.zju.edu.cn/english/), <sup>3</sup>[Tsinghua University](https://air.tsinghua.edu.cn/en/)
</div>

<p align="center">
<a href="https://arxiv.org/abs/2205.13943" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2205.13943-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/Westlake-AI/A2MIM/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23B7A800" /></a>
</p>

<p align="center">
<img src="https://user-images.githubusercontent.com/44519745/234438993-b5a145ab-d345-46ae-9267-25f68379bb62.png" width=100% height=100% 
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

- [x] Update camera-ready version of A2MIM [[arXiv](https://arxiv.org/abs/2205.13943)] [[poster](https://github.com/Lupin1998/Lupin1998.github.io/blob/master/Files/ICML_2023_A2MIM_poster.png)]
- [x] **ImageNet** pre-training and fine-tuning with [OpenMixup](https://github.com/Westlake-AI/openmixup) [[config_pretrain](configs/openmixup/pretrain)] [[config_finetune](configs/openmixup/finetune)]
- [ ] **ImageNet** pre-training and fine-tuning with [MMPretrain](https://github.com/open-mmlab/mmpretrain)
- [x] Downstream Transfer to Object Detection on **COCO** with [MMDetection](https://github.com/open-mmlab/mmdetection) [[config](det_mmdetection)]
- [x] Downstream Transfer to Semantic Segmentation on **ADE20K** [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) [[config](seg_mmsegmentation)]
- [x] Analysis tools and results [[rep_bottleneck](analysis_tools/representation_bottleneck)]
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

### 3. Implementation Details

* A2MIM model: In [a2mim.py](https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/selfsup/a2mim.py), the A2MIM method takes input samples, applies masking, encodes and caculates the MIM losses.
* A2MIM head: In [mim_head.py](https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/heads/mim_head.py), two MIM losses are computed, where [regression_loss.py](https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/losses/regression_loss.py) caculates the L1 loss and [focal_loss.py](https://github.com/Westlake-AI/openmixup/blob/main/openmixup/models/losses/focal_loss.py) caculates for the Fourier domain loss.
* Dataloader: In [masked_image.py](https://github.com/Westlake-AI/openmixup/blob/main/openmixup/datasets/masked_image.py), loading the processed RGB images with the masked RGB mean.

## Results and Models

We provide the summarization of pre-training (800 or 300 epochs) and fine-tuning (100 or 300 epochs) results of A2MIM and baselines on ImageNet-1K.

| Methods | # Params. | Supervision | SimMIM | A2MIM |
|---|:---:|:---:|:---:|:---:|
| Target | (M) | Label | RGB | RGB |
| ViT-S | 48.8 | 79.9 | 81.7 | 82.1 |
| ViT-B | 86.7 | 81.8 | 83.8 | 84.2 |
| ViT-L | 304.6 | 82.6 | 85.6 | 86.1 |
| ResNet-50 | 25.6 | 79.8 | 79.9 | 80.4 |
| ResNet-101 | 44.5 | 81.3 | 81.3 | 81.9 |
| ResNet-152 | 60.2 | 81.8 | 81.9 | 82.5 |
| ResNet-200 | 64.7 | 82.1 | 82.2 | 83.0 |
| ConvNeXt-S | 50.2 | 83.1 | 83.2 | 83.7 |
| ConvNeXt-B | 88.6 | 83.5 | 83.6 | 84.1 |

Config files, models, logs, and visualization of reconstructions are provided as follows. These files can also be downloaded from [a2mim-in1k-weights](https://github.com/Westlake-AI/A2MIM/releases/tag/a2mim-in1k-weights), [OpenMixup-a2mim-in1k-weights](https://github.com/Westlake-AI/openmixup/releases/tag/a2mim-in1k-weights) or **Baidu Cloud**: [A2MIM (3q5i)](https://pan.baidu.com/s/1aj3Lbj_wvyV_1BRzFhPcwQ?pwd=3q5i).

<details open>
  <summary>ViT-S/B/L on ImageNet-1K.</summary>

  | Method | Backbone | PT Epoch | FT Top-1 | Pre-training | Fine-tuning | Results |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | SimMIM | ViT-Small | 800 | 81.7 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/vit_small_sz224_8xb256_step_fp16_ep800.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_small_sz224_8xb256_step_fp16_ep800_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_small_sz224_8xb256_step_fp16_ep800_vis.zip) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_small_p16_swin_ft_simmim_sz224_8xb128_cos_ep100.py) | [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_small_sz224_8xb256_step_fp16_ep800_ft.pth) \| [log](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_small_sz224_8xb256_step_fp16_ep800_ft.log.json) |
  | A2MIM | ViT-Small | 800 | 82.1 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/vit_small_l0_sz224_8xb256_step_ep800.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_small_sz224_8xb256_step_fp16_ep800_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_small_sz224_8xb256_step_fp16_ep800_vis.zip) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_small_p16_swin_ft_simmim_sz224_8xb128_cos_ep200.py) | [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_small_sz224_8xb256_step_fp16_ep800_ft.pth) \| [log](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_small_sz224_8xb256_step_fp16_ep800_ft.log.json) |
  | SimMIM | ViT-Base | 800 | 83.8 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/vit_base_sz224_8xb128_accu2_step_fp16_ep800.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_base_sz224_8xb128_accu2_step_fp16_ep800_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_base_sz224_8xb128_accu2_step_fp16_ep800_vis.zip) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_simmim_sz224_4xb128_accu2_cos_ep100.py) | [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_vit_base_l0_res_fft01_sz224_4xb128_accu4_step_fp16_ep800_ft.pth) \| [log](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_vit_base_l0_res_fft01_sz224_4xb128_accu4_step_fp16_ep800_ft.log.json) |
  | A2MIM | ViT-Base | 800 | 84.3 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/vit_base_l0_sz224_8xb128_accu2_step_ep800.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/full_a2mim_vit_base_l0_res_fft01_sz224_4xb128_accu4_step_fp16_ep800.pth) \| [vis](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/visualization_a2mim_vit_base_l0_res_fft01_sz224_4xb128_accu4_step_fp16_ep800.zip) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_base_p16_swin_ft_simmim_sz224_4xb128_accu2_cos_ep100.py) | [ckpt](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/a2mim_vit_base_l0_res_fft01_sz224_4xb128_accu4_step_fp16_ep800_ft.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/a2mim_vit_base_l0_res_fft01_sz224_4xb128_accu4_step_fp16_ep800_ft.log.json) |
  | SimMIM | ViT-Large | 800 | 85.6 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/vit_large_sz224_8xb128_accu2_step_fp16_ep800.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_large_sz224_8xb128_accu2_step_fp16_ep800_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_large_sz224_8xb128_accu2_step_fp16_ep800_vis.zip) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_large_p16_swin_ft_simmim_sz224_8xb64_accu2_cos_ep100.py) | [log](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_vit_large_sz224_8xb128_accu2_step_fp16_ep800_ft.log.json) |
  | A2MIM | ViT-Large | 800 | 86.1 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/vit_large_l0_sz224_8xb128_accu2_step_ep800.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_vit_large_l0_sz224_8xb128_accu2_step_fp16_ep800_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_vit_large_l0_sz224_8xb128_accu2_step_fp16_ep800_vis.zip) | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/vit_large_p16_swin_ft_simmim_sz224_8xb64_accu2_cos_ep150.py) | [log](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_vit_large_l0_sz224_8xb128_accu2_step_fp16_ep800_ft.log.json) |
</details>
<details>
  <summary>ResNet-50/101/152/200 on ImageNet-1K.</summary>

  | Method | Backbone | PT Epoch | FT (A2) Top-1 | Pre-training | Fine-tuning | Results |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | SimMIM | ResNet-50 | 300 | 79.9 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/r50_sz224_8xb256_fp16_ep300.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r50_sz224_8xb256_cos_fp16_ep300_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r50_sz224_8xb256_cos_fp16_ep300_vis.zip) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_rsb_a2_ft_sz224_8xb256_cos_fp16_ep300.py) | - |
  | A2MIM | ResNet-50 | 100 | 78.8 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/r50_l3_sz224_init_8xb256_cos_ep100.py) \| [ckpt](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/full_a2mim_r50_l3_sz224_init_8xb256_cos_ep100.pth) \| [vis](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/visualization_a2mim_r50_l3_sz224_init_8xb256_cos_ep100.zip) | [RSB A3](https://github.com/Westlake-AI/openmixup/blob/main/configs/benchmarks/classification/imagenet/r50_rsb_a3_ft_sz160_4xb512_cos_fp16_ep100.py) | [ckpt](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/a2mim_r50_l3_sz224_init_8xb256_cos_ep100_ft_rsb_a3.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/a2mim_r50_l3_sz224_init_8xb256_cos_ep100_ft_rsb_a3.log.json) |
  | A2MIM | ResNet-50 | 300 | 80.4 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/r50_l3_sz224_init_8xb256_cos_ep300.py) \| [ckpt](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/full_a2mim_r50_l3_sz224_init_8xb256_cos_ep300.pth) \| [vis](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/visualization_a2mim_r50_l3_sz224_init_8xb256_cos_ep300.zip) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_rsb_a2_ft_sz224_8xb256_cos_fp16_ep300.py) | [ckpt](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/a2mim_r50_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.pth) \| [log](https://github.com/Westlake-AI/openmixup/releases/download/a2mim-in1k-weights/a2mim_r50_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.log.json) |
  | SimMIM | ResNet-101 | 300 | 81.3 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/r101_sz224_8xb256_fp16_ep300.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r101_sz224_8xb256_cos_fp16_ep300_full.pth) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_rsb_a2_ft_sz224_8xb256_cos_fp16_ep300.py) | [ckpt (A3)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r101_sz224_8xb256_ep300_ft_rsb_a3.pth) \| [log (A3)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r101_sz224_8xb256_ep300_ft_rsb_a3.log.json) |
  | A2MIM | ResNet-101 | 300 | 81.9 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/r101_l3_sz224_init_8xb256_cos_ep300.py) \| [ckpt (300ep)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r101_l3_sz224_init_8xb256_cos_ep300_full.pth) \| [ckpt (800ep)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r101_l3_sz224_init_8xb256_cos_ep800_full.pth) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_rsb_a2_ft_sz224_8xb256_cos_fp16_ep300.py) | [ckpt (A2)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r101_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.pth) \| [log (A2)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r101_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.log.json) |
  | SimMIM | ResNet-152 | 300 | 81.9 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/r152_sz224_8xb256_fp16_ep300.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r152_sz224_8xb256_cos_fp16_ep300_full.pth) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_rsb_a2_ft_sz224_8xb256_cos_fp16_ep300.py) | [log (A3)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r152_sz224_8xb256_ep300_ft_rsb_a3.log.json) |
  | A2MIM | ResNet-152 | 300 | 82.5 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/r152_l3_sz224_init_8xb256_cos_ep300.py) \| [ckpt (300ep)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r152_l3_sz224_init_8xb256_cos_ep300_full.pth) \| [ckpt (800ep)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r152_l3_sz224_init_8xb256_cos_ep800_full.pth) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_rsb_a2_ft_sz224_8xb256_cos_fp16_ep300.py) | [ckpt (A2)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r152_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.pth) \| [log (A2)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r152_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.log.json) |
  | SimMIM | ResNet-200 | 300 | 82.2 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/r200_sz224_8xb256_fp16_ep300.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r200_sz224_8xb256_cos_fp16_ep300_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r200_sz224_8xb256_cos_fp16_ep300_vis.zip) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_rsb_a2_ft_sz224_8xb256_cos_fp16_ep300.py) | [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r200_sz224_8xb256_cos_fp16_ep300_ft_rsb_a2.pth) \| [log](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_r200_sz224_8xb256_cos_fp16_ep300_ft_rsb_a2.log.json) |
  | A2MIM | ResNet-200 | 300 | 83.0 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/r200_l3_sz224_init_8xb256_cos_ep300.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r200_l3_sz224_init_8xb256_cos_ep300_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r200_l3_sz224_init_8xb256_cos_ep300_vis.zip) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/r50_rsb_a2_ft_sz224_8xb256_cos_fp16_ep300.py) | [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r200_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.pth) \| [log](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_r200_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.log.json) |
</details>
<details>
  <summary>ConvNeXt-S/B on ImageNet-1K.</summary>

  | Method | Backbone | PT Epoch | FT (A2) Top-1 | Pre-training | Fine-tuning | Results |
  |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
  | SimMIM | ConvNeXt-S | 300 | 83.2 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/convnext_small_sz224_8xb256_fp16_ep300.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_convnext_small_sz224_8xb256_cos_fp16_ep300_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_convnext_small_sz224_8xb256_cos_fp16_ep300_vis.zip) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/convnext_s_spark_ft_sz224_8xb256_cos_fp16_ep300.py) | - |
  | A2MIM | ConvNeXt-S | 300 | 83.7 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/convnext_s_l3_sz224_init_8xb256_cos_ep300.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_convnext_small_l3_sz224_init_8xb256_cos_ep300_full.pth) \| [vis](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_convnext_small_l3_sz224_init_8xb256_cos_ep300_vis.zip) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/convnext_s_spark_ft_sz224_8xb256_cos_fp16_ep300.py) | [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_convnext_small_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.pth) \| [log](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_convnext_small_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.log.json) |
  | SimMIM | ConvNeXt-B | 300 | 83.6 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/simmim/imagenet/convnext_base_sz224_8xb256_fp16_ep300.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_convnext_base_sz224_8xb256_cos_fp16_ep300_full.pth) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/convnext_b_spark_ft_sz224_8xb256_cos_fp16_ep300.py) | [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_convnext_base_sz224_8xb256_cos_fp16_ep300_ft.pth) \| [log](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/simmim_convnext_base_sz224_8xb256_cos_fp16_ep300_ft.log.json) |
  | A2MIM | ConvNeXt-B | 300 | 84.1 | [config](https://github.com/Westlake-AI/openmixup/tree/main/configs/selfsup/a2mim/imagenet/convnext_b_l3_sz224_init_8xb256_cos_ep300.py) \| [ckpt](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_convnext_base_l3_sz224_init_8xb256_cos_fp16_ep300_full.pth) | [RSB A2](https://github.com/Westlake-AI/openmixup/tree/main/configs/benchmarks/classification/imagenet/convnext_b_spark_ft_sz224_8xb256_cos_fp16_ep300.py) | [ckpt (A2)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_convnext_base_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.pth) \| [ckpt (A3)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_convnext_base_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a3.pth) \| [log (A2)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_convnext_base_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.log.json) \| [log (A3)](https://github.com/Westlake-AI/A2MIM/releases/download/a2mim-in1k-weights/a2mim_convnext_base_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a3.log.json) |
</details>

### 4. Empirical Studies

We provided interpretation of how masked image modeling works with [representation bottleneck]() based on ViTs and CNNs. As shown in Figure 1/5/A1/A2 in [A2MIM](https://arxiv.org/abs/2205.13943) and following figures, we visualize the multi-order interation strengths with [representation_bottleneck](https://github.com/Westlake-AI/A2MIM/tree/main/analysis_tools/representation_bottleneck). We also provided analysis from frequency perspectives in Figure A3/A4 in [A2MIM](https://arxiv.org/abs/2205.13943) based on [fourier_analysis](https://github.com/Westlake-AI/A2MIM/tree/main/analysis_tools/fourier_analysis).

<p align="center">
<img src="https://github.com/Westlake-AI/A2MIM/assets/44519745/1b5470b3-51f9-4585-9ff2-eeec34cef766" width=100% height=100% 
class="center">
</p>

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
@inproceedings{icml2023a2mim,
  title={Architecture-Agnostic Masked Image Modeling -- From ViT back to CNN},
  author={Li, Siyuan and Wu, Di and Wu, Fang and Zang, Zelin and Li, Stan. Z.},
  booktitle={International Conference on Machine Learning},
  year={2023},
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
