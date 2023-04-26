# Transferring to Object Detection with MMDetection

We follow the evaluation setting in MoCo when transferring to object detection on COCO using MMDetection.

## Installation

First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.
```shell
pip install openmim
mim install mmdet
```

Besides, please refer to MMDet for [installation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) and [data preparation](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md).

## Transferring Learning

After installing MMDet, you can run MMDetection with simple command. We provide scripts for the stage-4 only (`C4`) and `FPN` setting of object detection models.

```shell
bash det_mmdetection/mim_dist_train_c4.sh ${CONFIG} ${PRETRAIN} ${GPUS}
bash det_mmdetection/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}
```

Remarks:
- `CONFIG`: Use config files under `det_mmdetection/configs` or write your own config files
- `PRETRAIN`: the pre-trained model file (the backbone parameters only).
- `${GPUS}`: The number of GPUs that you want to use to train. We adopt 8 GPUs for detection tasks by default.
- Since repositories of OpenMMLab have support referring config files across different repositories, we can easily leverage the configs from MMDetection like:
```shell
_base_ = 'mmdet::mask_rcnn/mask-rcnn_r50-caffe-c4_1x_coco.py'
```

Example:

```shell
bash det_mmdetection/mim_dist_train_c4.sh \
det_mmdetection/configs/coco/mask-rcnn_r50-c4_ms-2x_coco.py ${PATH_TO_CHECKPOINT} 8
```

<p align="right">(<a href="#top">back to top</a>)</p>
