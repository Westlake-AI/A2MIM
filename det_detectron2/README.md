# Transferring to Object Detection with Detectron2

We also provide transferring learning experiments on COCO with [Detectrons](https://github.com/facebookresearch/detectron2) implementations following MoCo.

## Installation

Please refer to [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md) for installation and follow the [directory structure](https://github.com/facebookresearch/detectron2/tree/main/datasets) to prepare your datasets following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets) required by detectron2.

```shell
conda activate detectron2 # use detectron2 environment here, otherwise use open-mmlab environment
cd det_detectron2
python convert-pretrain-to-detectron2.py ${WEIGHT_FILE} ${OUTPUT_FILE} # must use .pkl as the output extension.
bash run.sh ${DET_CFG} ${OUTPUT_FILE}
```

<p align="right">(<a href="#top">back to top</a>)</p>

## Transferring Learning

After training, you can also run the command below with 8 GPUs to test your model.

```shell
bash benchmarks/mmdetection/mim_dist_test.sh ${CONFIG} ${CHECKPOINT}
```

Remarks:
- `${CHECKPOINT}`: The well-trained detection model that you want to test.

<p align="right">(<a href="#top">back to top</a>)</p>
