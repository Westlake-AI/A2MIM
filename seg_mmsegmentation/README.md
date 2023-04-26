# Transferring to Semantic Segmentation with MMSegmentation

For semantic segmentation task on ADE20K, we use MMSegmentation implementations. First, make sure you have installed [MIM](https://github.com/open-mmlab/mim), which is also a project of OpenMMLab.
```shell
pip install openmim
mim install mmsegmentation
```

Besides, please refer to MMSegmentation for [installation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md) and [data preparation](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md#prepare-datasets).

## Train

After installation, you can run MMSeg with simple command.
```shell
bash seg_mmsegmentation/mim_dist_train.sh ${CONFIG} ${PRETRAIN} ${GPUS}
```

Remarks:
- `CONFIG`: Use config files under `configs/benchmarks/mmsegmentation/` or write your own config files
- `PRETRAIN`: the pre-trained model file (the backbone parameters only).
- `${GPUS}`: The number of GPUs that you want to use to train. We adopt 4 GPUs for segmentation tasks by default.

## Test

After training, you can also run the command below to test your model.

```shell
bash seg_mmsegmentation/mim_dist_test.sh ${CONFIG} ${CHECKPOINT} ${GPUS}
```

Remarks:
- `${CHECKPOINT}`: The trained segmentation model that you want to test. 

<p align="right">(<a href="#top">back to top</a>)</p>
