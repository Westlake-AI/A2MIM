_base_ = '../imagenet/vit_small_p16_linear_sz224_8xb128_cos_ep90.py'

# dataset settings
data_source_cfg = dict(type='ImageNet')
# ImageNet dataset
data_train_list = 'data/meta/ImageNet100/train_labeled.txt'
data_train_root = 'data/ImageNet/train'
data_test_list = 'data/meta/ImageNet100/val_labeled.txt'
data_test_root = 'data/ImageNet/val/'

data = dict(
    train=dict(
        data_source=dict(
            list_file=data_train_list, root=data_train_root,
            **data_source_cfg),
    ),
    val=dict(
        data_source=dict(
            list_file=data_test_list, root=data_test_root, **data_source_cfg),
    ))
