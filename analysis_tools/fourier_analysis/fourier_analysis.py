import os
import copy
import math
import mmcv
import numpy as np
from einops import rearrange

import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from openmixup import datasets as openmixup_datasets

import sys
sys.path.append('./')
from utils import get_model, parse_args


class PatchEmbed(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)
        
    def forward(self, x, **kwargs):
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        return x


class Residual(nn.Module):
    def __init__(self, *fn):
        super().__init__()
        self.fn = nn.Sequential(*fn)
        
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class Lambda(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x)


def flatten(xs_list):
    return [x for xs in xs_list for x in xs]


def fourier(x):  # 2D Fourier transform
    f = torch.fft.fft2(x)
    f = f.abs() + 1e-6
    f = f.log()
    return f


def fft_shift(x):  # shift Fourier transformed feature map
    b, c, h, w = x.shape
    return torch.roll(x, shifts=(int(h/2), int(w/2)), dims=(2, 3))


def make_segments(x, y):  # make segment for `plot_segment`
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments


def plot_segment(ax, xs, ys, marker, liner='solid', cmap_name="plasma", alpha=1.0, zorder=1):
    # plot with cmap segments
    z = np.linspace(0.0, 1.0, len(ys))
    z = np.asarray(z)
    
    cmap = cm.get_cmap(cmap_name)
    norm = plt.Normalize(0.0, 1.0)
    segments = make_segments(xs, ys)
    lc = LineCollection(segments, array=z, cmap=cmap_name, norm=norm,
                        linewidth=2.0, linestyles=liner, alpha=alpha)
    ax.add_collection(lc)

    colors = [cmap(x) for x in xs]
    ax.scatter(xs, ys, color=colors, marker=marker, zorder=100 + zorder)
    return lc


def create_cmap(color_name, end=0.95):
    """ create custom cmap """
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap

    color = cm.get_cmap(color_name, 200)
    if end == 0.8:
        newcolors = color(np.linspace(0.75, end, 200))
    else:
        newcolors = color(np.linspace(max(0.5, end-0.4), end, 200))
    newcmp = ListedColormap(newcolors, name=color_name+"05_09")
    return newcmp


def make_proxy(color, marker, liner, **kwargs):
    """ add custom legend """
    from matplotlib.lines import Line2D
    cmap = cm.get_cmap(color)
    color = cmap(np.arange(4) / 4)
    return Line2D([0, 1], [0, 1], color=color[3], marker=marker, linestyle=liner)


def plot_fourier_features(latents):
    # Fourier transform feature maps
    fourier_latents = []
    for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
        latent = latent.cpu()
        
        if len(latent.shape) == 3:  # for ViT
            b, n, c = latent.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        elif len(latent.shape) == 4:  # for CNN
            b, c, h, w = latent.shape
        else:
            raise Exception("shape: %s" % str(latent.shape))
        latent = fourier(latent)
        latent = fft_shift(latent).mean(dim=(0, 1))
        latent = latent.diag()[int(h/2):]  # only use the half-diagonal components
        latent = latent - latent[0]  # visualize 'relative' log amplitudes 
                                    # (i.e., low-freq amp - high freq amp)
        fourier_latents.append(latent)
    
    return fourier_latents


def plot_variance_features(latents):
    # aggregate feature map variances
    variances = []
    for latent in latents:  # `latents` is a list of hidden feature maps in latent spaces
        latent = latent.cpu()
        
        if len(latent.shape) == 3:  # for ViT
            b, n, c = latent.shape
            h, w = int(math.sqrt(n)), int(math.sqrt(n))
            latent = rearrange(latent, "b (h w) c -> b c h w", h=h, w=w)
        elif len(latent.shape) == 4:  # for CNN
            b, c, h, w = latent.shape
        else:
            raise Exception("shape: %s" % str(latent.shape))
        variances.append(latent.var(dim=[-1, -2]).mean(dim=[0, 1]))
    
    return variances


def forward_model(args, device):

    #  ======================== build model ========================
    model, mean, std = get_model(args=args)
    model = model.to(device)
    model.eval()
    
    if "resnet" in args.model_name:
        # model → blocks. `blocks` is a sequence of blocks
        blocks = [
            nn.Sequential(model.conv1, model.bn1, model.act1, model.maxpool),
            *model.layer1,
            *model.layer2,
            *model.layer3,
            *model.layer4,
            nn.Sequential(model.global_pool, model.fc)
        ]
    elif "vit" in args.model_name or "deit" in args.model_name:
        # `blocks` is a sequence of blocks
        blocks = [
            PatchEmbed(model),
            *flatten([[Residual(b.norm1, b.attn), Residual(b.norm2, b.mlp)] 
                    for b in model.blocks]),
            nn.Sequential(Lambda(lambda x: x[:, 0]), model.norm, model.head),
        ]
    else:
        raise NotImplementedError
    # print('blocks:', len(blocks))

    #  ======================== build dataloader ========================
    test_dir = args.test_dir
    assert os.path.isdir(test_dir) and args.test_list is not None
    test_set = openmixup_datasets.build_dataset(
        dict(
            type='ClassificationDataset',
            data_source=dict(
                list_file=args.test_list, root=test_dir, **dict(type='ImageNet')),
            pipeline=[
                dict(type='Resize', size=256),
                dict(type='CenterCrop', size=224),
                dict(type='ToTensor'),
                dict(type='Normalize', mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ],
            prefetch=False)
        )
    test_loader = openmixup_datasets.build_dataloader(test_set, imgs_per_gpu=args.batch_size,
                                                      workers_per_gpu=2, dist=False, shuffle=False,
                                                      drop_last=True, prefetch=False, img_norm_cfg=dict())

    #  ======================== build dataloader ========================
    # load a sample ImageNet-1K image -- use the full val dataset for precise results
    latents = dict()
    for i, data in tqdm(enumerate(test_loader)):
        if isinstance(data, tuple):
            assert len(data) == 2
            img, label = data
        else:
            assert isinstance(data, dict)
            img = data['img']
            label = data['gt_label']

        with torch.no_grad():
            xs, label = img.to(device), label.to(device)
            # accumulate `latents` by collecting hidden states of a model
            for b,block in enumerate(blocks):
                if b == len(blocks) - 1:  # drop logit (output)
                    break
                xs = block(xs)
                if i == 0:
                    latents[str(b)] = list()
                    latents[str(b)].append(xs.detach().cpu())
                else:
                    latents[str(b)].append(xs.detach().cpu())
        if i == 25:
            break
    
    latent_list = list()
    for i in range(len(blocks)-1):
        l = torch.cat(latents[str(i)], dim=0)
        latent_list.append(l)
    latents = latent_list

    # for ViT/DeiT/pit_ti_224: Drop CLS token
    if "vit" in args.model_name or "deit" in args.model_name or "pit" in args.model_name:
        latents = [latent[:,1:] for latent in latents]
    
    return latents


def set_plot_args(model_name, idx=0, alpha_base=0.9):
    # setup
    linear_mapping = dict(cl="dashed", mim="solid", ssl="dashed", sl="dashdot")
    marker_mapping = dict(cl="s", mim="p", ssl="D", sl="o")
    colour_mapping = dict(cl=["YlGnBu", "Blues", "GnBu", "Greens", "YlGn", "winter"],
                            # mim=["Reds", "OrRd", "YlOrRd", "RdPu",],  # ResNet
                            mim=["Reds", "YlOrRd", "OrRd", "RdPu",],  # ViT
                            ssl=["PuRd",],  # red
                            sl=["autumn", "winter", ],
                        )
    zorder_mapping = dict(cl=3, mim=4, ssl=2, sl=1)

    prefix = model_name.split("_")[0]
    alpha = alpha_base if prefix != 'sl' else 0.7
    marker = marker_mapping[prefix]
    liner = linear_mapping[prefix]
    cmap_list = colour_mapping[prefix]
    cmap_name = create_cmap(cmap_list[idx % len(cmap_list)], end=0.8 if prefix == 'sl' else 0.95)
    zorder = zorder_mapping[prefix]
    # refine model_name
    model_name = model_name.split("_")[-1].replace("+", " \ ")
    model_name = r"$\mathrm{" + model_name + "}$"
    
    return model_name, alpha, marker, liner, cmap_name, zorder


def plot_fft_A(args, fourier_latents, save_path, save_format='png'):
    # A. Plot Fig 2a: "Relative log amplitudes of Fourier transformed feature maps"
    fig, ax1 = plt.subplots(1, 1, figsize=(3.3, 4), dpi=150)
    
    for i, latent in enumerate(reversed(fourier_latents[:-1])):
        freq = np.linspace(0, 1, len(latent))
        ax1.plot(freq, latent, color=cm.plasma_r(i / len(fourier_latents)))
    
    ax1.set_xlim(left=0, right=1)
    ax1.set_xlabel("Frequency")
    ax1.set_ylabel("$\Delta$ Log amplitude")

    from matplotlib.ticker import FormatStrFormatter
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.1fπ'))

    plt.show()
    plt.savefig(os.path.join(save_path, f'fft_features.{save_format}'))
    plt.close()


def plot_fft_B(args, fourier_latents, save_path, model_names=None, save_format='png'):
    # B. Plot Fig 8: "Relative log amplitudes of high-frequency feature maps"

    # plot settings
    alpha_base = 0.9
    font_size = 13
    cmap_name = "plasma"
    liner = "solid"

    if model_names is None:
        dpi = 120
        model_names = ['ssl_' + args.model_name]
        fourier_latents = [fourier_latents]
    else:
        dpi = 400
        assert isinstance(model_names, list) and len(model_names) >= 1
        zipped = zip(model_names, fourier_latents)
        zipped = sorted(zipped, key=lambda x:x[0])
        zipped = zip(*zipped)
        model_names, fourier_latents = [list(x) for x in zipped]
    
    fig, ax2 = plt.subplots(1, 1, figsize=(6.5, 5), dpi=dpi)
    proxy_list = []
    for i in range(len(model_names)):
        print(i, model_names[i], len(fourier_latents[i]))
        if "resnet" in args.model_name:
            pools = [4, 8, 14]
            msas = []
            marker = "D"
        elif "vit" in args.model_name or "deit" in args.model_name:
            pools = []
            msas = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,]  # vit-tiny
            marker = "o"
        else:
            import warnings
            warnings.warn("The configuration for %s are not implemented." % args.model_name, Warning)
            pools, msas = [], []
            marker = "s"
        
        # setup
        model_names[i], alpha, marker, liner, cmap_name, zorder = set_plot_args(model_names[i], i, alpha_base)
        # add legend
        proxy_list.append(make_proxy(cmap_name, marker, liner, linewidth=2))

        # Normalize
        depths = range(len(fourier_latents[i]))
        depth = len(depths) - 1
        depths = (np.array(depths)) / depth
        pools = (np.array(pools)) / depth
        msas = (np.array(msas)) / depth

        lc = plot_segment(ax2, depths, [latent[-1] for latent in fourier_latents[i]],
                     marker=marker, liner=liner, alpha=alpha, cmap_name=cmap_name, zorder=zorder)

    # ploting
    for pool in pools:
        ax2.axvspan(pool - 1.0 / depth, pool + 0.0 / depth, color="tab:blue", alpha=0.15, lw=0)
    for msa in msas:
        ax2.axvspan(msa - 1.0 / depth, msa + 0.0 / depth, color="tab:gray", alpha=0.15, lw=0)

    ax2.set_xlabel(r"$\mathrm{Normalized \ Depth}$", fontsize=font_size+2)
    ax2.set_ylabel(r"$\mathrm{\Delta \ Log \ Amplitude}$", fontsize=font_size+2)
    ax2.set_xlim(-0.01, 1.01)

    if len(model_names) > 1:
        # ax2.legend(proxy_list, model_names, loc='upper left', fontsize=font_size)
        ax2.legend(proxy_list, model_names, fontsize=font_size)
        plt.grid(ls='--', alpha=0.5, axis='y')

    from matplotlib.ticker import FormatStrFormatter
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.show()
    plt.savefig(
        os.path.join(save_path, f'high_freq_fft_features.{save_format}'),
        dpi=dpi, bbox_inches='tight', format=save_format)
    plt.close()


def plot_var_feat(args, variances, save_path, model_names=None, save_format='png'):
    # Plot Fig 9: "Feature map variance"

    # plot settings
    alpha_base = 0.9
    font_size = 13
    cmap_name = "plasma"
    liner = "solid"

    if model_names is None:
        dpi = 120
        model_names = ['ssl_' + args.model_name]
        variances = [variances]
    else:
        dpi = 400
        assert isinstance(model_names, list) and len(model_names) >= 1
        zipped = zip(model_names, variances)
        zipped = sorted(zipped, key=lambda x:x[0])
        zipped = zip(*zipped)
        model_names, variances = [list(x) for x in zipped]
    
    fig, ax2 = plt.subplots(1, 1, figsize=(6.5, 5), dpi=dpi)
    proxy_list = []
    for i in range(len(model_names)):
        print(i, model_names[i], len(variances[i]))
        if "resnet" in args.model_name:
            pools = [4, 8, 14]
            msas = []
            marker = "D"
            color = "tab:blue"
        elif "vit" in args.model_name or "deit" in args.model_name:
            pools = []
            msas = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23,]  # vit-tiny
            marker = "o"
            color = "tab:red"
        else:
            import warnings
            warnings.warn("The configuration for %s are not implemented." % args.model_name, Warning)
            pools, msas = [], []
            marker = "s"
            color = "tab:green"
        
        # setup
        model_names[i], alpha, marker, liner, cmap_name, zorder = set_plot_args(model_names[i], i, alpha_base)
        # add legend
        proxy_list.append(make_proxy(cmap_name, marker, liner, linewidth=2))

        # Normalize
        depths = range(len(variances[i]))
        depth = len(depths) - 1
        depths = (np.array(depths)) / depth
        pools = (np.array(pools)) / depth
        msas = (np.array(msas)) / depth

        lc = plot_segment(ax2, depths, variances[i],
                     marker=marker, liner=liner, alpha=alpha, cmap_name=cmap_name, zorder=zorder)
        # cmap = cm.get_cmap(cmap_name)
        # color = cmap(np.arange(4) / 4)[3]
        # ax2.plot(depths, variances[i], marker=marker, color=color, markersize=7)

    # ploting
    for pool in pools:
        ax2.axvspan(pool - 1.0 / depth, pool + 0.0 / depth, color="tab:blue", alpha=0.15, lw=0)
    for msa in msas:
        ax2.axvspan(msa - 1.0 / depth, msa + 0.0 / depth, color="tab:gray", alpha=0.15, lw=0)

    ax2.set_xlabel(r"$\mathrm{Normalized \ Depth}$", fontsize=font_size+2)
    ax2.set_ylabel(r"$\mathrm{Feature \ Map \ Variance}$", fontsize=font_size+2)
    ax2.set_xlim(-0.01, 1.01)

    if len(model_names) > 1:
        # ax2.legend(proxy_list, model_names, loc='upper left', fontsize=font_size)
        ax2.legend(proxy_list, model_names, fontsize=font_size)
        plt.grid(ls='--', alpha=0.5, axis='y')

    from matplotlib.ticker import FormatStrFormatter
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    plt.show()
    plt.savefig(
        os.path.join(save_path, f'variance_features.{save_format}'),
        dpi=dpi, bbox_inches='tight', format=save_format)
    plt.close()


def main(args, device, load_path=None, name_mapping=None, save_format='png'):

    exp_name = args.exp_name
    if isinstance(load_path, str):
        save_path = f"report/{exp_name}/{opt.model_name}/summary/"
        mmcv.mkdir_or_exist(save_path)
        assert os.path.exists(load_path)
        model_list = os.listdir(load_path)
        model_list.sort()

        if name_mapping is None:
            name_mapping = dict()
            for m in model_list:
                name_mapping[m] = "mim_" + m.split(".")[0]

        latents = []
        model_names = []
        for m in model_list:
            cur_model = m.split(".")[0]
            if cur_model not in name_mapping.keys():
                continue
            file_path = os.path.join(load_path, cur_model, f"{exp_name}_latents.pt")
            try:
                latents.append(torch.load(file_path)[f"{exp_name}_latents"])
            except:
                continue
            model_names.append(name_mapping[cur_model])
        
        if exp_name == "fourier":
            plot_fft_B(args, latents, save_path, model_names, save_format)
        else:
            plot_var_feat(args, latents, save_path, model_names, save_format)

    else:
        if opt.pretrained_path is not None:
            save_name = opt.pretrained_path.split("/")
            save_name = "{}_{}".format(save_name[-2].split(".pth")[0], save_name[-1].split(".pth")[0])
        else:
            save_name = opt.model_name
        save_path = f"report/{exp_name}/{opt.model_name}/{save_name}"
        print('start experiment:', save_name)

        latents = forward_model(args, device)
        save_path = save_path.split('.pth')[0]
        mmcv.mkdir_or_exist(save_path)

        if exp_name == "fourier":
            latents = plot_fourier_features(latents)
            torch.save(
                dict(fourier_latents=latents), os.path.join(save_path, f"{exp_name}_latents.pt"))
            plot_fft_A(args, latents, save_path, save_format)
            plot_fft_B(args, latents, save_path, save_format)
        else:
            latents = plot_variance_features(latents)
            torch.save(
                dict(variance_latents=latents), os.path.join(save_path, f"{exp_name}_latents.pt"))
            plot_var_feat(args, latents, save_path, save_format)
        

if __name__ == '__main__':
    opt = parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    assert opt.exp_name in ['fourier', 'variance', None,]
    
    # save_format = 'pdf'
    save_format = 'png'
    load_path = None
    name_mapping = None

    load_path = f"./{opt.exp_name}/resnet50/"

    name_mapping = dict(
        ### ResNet ###
        randinit="sl_Random",
        model_zoo_barlowtwins_r50_bs2048_ep300="cl_Barlow+Twins",
        model_zoo_byol_r50_bs4096_ep200="cl_BYOL",
        model_zoo_barlowtwins_r50_official_bs2048_ep1000="cl_BYOL",
        model_zoo_colorization_r50_vissl_in1k="mim_Inpainting",
        model_zoo_dino_r50_224_ep800="cl_DINO",
        model_zoo_timm_resnet50_rsb_a2_224_ep300="sl_DeiT+(Sup.)",
        model_zoo_mocov3_r50_official_ep300="cl_MoCoV3",
        r50_r50_m07_rgb_m_learn_l3_res_fc_k1_l1_sz224_fft05_re_fun0_4xb256_accu2_cos_fp16_ep100="mim_SimMIM",
        ### ViT ###
        deit_small_no_aug_smth_mix0_8_cut1_0_4xb256_ema_fp16_ep300_latest="sl_DeiT",
        model_zoo_vit_dino_deit_small_p16_224_ep300="cl_DINO",
        model_zoo_vit_mae_vit_base_p16_224_ep400="mim_MAE",
        model_zoo_vit_cae_vit_base_p16_224_ep300="mim_CAE",
        rand_model_zoo_vit_dino_deit_small_p16_224_ep300="sl_Random",
    )

    main(args=opt, device=device, load_path=load_path, name_mapping=name_mapping, save_format=save_format)
