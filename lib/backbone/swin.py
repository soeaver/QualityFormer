"""
Creates a Swin Model as defined in:
Liu Ze, Lin Yutong, Cao Yue, Hu Han, Wei Yixuan, Zhang Zheng, Li Stephen, Guo Baining. (2021 arxiv).
Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.
Copyright (c) Yang Lu, 2021
"""
import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from lib.layers import SwinTransformerBlock, make_norm


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, input_size=(224, 224), patch_size=(4, 4), dim_in=3, embed_dim=96, norm="LN", bn_eps=1e-6):
        super().__init__()
        patches_resolution = (input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        self.input_size = input_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.proj = nn.Conv2d(dim_in, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = make_norm(embed_dim, eps=bn_eps, norm=norm)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm="LN", bn_eps=1e-6):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = make_norm(4 * dim, eps=bn_eps, norm=norm)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.0, norm="LN", bn_eps=1e-6, act="GELU",
                 downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm=norm, bn_eps=bn_eps, act=act)
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm=norm, bn_eps=bn_eps)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Swin(nn.Module):
    def __init__(self, input_size=(224, 224), patch_size=(4, 4), embed_dim=96, depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24), mlp_ratios=(4, 4, 4, 4), window_size=7, qkv_bias=False, qk_scale=None,
                 norm="LN", bn_eps=1e-6, act="GELU", drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, ape=False,
                 use_checkpoint=False, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Swin, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.mlp_ratios = mlp_ratios
        self.ape = ape
        self.num_layers = len(depths)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

        self.patch_embed = PatchEmbed(input_size=input_size, patch_size=patch_size, dim_in=3, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                  self.patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer], num_heads=num_heads[i_layer], window_size=window_size,
                mlp_ratio=self.mlp_ratios[i_layer], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm=norm, bn_eps=bn_eps, act=act,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None, use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)
        self.norm = make_norm(self.num_features, eps=bn_eps, norm=norm)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self._init_weights()

    @property
    def stage_out_dim(self):
        return [self.embed_dim, self.embed_dim, self.embed_dim * 2, self.embed_dim * 4, self.embed_dim * 8]

    @property
    def stage_out_spatial(self):
        return [1 / 4., 1 / 4., 1 / 8., 1 / 16., 1 / 32.]

    def _init_weights(self):
        # weight initialization
        def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
            def norm_cdf(x):
                return (1. + math.erf(x / math.sqrt(2.))) / 2.

            if (mean < a - 2 * std) or (mean > b + 2 * std):
                warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                              "The distribution of values may be incorrect.", stacklevel=2)
            with torch.no_grad():
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)
                tensor.uniform_(2 * l - 1, 2 * u - 1)
                tensor.erfinv_()
                tensor.mul_(std * math.sqrt(2.))
                tensor.add_(mean)
                tensor.clamp_(min=a, max=b)
                return tensor

        if self.ape:
            trunc_normal_(self.absolute_pos_embed, std=.02)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x
