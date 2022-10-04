import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import lib.ops as ops
from .wrappers import make_act, make_norm


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class MLP(nn.Module):
    def __init__(self, inplanes, midplanes=None, outplanes=None, use_dwconv=False, act="GELU", drop=0.0):
        super().__init__()
        outplanes = outplanes or inplanes
        midplanes = midplanes or inplanes
        self.use_dwconv = use_dwconv

        self.fc1 = nn.Linear(inplanes, midplanes)
        if self.use_dwconv:
            self.dwconv = DWConv(midplanes)
        self.act = make_act(act=act)
        self.fc2 = nn.Linear(midplanes, outplanes)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H=None, W=None, rel_pos_bias=None):
        x = self.fc1(x)
        if self.use_dwconv:
            x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, mid_dim=None, num_head=8, qkv_bias=False, qk_scale=None, skip_connect=False,
                 attn_drop=0.0, proj_drop=0.0, sr_ratio=1, window_size=None):
        super().__init__()
        self.num_head = num_head
        head_dim = dim // num_head
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.mid_dim = mid_dim if mid_dim is not None else dim
        self.skip_connect = skip_connect

        self.qkv = nn.Linear(dim, self.mid_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.mid_dim, self.mid_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H=None, W=None, rel_pos_bias=None):
        if len(x.size()) == 3:
            B, N, _ = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_head, self.mid_dim // self.num_head).permute(2, 0, 3, 1, 4)
        elif len(x.size()) == 4:
            B, H, W, C = x.shape
            qkv = self.qkv(x).reshape(B, H * W, 3, self.num_head, C // self.num_head).permute(2, 0, 3, 1, 4)
        else:
            raise NotImplementedError
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if len(x.size()) == 3:
            x = (attn @ v).transpose(1, 2).reshape(B, N, self.mid_dim)
        elif len(x.size()) == 4:
            x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
        else:
            raise NotImplementedError
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.skip_connect:
            x = v.squeeze(1) + x  # the original x has different size with current x, use v to do skip connection
        return x


class AttentionSR(nn.Module):
    def __init__(self, dim, mid_dim=None, num_head=8, qkv_bias=False, qk_scale=None, skip_connect=False,
                 attn_drop=0.0, proj_drop=0.0, sr_ratio=1, window_size=None):
        super().__init__()
        self.num_head = num_head
        head_dim = dim // num_head
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.mid_dim = mid_dim if mid_dim is not None else dim
        self.skip_connect = skip_connect
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, self.mid_dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, self.mid_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.mid_dim, self.mid_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(self.mid_dim, self.mid_dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(self.mid_dim)

    def forward(self, x, H=None, W=None, rel_pos_bias=None):
        B, N, C = x.shape

        q = self.q(x).reshape(B, N, self.num_head, self.mid_dim // self.num_head).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_head, self.mid_dim // self.num_head).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_head, self.mid_dim // self.num_head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.mid_dim)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.skip_connect:
            x = v.squeeze(1) + x  # the original x has different size with current x, use v to do skip connection
        return x


class AttentionWindow(nn.Module):
    def __init__(self, dim, mid_dim=None, num_head=8, qkv_bias=False, qk_scale=None, skip_connect=False,
                 attn_drop=0.0, proj_drop=0.0, sr_ratio=1, window_size=None):
        super().__init__()
        self.num_head = num_head
        head_dim = dim // num_head
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.mid_dim = mid_dim if mid_dim is not None else dim
        self.skip_connect = skip_connect

        self.qkv = nn.Linear(dim, self.mid_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(self.mid_dim))
            self.v_bias = nn.Parameter(torch.zeros(self.mid_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            # 2*Wh-1 * 2*Ww-1, nH
            self.relative_position_bias_table = nn.Parameter(torch.zeros(self.num_relative_distance, num_head))
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.mid_dim, self.mid_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H=None, W=None, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_head, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            if attn.size()[-1] == relative_position_bias.size()[-1] and attn.size()[-2] == \
                    relative_position_bias.size()[-2]:
                attn = attn + relative_position_bias.unsqueeze(0)
            else:
                attn = attn + F.interpolate(relative_position_bias.unsqueeze(0), size=attn.size()[2:], mode='bilinear')

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.skip_connect:
            x = v.squeeze(1) + x  # the original x has different size with current x, use v to do skip connection
        return x


class AttnBlock(nn.Module):
    def __init__(self, dim, num_head, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm="LN", bn_eps=1e-6, act="GELU", att="", sr_ratio=1, use_dwconv=False,
                 init_value=0.0, window_size=None):
        super().__init__()
        if att == "SR":
            attention = AttentionSR
        elif att == "Window":
            attention = AttentionWindow
        else:
            attention = Attention
        self.norm1 = make_norm(dim, eps=bn_eps, norm=norm)
        self.attn = attention(
            dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            sr_ratio=sr_ratio, window_size=window_size
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = ops.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = make_norm(dim, eps=bn_eps, norm=norm)
        mlp_midplanes = int(dim * mlp_ratio)
        self.mlp = MLP(dim, midplanes=mlp_midplanes, use_dwconv=use_dwconv, act=act, drop=drop)

        if init_value > 0:
            self.gamma_1 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, H=None, W=None, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W, rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), H, W, rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x), H, W))
        return x


class TokenTransformer(nn.Module):
    def __init__(self, dim, mid_dim, num_head, mlp_ratio=1.0, qkv_bias=False, qk_scale=None, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm="LN", bn_eps=1e-5, act="GELU"):
        super().__init__()
        self.norm1 = make_norm(dim, eps=bn_eps, norm=norm)
        self.attn = Attention(
            dim, mid_dim, num_head=num_head, qkv_bias=qkv_bias, qk_scale=qk_scale, skip_connect=True,
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = ops.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = make_norm(mid_dim, eps=bn_eps, norm=norm)
        mlp_midplanes = int(mid_dim * mlp_ratio)
        self.mlp = MLP(mid_dim, midplanes=mlp_midplanes, act=act, drop=drop)

    def forward(self, x):
        x = self.attn(self.norm1(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TokenPerformer(nn.Module):
    def __init__(self, dim, mid_dim, num_head=1, kernel_ratio=0.5, dp1=0.1, dp2=0.1, norm="LN", bn_eps=1e-5,
                 act="GELU"):
        super().__init__()
        self.emb = mid_dim * num_head  # we use 1, so it is no need here
        self.kqv = nn.Linear(dim, 3 * self.emb)
        self.dp = nn.Dropout(dp1)
        self.proj = nn.Linear(self.emb, self.emb)
        self.norm1 = make_norm(dim, eps=bn_eps, norm=norm)
        self.norm2 = make_norm(self.emb, eps=bn_eps, norm=norm)
        self.epsilon = 1e-8  # for stable in division

        self.mlp = nn.Sequential(
            nn.Linear(self.emb, 1 * self.emb),
            make_act(act=act),
            nn.Linear(1 * self.emb, self.emb),
            nn.Dropout(dp2),
        )

        self.m = int(self.emb * kernel_ratio)
        self.w = torch.randn(self.m, self.emb)
        self.w = nn.Parameter(nn.init.orthogonal_(self.w) * math.sqrt(self.m), requires_grad=False)

    def prm_exp(self, x):
        # part of the function is borrow from https://github.com/lucidrains/performer-pytorch
        # and Simo Ryu (https://github.com/cloneofsimo)
        # ==== positive random features for gaussian kernels ====
        # x = (B, T, hs)
        # w = (m, hs)
        # return : x : B, T, m
        # SM(x, y) = E_w[exp(w^T x - |x|/2) exp(w^T y - |y|/2)]
        # therefore return exp(w^Tx - |x|/2)/sqrt(m)
        xd = ((x * x).sum(dim=-1, keepdim=True)).repeat(1, 1, self.m) / 2
        wtx = torch.einsum('bti,mi->btm', x.float(), self.w)

        return torch.exp(wtx - xd) / math.sqrt(self.m)

    def single_attn(self, x):
        k, q, v = torch.split(self.kqv(x), self.emb, dim=-1)
        kp, qp = self.prm_exp(k), self.prm_exp(q)  # (B, T, m), (B, T, m)
        D = torch.einsum('bti,bi->bt', qp, kp.sum(dim=1)).unsqueeze(dim=2)  # (B, T, m) * (B, m) -> (B, T, 1)
        kptv = torch.einsum('bin,bim->bnm', v.float(), kp)  # (B, emb, m)
        y = torch.einsum('bti,bni->btn', qp, kptv) / (D.repeat(1, 1, self.emb) + self.epsilon)  # (B, T, emb)/Diag
        # skip connection
        y = v + self.dp(self.proj(y))  # same as token_transformer in T2T layer, use v as skip connection

        return y

    def forward(self, x):
        x = self.single_attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


def get_sinusoid_encoding(n_position, d_hid):
    """ Sinusoid position encoding table """

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)
