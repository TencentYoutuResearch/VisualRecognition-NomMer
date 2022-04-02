# -*- encoding: utf-8 -*-
# ----------------------------------------------
# filename        :nommer.py
# description     :NomMer: Nominate Synergistic Context in Vision Transformer for Visual Recognition
# date            :2021/12/28 17:45:26
# author          :clark
# version number  :1.0
# ----------------------------------------------


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, depth=-1):
        super().__init__()
        self.dim = dim
        self.depth = depth
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        b, n, channels, h = *x.shape, self.heads
        qkv = self.to_qkv(x).reshape(b, n, 3, h, channels // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(b, n, channels)

        return self.to_out(x)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    def __init__(self, inplanes, planes, expansion=4, stride=1, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes // expansion) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes)
        self.bn3 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


def init_dct_kernel(in_ch, ksize=8, rsize=2):
    """[init a conv2d kernel for dct]

    Args:
        in_ch ([int]): [input dims]
        ksize (int, optional): [kernel size for dct]. Defaults to 8.
        rsize (int, optional): [reserve size for dct kernel]. Defaults to 2.

    Returns:
        [nn.Conv2d]: []
    """
    DCT_filter_n = np.zeros([ksize, ksize, 1, rsize**2])
    XX, YY = np.meshgrid(range(ksize), range(ksize))
    # DCT basis as filters
    C = np.ones(ksize)
    C[0] = 1 / np.sqrt(2)
    for v in range(rsize):
        for u in range(rsize):
            kernel = (
                (2 * C[v] * C[u] / ksize)
                * np.cos((2 * YY + 1) * v * np.pi / (2 * ksize))
                * np.cos((2 * XX + 1) * u * np.pi / (2 * ksize))
            )
            DCT_filter_n[:, :, 0, u + v * rsize] = kernel
    DCT_filter_n = np.transpose(DCT_filter_n, (3, 2, 0, 1))
    DCT_filter = torch.tensor(DCT_filter_n).float()

    DCT_filters = [DCT_filter for i in range(0, in_ch)]
    DCT_filters = torch.cat(DCT_filters, 0)

    dct_conv = nn.Conv2d(
        in_ch, rsize**2 * in_ch, kernel_size=(ksize, ksize), stride=ksize, padding=0, groups=in_ch, bias=False
    )
    dct_conv.weight = torch.nn.Parameter(DCT_filters)
    dct_conv.weight.requires_grad = False
    dct_conv.requires_grad = False

    return dct_conv


def init_idct_kernel(out_ch, ksize=8, rsize=2):
    """[init a conv2d kernel for idct]

    Args:
        out_ch ([int]): [output dims]
        ksize (int, optional): [kernel size for idct]. Defaults to 8.
        rsize (int, optional): [reserve size for idct kernel]. Defaults to 2.

    Returns:
        [nn.Conv2d]: []
    """
    IDCT_filter_n = np.zeros([1, 1, rsize**2, ksize**2])
    # IDCT basis as filters
    C = np.ones(ksize)
    C[0] = 1 / np.sqrt(2)
    for v in range(rsize):
        for u in range(rsize):
            for j in range(ksize):
                for i in range(ksize):
                    kernel = (
                        (2 * C[v] * C[u] / ksize)
                        * np.cos((2 * j + 1) * v * np.pi / (2 * ksize))
                        * np.cos((2 * i + 1) * u * np.pi / (2 * ksize))
                    )
                    IDCT_filter_n[0, 0, u + v * rsize, i + j * ksize] = kernel

    IDCT_filter_n = np.transpose(IDCT_filter_n, (3, 2, 0, 1))
    IDCT_filter = torch.tensor(IDCT_filter_n).float()
    IDCT_filters = [IDCT_filter for i in range(0, out_ch)]
    IDCT_filters = torch.cat(IDCT_filters, 0)

    idct_conv = nn.Conv2d(
        rsize**2 * out_ch, ksize**2 * out_ch, kernel_size=(1, 1), stride=1, padding=0, groups=out_ch, bias=False
    )
    idct_conv.weight = torch.nn.Parameter(IDCT_filters)
    idct_conv.weight.requires_grad = False
    idct_conv.requires_grad = False

    return idct_conv


class DCTAttention(nn.Module):
    def __init__(self, dim, ksize, heads):
        super().__init__()
        self.ksize = ksize
        reserve_kernel = (ksize + 1) // 2
        reserve_size = (reserve_kernel) ** 2
        self.dct_conv_8x8 = init_dct_kernel(dim, ksize, reserve_kernel)
        self.dw = nn.Conv2d(reserve_size * dim, dim, kernel_size=1, stride=1, padding=0, groups=8, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim // heads)
        self.up = nn.Conv2d(dim, reserve_size * dim, kernel_size=1, stride=1, padding=0, groups=8, bias=False)
        self.idct_conv = nn.Sequential(init_idct_kernel(dim, ksize, reserve_kernel), nn.PixelShuffle(ksize))

    def forward(self, x):
        B, _, _, _ = x.shape

        input_8x8 = self.dct_conv_8x8(x)
        _, _, h, w = input_8x8.shape

        x = self.dw(input_8x8)
        x = self.bn1(x)

        x = self.attn(x.permute(0, 2, 3, 1).flatten(1, 2))

        x = self.up(x.permute(0, 2, 1).view(B, -1, h, w))
        x = self.idct_conv(x)

        return x


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    U = U.cuda()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, tau=1):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    from https://github.com/ericjang/gumbel-softmax
    when we use torch.nn.functional.gumbel_softmax and set amp-opt-level==1, train failed(Gradient overflow always)
    """
    y = gumbel_softmax_sample(logits, tau)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard


class HybridAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0, depth=0, wsize=-1, psize=-1, cnn_expansion=4):
        """[a hybrid attention that can dynamically Nominate the synergistic global-local context]

        Args:
            dim ([int]): []
            heads (int, optional): []. Defaults to 8.
            dropout ([float], optional): []. Defaults to 0..
            depth (int, optional): []. Defaults to 0.
            wsize (int, optional): []. Defaults to -1.
            psize (int, optional): [used for dct-attention]. Defaults to -1.
            cnn_expansion (int, optional): []. Defaults to 4.
        """
        super().__init__()
        self.depth = depth
        self.wsize = wsize
        self.psize = psize
        self.cnn = Bottleneck(dim, dim, expansion=cnn_expansion)
        self.attention = WindowAttention(dim, to_2tuple(wsize), heads, proj_drop=0.0)
        self.attentionG = DCTAttention(dim, psize, heads)
        self.fc = nn.Linear(dim, 3)

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x):
        _, H, W, _ = x.shape

        x1 = self.cnn(x.permute(0, 3, 1, 2))
        x1 = x1.permute(0, 2, 3, 1)

        x2 = self.window_partition(x, self.wsize).flatten(1, 2)
        x2 = self.attention(x2)
        x2 = self.window_reverse(x2, self.wsize, H, W)

        x3 = self.attentionG(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        hybrid_x = x1 + x2 + x3
        logits = self.fc(hybrid_x)

        if self.training:
            logits = gumbel_softmax(logits, tau=1, hard=True)
        else:
            _values, _indexes = torch.max(logits, -1)
            _values = _values.unsqueeze(-1)
            logits = torch.eq(logits, _values).long()

        x = torch.stack([x1, x2, x3], -2)
        x = torch.mul(x, logits.unsqueeze(-1))
        x = torch.sum(x, -2)

        return x


class HybridNet(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, wsize, psize, cnn_expansion, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        for n in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            HybridAttention(
                                dim,
                                heads=heads,
                                dropout=dropout,
                                depth=n,
                                wsize=wsize,
                                psize=psize,
                                cnn_expansion=cnn_expansion,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        DropPath(drop_path[n]),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff, drop in self.layers:
            shortcut = x
            x = shortcut + drop(attn(x))
            x = x + drop(ff(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.0, drop_path=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        assert dim % heads == 0
        for n in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim // heads, dropout=dropout, depth=n)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        DropPath(drop_path[n]),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff, drop in self.layers:
            shortcut = x
            x = shortcut + drop(attn(x))
            x = x + drop(ff(x))
        return x


class MergeBlock(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 1, 1)
        self.pool = nn.MaxPool2d((2, 2))
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        if x.dim() == 4:
            B, H, W, C = x.shape
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv(x)
            x = self.pool(x)
            x = x.permute(0, 2, 3, 1).contiguous()
        else:
            B, new_HW, C = x.shape
            H = W = int(np.sqrt(new_HW))
            x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
            x = self.conv(x)
            x = self.pool(x)
            B, C = x.shape[:2]
            x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x


class NomMerAttn(nn.Module):
    def __init__(
        self,
        emd_dim=128,
        depths=None,
        num_heads=None,
        input_size=224,
        win_size=7,
        pool_size=None,
        cnn_expansion=None,
        drop_path_rate=0.1,
        num_class=1000,
    ):
        """
        Args:
            emd_dim (int, optional): []. Defaults to 128.
            depths (list, optional): []. Defaults to [2,2,16,2].
            num_heads (list, optional): []. Defaults to [2,4,8,16].
            input_size (int, optional): []. Defaults to 224.
            win_size (int, optional): []. Defaults to 7.
            cnn_expansion (list, optional): []. Defaults to [4,4].
            drop_path_rate (float, optional): []. Defaults to 0.1.
            num_class (int, optional): []. Defaults to 1000.
        """
        super().__init__()

        self.cnn1 = nn.Conv2d(3, emd_dim, kernel_size=4, stride=4, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(emd_dim)
        self.relu = nn.ReLU(inplace=True)

        self.pos_embedding = nn.Parameter(torch.randn(1, input_size // 16, input_size // 16, emd_dim * 4))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emd_dim * 8))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.transformer1 = HybridNet(
            emd_dim,
            depths[0],
            num_heads[0],
            emd_dim * 2,
            win_size,
            pool_size[0],
            cnn_expansion[0],
            0.0,
            dpr[0 : sum(depths[:1])],
        )
        self.merge1 = MergeBlock(emd_dim, emd_dim * 2)

        self.transformer2 = HybridNet(
            emd_dim * 2,
            depths[1],
            num_heads[1],
            emd_dim * 4,
            win_size,
            pool_size[1],
            cnn_expansion[1],
            0.0,
            dpr[sum(depths[:1]) : sum(depths[:2])],
        )
        self.merge2 = MergeBlock(emd_dim * 2, emd_dim * 4)

        self.transformer3 = Transformer(
            emd_dim * 4, depths[2], num_heads[2], emd_dim * 8, 0.0, dpr[sum(depths[:2]) : sum(depths[:3])]
        )
        self.merge3 = MergeBlock(emd_dim * 4, emd_dim * 8)

        self.transformer4 = Transformer(emd_dim * 8, depths[3], num_heads[3], emd_dim * 8, 0.0, dpr[sum(depths[:3]) :])

        self.num_class = num_class
        self.mlp_head = nn.Sequential(nn.LayerNorm(emd_dim * 8), nn.Linear(emd_dim * 8, self.num_class))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.cnn1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = x.permute(0, 2, 3, 1)
        b, _, _, _ = x.shape

        x = self.transformer1(x)
        x = self.merge1(x)

        x = self.transformer2(x)
        x = self.merge2(x)

        x = x + self.pos_embedding
        x = torch.flatten(x, 1, 2)
        x = self.transformer3(x)
        x = self.merge3(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.transformer4(x)

        x = x[:, 0]
        out = self.mlp_head(x)

        return out
