# Copyright 2021  Facebook. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This code is modified by Zilliz.
import collections
import math
from typing import Tuple, List, Callable

import numpy
import torch
from torch import nn
from functools import partial
from itertools import repeat

def round_width(width, multiplier, min_width=8, divisor=8, ceil=False):
    """
    Round width of filters based on width multiplier
    Args:
        width ('int'):
            The channel dimensions of the input.
        multiplier ('float'):
            The multiplication factor.
        min_width ('int'):
            The minimum width after multiplication.
        divisor ('int'):
            The new width should be dividable by divisor.
        ceil ('bool'):
            If True, use ceiling as the rounding method.
    """
    if not multiplier:
        return width

    width *= multiplier
    min_width = min_width or divisor
    if ceil:
        width_out = max(min_width, int(math.ceil(width / divisor)) * divisor)
    else:
        width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)

def _ntuple(n: int) -> Callable:
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def init_vit_weights(model, trunc_normal_std=0.02) -> None:
    """
    Weight initialization for vision transformers.
    Args:
        model(nn.Module):
            Model to be initialized.
        trunc_normal_std(float):
            The expected standard deviation for fully-connected layer and ClsPositionalEncoding.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=trunc_normal_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, SpatialTemporalClsPositionalEncoding):
            for weights in m.parameters():
                nn.init.trunc_normal_(weights, std=trunc_normal_std)


class PatchEmbed2D(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, h, w = x.shape
        assert h == self.img_size[0] and w == self.img_size[1], \
            f'Input image size ({h}*{w}) doesn\'t match model ({self.img_size[0]}*{self.img_size[1]}).'
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x




class AttentionPool(nn.Module):
    """
    A MLP block that contains two linear layers with a normalization layer.
    the MLP block is used in a transformer model after the attention block.
    ::
                                        Input
                                           ↓
                                        Reshape
                                           ↓
                                          Pool
                                           ↓
                                        Reshape
                                           ↓
                                          norm
    Args:
        thw_shape(List):
            the shape of the input tensor (before flattening).
        pool(Callable):
            Pool operation that is applied to the input tensor.
            If pool is None, return the input tensor.
        has_cls_embed(bool):
            whether the input tensor contains cls token. Pool operation excludes cls token.
        norm(Callable):
            Optional normalization operation applied to tensor after pool.
    Returns:
        tensor(torch.Tensor):
            Input tensor after pool.
        thw_shape(List[int]):
            Output tensor shape (before flattening).
    """
    def __init__(
        self,
        thw_shape,
        pool=None,
        has_cls_embed=True,
        norm=None
    ) -> None:
        super().__init__()
        self.pool = pool
        self.thw_shape = thw_shape
        self.has_cls_embed = has_cls_embed
        self.norm = norm

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        if self.pool is None:
            return x, self.thw_shape
        tensor_dim = x.ndim
        if tensor_dim == 4:
            pass
        elif tensor_dim == 3:
            x = x.unsqueeze(1)
        else:
            raise NotImplementedError(f"Unsupported input dimension {x.shape}")

        if self.has_cls_embed:
            cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]

        b, n, _, c = x.shape
        t, h, w = self.thw_shape
        x = x.reshape(b * n, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()

        x = self.pool(x)

        thw_shape = [x.shape[2], x.shape[3], x.shape[4]]
        l_pooled = x.shape[2] * x.shape[3] * x.shape[4]
        x = x.reshape(b, n, c, l_pooled).transpose(2, 3)
        if self.has_cls_embed:
            x = torch.cat((cls_tok, x), dim=2)
        if self.norm is not None:
            x = self.norm(x)

        if tensor_dim == 4:
            pass
        else:  # For the case tensor_dim == 3.
            x = x.squeeze(1)
        return x, thw_shape


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MultiScaleBlock(nn.Module):
    """
    A multiscale vision transformer block.
    Each block contains a multiscale attention layer and a Mlp layer.
    ::
                                      Input
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                MultiScaleAttention        Pool
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation ←-------------+
                                        |
                                        |-------------------+
                                        ↓                   |
                                       Norm                 |
                                        ↓                   |
                                       Mlp                 Proj
                                        ↓                   |
                                     DropPath               |
                                        ↓                   |
                                    Summation  ←------------+
    Args:
        dim(int):
            Input feature dimension.
        dim_out(int):
            Output feature dimension.
        num_heads(int):
            Number of heads in the attention layer.
        mlp_ratio(float):
            MLP ratio which controls the feature dimension in the hidden layer of the MLP block.
        qkv_bias(bool):
            If set to False, the qkv layer will not learn an additive bias.
        dropout_rate(float):
            DropOut rate. If set to 0, DropOut is disabled.
        droppath_rate(float):
            DropPath rate. If set to 0, DropPath is disabled.
        activation(nn.Module):
            Activation layer used in the MLP layer.
        norm_layer(nn.Module):
            Normalization layer.
        kernel_q(_size_3_t):
            Pooling kernel size for q. If pooling kernel size is 1 for all the dimensions.
        kernel_kv(_size_3_t):
            Pooling kernel size for kv. If pooling kernel size is 1 for all the dimensions, pooling is not used.
        stride_q(_size_3_t):
            Pooling kernel stride for q.
        stride_kv(_size_3_t):
            Pooling kernel stride for kv.
        pool_mode(nn.Module):
            Pooling mode.
        has_cls_embed(bool):
            If set to True, the first token of the input tensor should be a cls token.
            Otherwise, the input tensor does not contain a cls token. Pooling is not applied to the cls token.
        pool_first(bool):
            If set to True, pool is applied before qkv projection. Otherwise, pool is applied after qkv projection.
    """

    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        dropout_rate=0.0,
        droppath_rate=0.0,
        activation=nn.GELU,
        norm_layer=nn.LayerNorm,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        pool_mode=nn.Conv3d,
        has_cls_embed=True,
        pool_first=False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dropout_rate=dropout_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            pool_mode=pool_mode,
            pool_first=pool_first,
        )
        self.drop_path = (
            DropPath(drop_prob=droppath_rate) if droppath_rate > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim_out,
            act_layer=activation,
            drop=dropout_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool3d(kernel_skip, stride_skip, padding_skip, ceil_mode=False)
            if len(kernel_skip) > 0
            else None
        )

    def forward(
        self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x(torch.Tensor):
                Input tensor.
            thw_shape(List):
                The shape of the input tensor (before flattening).
        """

        x_block, thw_shape_new = self.attn(self.norm1(x), thw_shape)
        atn = AttentionPool(
            pool=self.pool_skip,
            thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed
        )
        x_res, _ = atn(x)
        x = x_res + self.drop_path(x_block)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp)
        return x, thw_shape_new


class MultiScaleAttention(nn.Module):
    """
    A multiscale attention block.
    compare to a conventional attention block, a multiscale attention block optionally
    supports pooling (either before or after qkv projection). If pooling is not used, a
    multiscale attention block is equivalent to a conventional attention block.
    ::
                                   Input
                                     |
                    |----------------|-----------------|
                    ↓                ↓                 ↓
                  Linear           Linear            Linear
                    &                &                 &
                 Pool (Q)         Pool (K)          Pool (V)
                    → -------------- ←                 |
                             ↓                         |
                       MatMul & Scale                  |
                             ↓                         |
                          Softmax                      |
                             → ----------------------- ←
                                         ↓
                                   MatMul & Scale
                                         ↓
                                      DropOut
    Args:
        dim(int):
            Input feature dimension.
        num_heads(int):
            number of heads in the attention layer.
        qkv_bias(bool):
            If set to False, the qkv layer will not learn an additive bias.
        dropout_rate(float):
            Dropout rate.
        kernel_q(_size_3_t):
            Pooling kernel size for q. If both pooling kernel
            size and pooling stride size are 1 for all the dimensions, pooling is disabled.
        kernel_kv(_size_3_t):
            Pooling kernel size for kv. If both pooling kernel size and pooling stride size
            are 1 for all the dimensions, pooling is disabled.
        stride_q(_size_3_t):
            Pooling kernel stride for q.
        stride_kv(_size_3_t):
            Pooling kernel stride for kv.
        norm_layer(nn.Module):
            normalization layer used after pooling.
        has_cls_embed(bool):
            If set to True, the first token of the input tensor
            should be a cls token. Otherwise, the input tensor does not contain a cls token.
            Pooling is not applied to the cls token.
        pool_mode(str):
            Pooling mode. Option includes "conv" (learned pooling), "avg"
            (average pooling), and "max" (max pooling).
        pool_first(bool):
            If set to True, pool is applied before qkv projection.
            Otherwise, pool is applied after qkv projection.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        dropout_rate=0.0,
        kernel_q=(1, 1, 1),
        kernel_kv=(1, 1, 1),
        stride_q=(1, 1, 1),
        stride_kv=(1, 1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        pool_mode=nn.Conv3d,
        pool_first=False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.dropout_rate = dropout_rate
        self.kernel_q = kernel_q
        self.kernel_kv = kernel_kv
        self.stride_q = stride_q
        self.stride_kv = stride_kv
        self.norm_layer = norm_layer
        self.has_cls_embed = has_cls_embed
        self.pool_mode = pool_mode
        self.pool_first = pool_first

        assert self.pool_mode in [nn.Conv3d, nn.AvgPool3d, nn.MaxPool3d]

        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.padding_q = [int(q // 2) for q in self.kernel_q]
        self.padding_kv = [int(kv // 2) for kv in self.kernel_kv]

        self.q = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
        self.k = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
        self.v = nn.Linear(self.dim, self.dim, bias=self.qkv_bias)
        self.proj = nn.Linear(self.dim, self.dim)
        if self.dropout_rate > 0.0:
            self.proj_drop = nn.Dropout(self.dropout_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if (
            self.kernel_q is not None
            and numpy.prod(self.kernel_q) == 1
            and numpy.prod(self.stride_q) == 1
        ):
            self.kernel_q = None
        if (
            self.kernel_kv is not None
            and numpy.prod(self.kernel_kv) == 1
            and numpy.prod(self.stride_kv) == 1
        ):
            self.kernel_kv = None

        if self.pool_mode in (nn.AvgPool3d, nn.MaxPool3d):
            pool_op = nn.MaxPool3d if pool_mode == nn.MaxPool3d else nn.AvgPool3d
            self.pool_q = (
                pool_op(self.kernel_q, self.stride_q, self.padding_q, ceil_mode=False)
                if self.kernel_q is not None
                else None
            )
            self.pool_k = (
                pool_op(self.kernel_kv, self.stride_kv, self.padding_kv, ceil_mode=False)
                if self.kernel_kv is not None
                else None
            )
            self.pool_v = (
                pool_op(self.kernel_kv, self.stride_kv, self.padding_kv, ceil_mode=False)
                if self.kernel_kv is not None
                else None
            )
        elif self.pool_mode == nn.Conv3d:
            self.pool_q = (
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    self.kernel_q,
                    stride=self.stride_q,
                    padding=self.padding_q,
                    groups=self.head_dim,
                    bias=False,
                )
                if self.kernel_q is not None
                else None
            )
            self.norm_q = self.norm_layer(self.head_dim) if self.kernel_q is not None else None
            self.pool_k = (
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    self.kernel_kv,
                    stride=self.stride_kv,
                    padding=self.padding_kv,
                    groups=self.head_dim,
                    bias=False,
                )
                if self.kernel_kv is not None
                else None
            )
            self.norm_k = self.norm_layer(self.head_dim) if self.kernel_kv is not None else None
            self.pool_v = (
                nn.Conv3d(
                    self.head_dim,
                    self.head_dim,
                    self.kernel_kv,
                    stride=self.stride_kv,
                    padding=self.padding_kv,
                    groups=self.head_dim,
                    bias=False,
                )
                if self.kernel_kv is not None
                else None
            )
            self.norm_v = self.norm_layer(self.head_dim) if self.kernel_kv is not None else None
        else:
            raise NotImplementedError("Unsupported model.")

    def qkv_proj(
            self,
            q,
            q_size,
            k,
            k_size,
            v,
            v_size,
            batch_size,
            chan_size,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            q(torch.Tensor):
                q tensor.
            q_size(List[int]):
                q tensor size.
            k(torch.Tensor):
                k tensor.
            k_size(List[int]):
                k tensor size.
            v(torch.Tensor):
                v tensor.
            v_size(List[int]):
                v tensor size.
            batch_size(List[int]):
                batch size.
            chan_size(List[int]):
                channel size.
        """
        q = (
            self.q(q)
                .reshape(batch_size, q_size, self.num_heads, chan_size // self.num_heads)
                .permute(0, 2, 1, 3)
        )
        k = (
            self.k(k)
                .reshape(batch_size, k_size, self.num_heads, chan_size // self.num_heads)
                .permute(0, 2, 1, 3)
        )
        v = (
            self.v(v)
                .reshape(batch_size, v_size, self.num_heads, chan_size // self.num_heads)
                .permute(0, 2, 1, 3)
        )
        return q, k, v

    def qkv_pool(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            thw_shape: Tuple[torch.Tensor, List[int]],
    ) -> Tuple[
        torch.Tensor, List[int], torch.Tensor, List[int], torch.Tensor, List[int]
    ]:
        """
        Args:
            q(torch.Tensor):
                q tensor.
            k(torch.Tensor):
                k tensor.
            v(torch.Tensor):
                v tensor.
            thw_shape(Tuple[torch.Tensor, List[int]]):
                The shape of the input tensor.
        """
        ap = AttentionPool(
            pool=self.pool_q,
            thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        q, q_shape = ap(q)
        ap = AttentionPool(
            pool=self.pool_k,
            thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        k, k_shape = ap(k)
        ap = AttentionPool(
            pool=self.pool_v,
            thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )
        v, v_shape = ap(v)
        return q, q_shape, k, k_shape, v, v_shape

    def get_qkv_length(
            self,
            q_shape,
            k_shape,
            v_shape,
    ) -> Tuple[int]:
        """
        Args:
            q_shape(List[int]):
                q tensor shape.
            k_shape(List[int]):
                k tensor shape.
            v_shape(List[int]):
                v tensor shape.
        """
        q_n = numpy.prod(q_shape) + 1 if self.has_cls_embed else numpy.prod(q_shape)
        k_n = numpy.prod(k_shape) + 1 if self.has_cls_embed else numpy.prod(k_shape)
        v_n = numpy.prod(v_shape) + 1 if self.has_cls_embed else numpy.prod(v_shape)
        return q_n, k_n, v_n

    def reshape_qkv_to_seq(
            self,
            q,
            k,
            v,
            q_n,
            v_n,
            k_n,
            b,
            c,
    ) -> Tuple[int]:
        """
        Args:
            q(torch.Tensor):
                q tensor.
            k(torch.Tensor):
                k tensor.
            v(torch.Tensor):
                v tensor.
            q_n(int):
                k tensor size.
            v_n(int):
                v tensor size.
            k_n(int):
                k tensor size.
            b(int):
                Reshaped size.
            c(int):
                Reshaped size.
        """
        q = q.permute(0, 2, 1, 3).reshape(b, q_n, c)
        v = v.permute(0, 2, 1, 3).reshape(b, v_n, c)
        k = k.permute(0, 2, 1, 3).reshape(b, k_n, c)
        return q, k, v

    def forward(
        self, x: torch.Tensor, thw_shape: List[int]
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Args:
            x(torch.Tensor):
                Input tensor.
            thw_shape(List):
                The shape of the input tensor (before flattening).
        """

        b, n, c = x.shape
        if self.pool_first:
            x = x.reshape(b, n, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
            q = k = v = x
            q, q_shape, k, k_shape, v, v_shape = self.qkv_pool(q, k, v, thw_shape)
            q_n, k_n, v_n = self.get_qkv_length(q_shape, k_shape, v_shape)
            q, k, v = self.reshape_qkv_to_seq(q, k, v, q_n, v_n, k_n, b, c)
            q, k, v = self.qkv_proj(q, q_n, k, k_n, v, v_n, b, c)
        else:
            q = k = v = x
            q, k, v = self.qkv_proj(q, n, k, n, v, n, b, c)
            q, q_shape, k, k_shape, v, v_shape = self.qkv_pool(q, k, v, thw_shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        n = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        if self.dropout_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape

class SequencePool(nn.Module):
    """
    Sequence pool produces a single embedding from a sequence of embeddings.
    Currently it supports "mean" and "cls".
    """

    def __init__(self, mode: str) -> None:
        """
        Args:
            mode ('str'):
                If set to "cls", it assumes the first element in the input is the cls token and returns it.
                If set to "mean", it returns the mean of the entire sequence.
        """
        super().__init__()
        self.mode = mode
        assert mode in ["cls", "mean"], "Unsupported mode for SequencePool."

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "cls":
            x = x[:, 0]
        elif self.mode == "mean":
            x = x.mean(1)
        else:
            raise NotImplementedError
        return x



class VisionTransformerBasicHead(nn.Module):
    """
    Vision transformer basic head.
    ::
                                      SequencePool
                                           ↓
                                        Dropout
                                           ↓
                                       Projection
                                           ↓
                                       Activation
    Args:
        in_features ('int'):
            Input channel size of the resnet head.
        out_features ('int'):
            Output channel size of the resnet head.
        seq_pool_type ('str'):
            Pooling type. It supports "cls", "mean " and "none". If set to
            "cls", it assumes the first element in the input is the cls token and
            returns it. If set to "mean", it returns the mean of the entire sequence.
        activation ('callable'):
            A callable that constructs vision transformer head activation layer,
            examples include: nn.ReLU, nn.Softmax, nn.Sigmoid, and None (not applying activation).
        dropout_rate ('float'):
            Dropout rate.
    """

    def __init__(
        self,
        *,
        in_features,
        out_features,
        seq_pool_type="cls",
        dropout_rate=0.5,
        activation=None,
    ) -> None:
        super().__init__()
        assert seq_pool_type in ["cls", "mean", "none"]

        if seq_pool_type in ["cls", "mean"]:
            self.seq_pool_model = SequencePool(seq_pool_type)
        elif seq_pool_type == "none":
            self.seq_pool_model = None
        else:
            raise NotImplementedError

        if activation is None:
            self.activation_model = None
        elif activation == nn.Softmax:
            self.activation_model = self.activation(dim=1)
        else:
            self.activation_model = self.activation()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None
        self.proj = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Performs pooling.
        if self.seq_pool_model is not None:
            x = self.seq_pool_model(x)
        # Performs dropout.
        if self.dropout is not None:
            x = self.dropout(x)
        # Performs projection.
        x = self.proj(x)
        # Performs activation.
        if self.activation_model is not None:
            x = self.activation_model(x)
        return x

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention layer.
    Args:
        dim (int): number of features
        num_heads (int): number of heads
        qkv_bias (bool): if add bias to qkv layer
        qk_scale (float): number to scale qk
        attn_drop_ratio (float): drop rate of attention layer
        proj_drop_ratio (float): drop rate of projection layer
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0,
                 proj_drop_ratio=0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        batch_size, new_num_patch, dim = x.shape

        qkv = self.qkv(x).reshape(
            batch_size,
            new_num_patch,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, new_num_patch, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SpatialTemporalClsPositionalEncoding(nn.Module):
    """
    Add a cls token and apply a spatial-temporal encoding to a tensor.
    Args:
        embed_dim(int):
            Embedding dimension for input sequence.
        patch_embed_shape(Tuple):
            The number of patches in each dimension (T, H, W) after patch embedding.
        sep_pos_embed(bool):
            If set to true, one positional encoding is used for
            spatial patches and another positional encoding is used for temporal
            sequence. Otherwise, only one positional encoding is used for all the patches.
        has_cls(bool):
            If set to true, a cls token is added in the beginning of each input sequence.
    """

    def __init__(
        self,
        embed_dim,
        patch_embed_shape,
        sep_pos_embed=False,
        has_cls=True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed_shape = patch_embed_shape
        self.sep_pos_embed = sep_pos_embed
        self.has_cls = has_cls
        assert (
            len(self.patch_embed_shape) == 3
        ), "Patch_embed_shape should be in the form of (T, H, W)."
        self.num_spatial_patch = patch_embed_shape[1] * patch_embed_shape[2]
        self.num_temporal_patch = patch_embed_shape[0]

        if self.has_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            num_patches = self.num_spatial_patch * self.num_temporal_patch + 1
        else:
            num_patches = self.num_spatial_patch * self.num_temporal_patch

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, self.num_spatial_patch, self.embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.num_temporal_patch, self.embed_dim)
            )
            if self.has_cls:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

    def get_patch_embed_shape(self):
        return self.patch_embed_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x(torch.Tensor):
                Input tensor.
        """
        b, _, _ = x.shape
        if self.has_cls:
            cls_tokens = self.cls_token.expand(b, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.num_temporal_patch, 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.num_spatial_patch,
                dim=1,
            )
            if self.has_cls:
                pos_embed = torch.cat([self.pos_embed_class, pos_embed], 1)
            x = x + pos_embed
        else:
            x = x + self.pos_embed

        return x


class MultiscaleVisionTransformers(nn.Module):
    """
    Multiscale Vision Transformers
    https://arxiv.org/abs/2104.11227
    ::
                                       PatchEmbed
                                           ↓
                                   PositionalEncoding
                                           ↓
                                        Dropout
                                           ↓
                                     Normalization
                                           ↓
                                         Block 1
                                           ↓
                                           .
                                           .
                                           .
                                           ↓
                                         Block N
                                           ↓
                                     Normalization
                                           ↓
                                          Head
    Args:
        patch_embed ('nn.Module'):
            Patch embed module.
        cls_positional_encoding ('nn.Module'):
            Positional encoding module.
        pos_drop ('nn.Module'):
            Dropout module after patch embed.
        norm_patch_embed ('nn.Module'):
            Normalization module after patch embed.
        blocks ('nn.ModuleList'):
            Stack of multi-scale transformer blocks.
        norm_embed ('nn.Module'):
            Normalization layer before head.
        head ('nn.Module'):
            Head module.
    """

    def __init__(
        self,
        *,
        patch_embed,
        cls_positional_encoding,
        pos_drop,
        norm_patch_embed,
        blocks,
        norm_embed,
        head,
    ) -> None:
        super().__init__()
        self.patch_embed: nn.Module = patch_embed
        self.cls_positional_encoding: nn.Module = cls_positional_encoding
        self.pos_drop: nn.Module = pos_drop
        self.norm_patch_embed: nn.ModuleList = norm_patch_embed
        self.blocks: nn.ModuleList = blocks
        self.norm_embed: nn.Module = norm_embed
        self.head: nn.Module = head

        assert hasattr(
            cls_positional_encoding, "patch_embed_shape"
        ), "cls_positional_encoding should have attribute patch_embed_shape."
        init_vit_weights(self, trunc_normal_std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_embed is not None:
            x = self.patch_embed(x)
        x = self.cls_positional_encoding(x)

        if self.pos_drop is not None:
            x = self.pos_drop(x)

        if self.norm_patch_embed is not None:
            x = self.norm_patch_embed(x)

        thw = self.cls_positional_encoding.patch_embed_shape
        for blk in self.blocks:
            x, thw = blk(x, thw)
        if self.norm_embed is not None:
            x = self.norm_embed(x)
        if self.head is not None:
            x = self.head(x)
        return x


def create_multiscale_vision_transformers(
    spatial_size,
    temporal_size,
    cls_embed_on=True,
    sep_pos_embed=True,
    depth=16,
    norm=nn.LayerNorm,
    enable_patch_embed=True,
    input_channels=3,
    patch_embed_dim=96,
    conv_patch_embed_kernel=(3, 7, 7),
    conv_patch_embed_stride=(2, 4, 4),
    conv_patch_embed_padding=(1, 3, 3),
    enable_patch_embed_norm=False,
    use_2d_patch=False,
    num_heads=1,
    mlp_ratio=4.0,
    qkv_bias=True,
    dropout_rate_block=0.0,
    droppath_rate_block=0.0,
    pooling_mode=nn.Conv3d,
    pool_first=False,
    embed_dim_mul=None,
    atten_head_mul=None,
    pool_q_stride_size=None,
    pool_kv_stride_size=None,
    pool_kv_stride_adaptive=None,
    pool_kvq_kernel=None,
    head=VisionTransformerBasicHead,
    head_dropout_rate=0.5,
    head_activation=None,
    head_num_classes=400
) -> nn.Module:
    """
    Multiscale Vision Transformers
    https://arxiv.org/abs/2104.11227
    ::
                                           PatchEmbed
                                               ↓
                                       PositionalEncoding
                                               ↓
                                            Dropout
                                               ↓
                                         Normalization
                                               ↓
                                             Block 1
                                               ↓
                                               .
                                               .
                                               .
                                               ↓
                                             Block N
                                               ↓
                                         Normalization
                                               ↓
                                              Head
    Args:
        spatial_size ('_size_2_t'):
            Input video spatial resolution (H, W). If a single int is given,
            it assumes the width and the height are the same.
        temporal_size ('int'):
            Number of frames in the input video.
        cls_embed_on ('bool'):
            If True, use cls embed in the model. Otherwise features are average
            pooled before going to the final classifier.
        sep_pos_embed ('bool'):
            If True, perform separate spatiotemporal embedding.
        depth ('int'):
            The depth of the model.
        norm ('Callable'):
            Normalization layer.
        enable_patch_embed ('bool'):
            If true, patchify the input video. If false, it assumes the input should
            have the feature dimension of patch_embed_dim.
        input_channels ('int'):
            Channel dimension of the input video.
        patch_embed_dim ('int'):
            Embedding dimension after patchifing the video input.
        conv_patch_embed_kernel ('Tuple[int]'):
            Kernel size of the convolution for patchifing the video input.
        conv_patch_embed_stride ('Tuple[int]'):
            Stride size of the convolution for patchifing the video input.
        conv_patch_embed_padding ('Tuple[int]'):
            Padding size of the convolution for patchifing the video input.
        enable_patch_embed_norm ('bool'):
            If True, apply normalization after patchifing the video input.
        use_2d_patch ('bool'):
            If True, use 2D convolutions to get patch embed. Otherwise, use 3D convolutions.
        num_heads ('int'):
            Number of heads in the first transformer block.
        mlp_ratio ('float'):
            MLP ratio which controls the feature dimension in the hidden layer of the Mlp block.
        qkv_bias ('bool'):
            If set to False, the qkv layer will not learn an additive bias.
        dropout_rate_block ('float'):
            Dropout rate for the attention block.
        droppath_rate_block ('float'):
            Droppath rate for the attention block.
        pooling_mode ('Callable'):
            Pooling mode.
        pool_first ('bool'):
            If set to True, pool is applied before qkv projection. Otherwise, pool is applied after qkv projection.
        embed_dim_mul ('List[List[int]]'):
            Dimension multiplication at layer i. If X is used, then the next block will increase
            the embed dimension by X times. Format: [depth_i, mul_dim_ratio].
        atten_head_mul ('List[List[int]]'):
            Head dimension multiplication at  layer i. If X is used, then the next block will increase
            the head by X times. Format: [depth_i, mul_dim_ratio].
        pool_q_stride_size ('List[List[int]]'):
            List of stride sizes for the pool q at each layer. Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,].
        pool_kv_stride_size ('List[List[int]]'):
            List of stride sizes for the pool kv at each layer. Format: [[i, stride_t_i, stride_h_i, stride_w_i], ...,].
        pool_kv_stride_adaptive ('_size_3_t'):
            Initial kv stride size for the first block. The stride size will be further reduced at the layer
            where q is pooled with the ratio of the stride of q pooling. If pool_kv_stride_adaptive is set,
            then pool_kv_stride_size should be none.
        pool_kvq_kernel ('_size_3_t'):
            Pooling kernel size for q and kv. It None, the kernel_size is [s + 1 if s > 1 else s for s in stride_size].
        head ('Callable'):
            Head model.
        head_dropout_rate ('float'):
            Dropout rate in the head.
        head_activation ('float'):
            Activation in the head.
        head_num_classes ('int'):
            Number of classes in the final classification head.
    """

    if use_2d_patch:
        assert temporal_size == 1, "If use_2d_patch, temporal_size needs to be 1."
    if pool_kv_stride_adaptive is not None:
        assert (
                pool_kv_stride_size is None
        ), "pool_kv_stride_size should be none if pool_kv_stride_adaptive is set."
    if norm == nn.LayerNorm:
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
    else:
        raise NotImplementedError("Only supports layernorm.")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    conv_patch_op = nn.Conv2d if use_2d_patch else nn.Conv3d
    if enable_patch_embed:
        patch_embed = PatchEmbed2D(
            in_channels=input_channels,
            out_channels=patch_embed_dim,
            conv_kernel_size=conv_patch_embed_kernel,
            conv_stride=conv_patch_embed_stride,
            conv_padding=conv_patch_embed_padding,
            conv=conv_patch_op,
        )
    else:
        patch_embed = None
    input_dims = [temporal_size, spatial_size[0], spatial_size[1]]
    if use_2d_patch:
        input_stirde = (1,) + tuple(conv_patch_embed_stride)
    else:
        input_stirde = conv_patch_embed_stride

    if enable_patch_embed:
        patch_embed_shape = [input_dims[i] // input_stirde[i] for i in range(len(input_dims))]
    else:
        patch_embed_shape = input_dims

    cls_positional_encoding = SpatialTemporalClsPositionalEncoding(
        embed_dim=patch_embed_dim,
        patch_embed_shape=patch_embed_shape,
        sep_pos_embed=sep_pos_embed,
        has_cls=cls_embed_on,
    )

    # stochastic depth decay rule
    dpr = [
        x.item() for x in torch.linspace(0, droppath_rate_block, depth)
    ]

    if dropout_rate_block > 0.0:
        pos_drop = nn.Dropout(p=dropout_rate_block)

    dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)
    if embed_dim_mul is not None:
        for i in range(len(embed_dim_mul)):
            dim_mul[embed_dim_mul[i][0]] = embed_dim_mul[i][1]
    if atten_head_mul is not None:
        for i in range(len(atten_head_mul)):
            head_mul[atten_head_mul[i][0]] = atten_head_mul[i][1]

    norm_patch_embed = norm_layer(patch_embed_dim) if enable_patch_embed_norm else None

    pool_q = [[] for i in range(depth)]
    pool_kv = [[] for i in range(depth)]
    stride_q = [[] for i in range(depth)]
    stride_kv = [[] for i in range(depth)]

    if pool_q_stride_size is not None:
        for i in range(len(pool_q_stride_size)):
            stride_q[pool_q_stride_size[i][0]] = pool_q_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_q[pool_q_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_q[pool_q_stride_size[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_q_stride_size[i][1:]
                ]

    if pool_kv_stride_adaptive is not None:
        stride_kv = pool_kv_stride_adaptive
        pool_kv_stride_size = []
        for i in range(depth):
            if len(stride_q[i]) > 0:
                stride_kv = [
                    max(stride_kv[d] // stride_q[i][d], 1)
                    for d in range(len(stride_kv))
                ]
            pool_kv_stride_size.append([i] + stride_kv)

    if pool_kv_stride_size is not None:
        for i in range(len(pool_kv_stride_size)):
            stride_kv[pool_kv_stride_size[i][0]] = pool_kv_stride_size[i][1:]
            if pool_kvq_kernel is not None:
                pool_kv[pool_kv_stride_size[i][0]] = pool_kvq_kernel
            else:
                pool_kv[pool_kv_stride_size[i][0]] = [
                    s + 1 if s > 1 else s for s in pool_kv_stride_size[i][1:]
                ]

    mvit_blocks = nn.ModuleList()
    for i in range(depth):
        num_heads = round_width(num_heads, head_mul[i], min_width=1, divisor=1)
        patch_embed_dim = round_width(patch_embed_dim, dim_mul[i], divisor=num_heads)
        dim_out = round_width(
            patch_embed_dim,
            dim_mul[i + 1],
            divisor=round_width(num_heads, head_mul[i + 1]),
        )

        mvit_blocks.append(
            MultiScaleBlock(
                dim=patch_embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate_block,
                droppath_rate=dpr[i],
                norm_layer=norm_layer,
                kernel_q=pool_q[i],
                kernel_kv=pool_kv[i],
                stride_q=stride_q[i],
                stride_kv=stride_kv[i],
                pool_mode=pooling_mode,
                has_cls_embed=cls_embed_on,
                pool_first=pool_first,
            )
        )

    embed_dim = dim_out
    norm_embed = norm_layer(embed_dim)
    if head is not None:
        head_model = head(
            in_features=embed_dim,
            out_features=head_num_classes,
            seq_pool_type="cls" if cls_embed_on else "mean",
            dropout_rate=head_dropout_rate,
            activation=head_activation,
        )
    else:
        head_model = None
    return MultiscaleVisionTransformers(
        patch_embed=patch_embed,
        cls_positional_encoding=cls_positional_encoding,
        pos_drop=pos_drop if dropout_rate_block > 0.0 else None,
        norm_patch_embed=norm_patch_embed,
        blocks=mvit_blocks,
        norm_embed=norm_embed,
        head=head_model,
    )

if __name__ == '__main__':
    model = create_multiscale_vision_transformers(spatial_size=7, temporal_size=3)
    print(model)
