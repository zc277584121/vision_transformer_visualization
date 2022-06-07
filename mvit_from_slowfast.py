import math
from functools import partial
from typing import Tuple, List

import numpy
import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.init import trunc_normal_
from layers_ours import *
from utils import show_cam


def find_most_h_w(all_layer_matrices):
    shape_info = torch.vstack(
        [torch.as_tensor([layer_matrix.shape[-2] for layer_matrix in all_layer_matrices]),
         torch.as_tensor([layer_matrix.shape[-1] for layer_matrix in all_layer_matrices])])
    return torch.mode(shape_info).values[0], torch.mode(shape_info).values[1]


def resize_last_dim_to_most(layer_matrix, last_dim_size):
    if layer_matrix.shape[-1] == last_dim_size:
        return layer_matrix
    else:
        cls_token, attn = layer_matrix[..., 1], layer_matrix[..., 1:]
        if attn.shape[-1] > last_dim_size - 1:
            assert attn.shape[-1] % (last_dim_size - 1) == 0
        else:
            assert (last_dim_size - 1) % attn.shape[-1] == 0
        factor = (last_dim_size - 1) / attn.shape[-1]
        attn = torch.nn.functional.interpolate(attn, size=last_dim_size - 1, mode='nearest')
        attn = attn * factor
        return torch.cat([cls_token.unsqueeze(dim=-1), attn], dim=-1)


def align_scale(all_layer_matrices):
    most_attn_h, most_attn_w = find_most_h_w(all_layer_matrices)
    print('most_attn_h', most_attn_h)
    print('most_attn_w', most_attn_w)
    aligned_layer_matrices = []
    for layer_matrix in all_layer_matrices:
        layer_matrix = resize_last_dim_to_most(layer_matrix, most_attn_w)
        layer_matrix = layer_matrix.permute(0, 2, 1)
        layer_matrix = resize_last_dim_to_most(layer_matrix, most_attn_h)
        layer_matrix = layer_matrix.permute(0, 2, 1)
        aligned_layer_matrices.append(layer_matrix)
    return aligned_layer_matrices


def compute_rollout_attention(all_layer_matrices, start_layer=0):
    all_layer_matrices = align_scale(all_layer_matrices)
    # filtered_layer_matrices = []
    # for layer_matrix in all_layer_matrices:
    #     if layer_matrix.shape == (1, 197, 197):
    #         filtered_layer_matrices.append(layer_matrix)
    # all_layer_matrices = filtered_layer_matrices.copy()

    # adding residual consideration
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    # all_layer_matrices = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
    #                       for i in range(len(all_layer_matrices))]
    joint_attention = all_layer_matrices[start_layer]
    for i in range(start_layer+1, len(all_layer_matrices)):
        joint_attention = all_layer_matrices[i].bmm(joint_attention)
    return joint_attention

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
        # thw_shape,
        pool=None,
        has_cls_embed=True,
        norm=None
    ) -> None:
        super().__init__()
        self.pool = pool
        # self.thw_shape = thw_shape
        self.has_cls_embed = has_cls_embed
        self.norm = norm
        self.tensor_dim = 4

    def forward(self, x: torch.Tensor, thw_shape) -> Tuple[torch.Tensor, List[int]]:
        if self.pool is None:
            # return x, self.thw_shape
            return x, thw_shape
        self.tensor_dim = x.ndim
        if self.tensor_dim == 4:
            pass
        elif self.tensor_dim == 3:  # For the case tensor_dim == 3. #旁路的add
            x = x.unsqueeze(1)
        else:
            raise NotImplementedError(f"Unsupported input dimension {x.shape}")

        if self.has_cls_embed:
            cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]

        b, n, _, c = x.shape
        # t, h, w = self.thw_shape
        t, h, w = thw_shape
        # 1, 1, 3136, 96
        # x = x.reshape(b * n, t, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        x = rearrange(x, 'b n (t h w) c -> (b n) c t h w', b=b, n=n, c=c, t=t, h=h, w=w)

        # 1, 96, 1, 56, 56
        x = self.pool(x)
        # 1, 96, 1, 14, 14
        out_thw_shape = [x.shape[2], x.shape[3], x.shape[4]] # [1, 14, 14]
        # l_pooled = x.shape[2] * x.shape[3] * x.shape[4] # 196
        # x = x.reshape(b, n, c, l_pooled).transpose(2, 3)
        x = rearrange(x, '(b n) c t h w  -> b n (t h w) c', b=b, n=n, c=c)
        # 1, 1, 196, 96
        if self.has_cls_embed:
            x = torch.cat((cls_tok, x), dim=2)
        if self.norm is not None:
            x = self.norm(x)

        if self.tensor_dim == 4:
            pass
        else:  # For the case tensor_dim == 3. #旁路的add
            x = x.squeeze(1)
        return x, out_thw_shape

    def relprop(self, cam, **kwargs):
        thw = kwargs.pop("thw")
        t, h, w = thw[0], thw[1], thw[2]
        if self.pool is None:
            return cam, thw
        if self.tensor_dim == 4:
            pass
        else:
            cam = cam.unsqueeze(1)
        if self.norm is not None:
            cam = self.norm.relprop(cam, **kwargs)
        if self.has_cls_embed:
            cls_tok, cam = cam[:, :, :1, :], cam[:, :, 1:, :]
        b, n, _, c = cam.shape
        cam = rearrange(cam, ' b n (t h w) c -> (b n) c t h w', t=t, h=h, w=w)
        cam = self.pool.relprop(cam, **kwargs)
        input_thw = cam.shape[-3:]
        cam = rearrange(cam, '(b n) c t h w -> b n (t h w) c', b=b, n=n, c=c)
        if self.has_cls_embed:
            cam = torch.cat((cls_tok, cam), dim=2)
        if self.tensor_dim == 4:
            pass
        elif self.tensor_dim == 3:  # For the case tensor_dim == 3. #旁路的add
            cam = cam.squeeze(1)
        else:
            raise NotImplementedError(f"Unsupported input dimension {cam.shape}")
        return cam, input_thw

# def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
#     if pool is None:
#         return tensor, thw_shape
#     tensor_dim = tensor.ndim
#     if tensor_dim == 4:
#         pass
#     elif tensor_dim == 3:
#         tensor = tensor.unsqueeze(1)
#     else:
#         raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")
#
#     if has_cls_embed:
#         cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]
#
#     B, N, L, C = tensor.shape
#     T, H, W = thw_shape
#     tensor = (
#         tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
#     )
#
#     tensor = pool(tensor)
#
#     thw_shape = [tensor.shape[2], tensor.shape[3], tensor.shape[4]]
#     L_pooled = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
#     tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
#     if has_cls_embed:
#         tensor = torch.cat((cls_tok, tensor), dim=2)
#     if norm is not None:
#         tensor = norm(tensor)
#     # Assert tensor_dim in [3, 4]
#     if tensor_dim == 4:
#         pass
#     else:  # tensor_dim == 3:
#         tensor = tensor.squeeze(1)
#     return tensor, thw_shape


def round_width(width, multiplier, min_width=1, divisor=1, verbose=False):
    if not multiplier:
        return width
    width *= multiplier
    min_width = min_width or divisor
    # if verbose:
    #     logger.info(f"min width {min_width}")
    #     logger.info(f"width {width} divisor {divisor}")
    #     logger.info(f"other {int(width + divisor / 2) // divisor * divisor}")

    width_out = max(min_width, int(width + divisor / 2) // divisor * divisor)
    if width_out < 0.9 * width:
        width_out += divisor
    return int(width_out)


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=GELU,
            drop_rate=0.0,
    ):
        super().__init__()
        self.drop_rate = drop_rate
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        if self.drop_rate > 0.0:
            self.drop = Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        x = self.fc2(x)
        if self.drop_rate > 0.0:
            x = self.drop(x)
        return x


    def relprop(self, cam, **kwargs):
        if self.drop_rate > 0.0:
            cam = self.drop.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        if self.drop_rate > 0.0:
            cam = self.drop.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
        return cam


class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """
    PatchEmbed.
    """

    def __init__(
            self,
            dim_in=3,
            dim_out=768,
            kernel=(1, 16, 16),
            stride=(1, 4, 4),
            padding=(1, 7, 7),
            conv_2d=False,
    ):
        super().__init__()
        self.conv_2d = conv_2d
        if self.conv_2d:
            conv = Conv2d
        else:
            conv = Conv3d
        self.proj = conv(
            dim_in,
            dim_out,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
        )
        self.output_t = 1
        self.output_h = None
        self.output_w = None

    def forward(self, x):
        x = self.proj(x)
        if self.conv_2d:
            self.output_h, self.output_w = x.shape[-2:]
        else:
            self.output_t, self.output_h, self.output_w = x.shape[-3:]
        # B C (T) H W -> B (T)HW C
        return x.flatten(2).transpose(1, 2)

    def relprop(self, cam, **kwargs):
        cam = cam.transpose(1, 2)
        # B C (T)HW
        if self.conv_2d:
            cam = cam.reshape(cam.shape[0], cam.shape[1], self.output_h, self.output_w)
        else:
            cam = cam.reshape(cam.shape[0], cam.shape[1], self.output_t, self.output_h, self.output_w)
        # B C (T) H W
        return self.proj.relprop(cam, **kwargs)


class TransformerBasicHead(nn.Module):
    """
    BasicHead. No pool.
    """

    def __init__(
            self,
            dim_in,
            num_classes,
            dropout_rate=0.0,
            act_func="softmax",
    ):
        """
        Perform linear projection and activation as head for tranformers.
        Args:
            dim_in (int): the channel dimension of the input to the head.
            num_classes (int): the channel dimensions of the output to the head.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(TransformerBasicHead, self).__init__()
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.projection = Linear(dim_in, num_classes, bias=True)

        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = Softmax(dim=1)
        elif act_func == "sigmoid":
            self.act = Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, x):
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            x = self.act(x)
        return x

    def relprop(self, cam, **kwargs):
        if not self.training:
            cam = self.act.relprop(cam, **kwargs)
        cam = self.projection.relprop(cam, **kwargs)
        if hasattr(self, "dropout"):
            cam = self.dropout.relprop(cam, **kwargs)
        return cam

class MultiScaleAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            drop_rate=0.0,
            kernel_q=(1, 1, 1),
            kernel_kv=(1, 1, 1),
            stride_q=(1, 1, 1),
            stride_kv=(1, 1, 1),
            norm_layer=LayerNorm,
            has_cls_embed=True,
            # Options include `conv`, `avg`, and `max`.
            mode="conv",
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = Linear(dim, dim)
        # self.show_attn = nn.Softmax(dim=-1)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # A = Q*K^T
        self.matmul1 = einsum('bhid,bhjd->bhij')
        # attn = A*V
        self.matmul2 = einsum('bhij,bhjd->bhid')

        self.softmax = Softmax(dim=-1)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if numpy.prod(kernel_q) == 1 and numpy.prod(stride_q) == 1:
            kernel_q = ()
        if numpy.prod(kernel_kv) == 1 and numpy.prod(stride_kv) == 1:
            kernel_kv = ()
        pool_q = pool_k = pool_v = norm_q = norm_k = norm_v = None
        if mode == "avg":
            pool_q = (
                nn.AvgPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            pool_k = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            pool_v = (
                nn.AvgPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "max":
            pool_q = (
                nn.MaxPool3d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            pool_k = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            pool_v = (
                nn.MaxPool3d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv":
            pool_q = (
                Conv3d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None
            pool_k = (
                Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None
            pool_v = (
                Conv3d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.atn_pool_q = AttentionPool(
            pool=pool_q,
            # thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=norm_q,
        )
        self.atn_pool_k = AttentionPool(
            pool=pool_k,
            # thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=norm_k,
        )
        self.atn_pool_v = AttentionPool(
            pool=pool_v,
            # thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=norm_v,
        )
        self.attn_cam = None
        self.attn = None
        self.v = None
        self.v_cam = None
        self.attn_gradients = None

    def get_attn(self):
        return self.attn

    def save_attn(self, attn):
        self.attn = attn

    def save_attn_cam(self, cam):
        self.attn_cam = cam

    def get_attn_cam(self):
        return self.attn_cam

    def get_v(self):
        return self.v

    def save_v(self, v):
        self.v = v

    def save_v_cam(self, cam):
        self.v_cam = cam

    def get_v_cam(self):
        return self.v_cam

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def forward(self, x, thw_shape):
        B, N, C = x.shape
        qkv = self.qkv(x)
        # qkv = (
        #     self.qkv(x)
        #         .reshape(B, N, 3, self.num_heads, C // self.num_heads)
        #         .permute(2, 0, 3, 1, 4)
        # )
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.num_heads)

        self.save_v(v) #todo: 放前面不是后面？
        # q, k, v = qkv[0], qkv[1], qkv[2]
        q, q_shape = self.atn_pool_q(q, thw_shape)
        k, k_shape = self.atn_pool_k(k, thw_shape)
        v, v_shape = self.atn_pool_v(v, thw_shape)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        dots = self.matmul1([q, k]) * self.scale
        attn = self.softmax(dots)

        self.save_attn(attn)
        # attn.register_hook(self.save_attn_gradients)# todo:ppp

        # N = q.shape[2]
        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.matmul2([attn, v])
        print('\nattn.shape =', attn.shape)
        print('v.shape =', v.shape)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, q_shape

    def relprop(self, cam, **kwargs):
        thw = kwargs.pop("thw")
        if self.drop_rate > 0.0:
            cam = self.proj_drop.relprop(cam, **kwargs)
        cam = self.proj.relprop(cam, **kwargs)
        cam = rearrange(cam, 'b n (h d) -> b h n d', h=self.num_heads)

        # attn = A*V
        (cam1, cam_v) = self.matmul2.relprop(cam, **kwargs)
        cam1 /= 2
        cam_v /= 2

        self.save_v_cam(cam_v)
        self.save_attn_cam(cam1)

        # cam1 = self.attn_drop.relprop(cam1, **kwargs)
        cam1 = self.softmax.relprop(cam1, **kwargs)

        # A = Q*K^T
        (cam_q, cam_k) = self.matmul1.relprop(cam1, **kwargs)
        cam_q /= 2
        cam_k /= 2

        t = thw[0]
        kv_h = kv_w = int(math.sqrt(cam_k.shape[-2] - 1))
        #todo: q,k,v relprop
        cam_q, input_thw = self.atn_pool_q.relprop(cam_q, thw=thw, **kwargs)
        cam_k, input_thw_k = self.atn_pool_k.relprop(cam_k, thw=[t, kv_h, kv_w], **kwargs)
        cam_v, input_thw_v = self.atn_pool_v.relprop(cam_v, thw=[t, kv_h, kv_w], **kwargs)

        cam_qkv = rearrange([cam_q, cam_k, cam_v], 'qkv b h n d -> b n (qkv h d)', qkv=3, h=self.num_heads)

        return self.qkv.relprop(cam_qkv, **kwargs), input_thw


class MultiScaleBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_out,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            drop_path=0.0,
            act_layer=GELU,
            norm_layer=LayerNorm,
            up_rate=None,
            kernel_q=(1, 1, 1),
            kernel_kv=(1, 1, 1),
            kernel_skip=(1, 1, 1),
            stride_q=(1, 1, 1),
            stride_kv=(1, 1, 1),
            stride_skip=(1, 1, 1),
            mode="conv",
            has_cls_embed=True,
    ):
        super().__init__()
        self.add1 = Add()
        self.add2 = Add()
        self.clone1 = Clone()
        self.clone2 = Clone()
        self.clone_norm = Clone()

        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
        )
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop_rate=drop_rate,
        )
        if dim != dim_out:
            self.proj = Linear(dim, dim_out)

        self.pool_skip = (
            MaxPool3d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(kernel_skip) > 0
            else None
        )
        self.atn_pool = AttentionPool(
            pool=self.pool_skip,
            # thw_shape=thw_shape,
            has_cls_embed=self.has_cls_embed
        )

    def forward(self, x, thw_shape):

        x1, x2 = self.clone1(x, 2)
        x2, thw_shape_new = self.attn(self.norm1(x2), thw_shape)

        # atn_pool = AttentionPool(
        #     pool=self.pool_skip,
        #     # thw_shape=thw_shape,
        #     has_cls_embed=self.has_cls_embed
        # )
        x1, _ = self.atn_pool(x1, thw_shape)
        # x1, _ = attention_pool(
        #     x1, self.pool_skip, thw_shape, has_cls_embed=self.has_cls_embed
        # )

        x = self.add1([x1, self.drop_path(x2)])

        ############## original ##############
        # x1, x2 = self.clone2(x, 2)
        # x_norm = self.norm2(x2)
        # x2 = self.mlp(x_norm)
        # if self.dim != self.dim_out:
        #     x1 = self.proj(x_norm)
        # x = self.add2([x1, self.drop_path(x2)])
        ######################################

        if self.dim != self.dim_out:
            # _, x = self.clone2(x, 2)
            x_norm = self.norm2(x)
            x_norm2, x_norm1 = self.clone_norm(x_norm, 2)
            x2 = self.mlp(x_norm2)
            x1 = self.proj(x_norm1)
            x = self.add2([x1, self.drop_path(x2)])
        else:
            x1, x2 = self.clone2(x, 2)
            x_norm = self.norm2(x2)
            x2 = self.mlp(x_norm)
            x = self.add2([x1, self.drop_path(x2)])

        return x, thw_shape_new





    def relprop(self, cam, **kwargs):
        thw = kwargs.pop("thw")
        # (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        # # todo: mlp -> drop path
        # cam_norm = self.mlp.relprop(cam2, **kwargs)
        # if self.dim != self.dim_out:
        #     cam_norm = self.proj.relprop(cam1, **kwargs)
        # cam2 = self.norm2.relprop(cam_norm, **kwargs)
        # cam = self.clone2.relprop((cam1, cam2), **kwargs)
        if self.dim != self.dim_out:
            (cam1, cam2) = self.add2.relprop(cam, **kwargs)
            # todo:  cam2 drop path relprop
            cam_norm1 = self.proj.relprop(cam1, **kwargs)
            cam_norm2 = self.mlp.relprop(cam2, **kwargs)
            cam_norm = self.clone_norm.relprop((cam_norm2, cam_norm1), **kwargs)
            cam = self.norm2.relprop(cam_norm, **kwargs)
        else:
            (cam1, cam2) = self.add2.relprop(cam, **kwargs)
            # todo:  cam2 drop path relprop
            cam_norm = self.mlp.relprop(cam2, **kwargs)
            cam2 = self.norm2.relprop(cam_norm, **kwargs)
            cam = self.clone2.relprop((cam1, cam2), **kwargs)

        #
        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        # todo: mlp -> drop path
        cam1, _ = self.atn_pool.relprop(cam1, thw=thw, **kwargs)
        cam2, input_thw = self.attn.relprop(cam2, thw=thw, **kwargs) #todo:thw
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam, input_thw


class MViT(nn.Module):
    """
    Multiscale Vision Transformers
    Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, Jitendra Malik, Christoph Feichtenhofer
    https://arxiv.org/abs/2104.11227
    """

    def __init__(self, cfg):
        super().__init__()
        # Get parameters.
        assert cfg.DATA.TRAIN_CROP_SIZE == cfg.DATA.TEST_CROP_SIZE
        self.cfg = cfg
        # Prepare input.
        spatial_size = cfg.DATA.TRAIN_CROP_SIZE
        temporal_size = cfg.DATA.NUM_FRAMES
        in_chans = cfg.DATA.INPUT_CHANNEL_NUM[0]
        use_2d_patch = cfg.MVIT.PATCH_2D
        self.patch_stride = cfg.MVIT.PATCH_STRIDE
        if use_2d_patch:
            self.patch_stride = [1] + self.patch_stride
        # Prepare output.
        num_classes = cfg.MODEL.NUM_CLASSES
        embed_dim = cfg.MVIT.EMBED_DIM
        # Prepare backbone
        num_heads = cfg.MVIT.NUM_HEADS
        mlp_ratio = cfg.MVIT.MLP_RATIO
        qkv_bias = cfg.MVIT.QKV_BIAS
        self.drop_rate = cfg.MVIT.DROPOUT_RATE
        depth = cfg.MVIT.DEPTH
        drop_path_rate = cfg.MVIT.DROPPATH_RATE
        mode = cfg.MVIT.MODE
        self.cls_embed_on = cfg.MVIT.CLS_EMBED_ON
        self.sep_pos_embed = cfg.MVIT.SEP_POS_EMBED
        if cfg.MVIT.NORM == "layernorm":
            norm_layer = partial(LayerNorm, eps=1e-6)
        else:
            raise NotImplementedError("Only supports layernorm.")
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.MVIT.PATCH_KERNEL,
            stride=cfg.MVIT.PATCH_STRIDE,
            padding=cfg.MVIT.PATCH_PADDING,
            conv_2d=use_2d_patch,
        )
        self.input_dims = [temporal_size, spatial_size, spatial_size]
        assert self.input_dims[1] == self.input_dims[2]
        self.patch_dims = [
            self.input_dims[i] // self.patch_stride[i]
            for i in range(len(self.input_dims))
        ]
        num_patches = math.prod(self.patch_dims)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        if self.cls_embed_on:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            pos_embed_dim = num_patches + 1
        else:
            pos_embed_dim = num_patches

        if self.sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(
                    1, self.patch_dims[1] * self.patch_dims[2], embed_dim
                )
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, self.patch_dims[0], embed_dim)
            )
            self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, pos_embed_dim, embed_dim)
            )
        self.add = Add()
        if self.drop_rate > 0.0:
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        pool_q = cfg.MVIT.POOL_Q_KERNEL
        pool_kv = cfg.MVIT.POOL_KV_KERNEL
        pool_skip = cfg.MVIT.POOL_SKIP_KERNEL
        stride_q = cfg.MVIT.POOL_Q_STRIDE
        stride_kv = cfg.MVIT.POOL_KV_STRIDE
        stride_skip = cfg.MVIT.POOL_SKIP_STRIDE

        dim_mul, head_mul = torch.ones(depth + 1), torch.ones(depth + 1)

        if len(cfg.MVIT.DIM_MUL) > 1:
            for k in cfg.MVIT.DIM_MUL:
                dim_mul[k[0]] = k[1]
        if len(cfg.MVIT.HEAD_MUL) > 1:
            for k in cfg.MVIT.HEAD_MUL:
                head_mul[k[0]] = k[1]

        self.norm_stem = norm_layer(embed_dim) if cfg.MVIT.NORM_STEM else None

        self.blocks = nn.ModuleList()
        for i in range(depth):
            print('\ndepth:', i)
            num_heads = round_width(num_heads, head_mul[i])
            embed_dim = round_width(embed_dim, dim_mul[i], divisor=num_heads)
            dim_out = round_width(
                embed_dim,
                dim_mul[i + 1],
                divisor=round_width(num_heads, head_mul[i + 1]),
            )
            print('embed_dim:', embed_dim)
            print('dim_out:', dim_out)
            self.blocks.append(
                MultiScaleBlock(
                    dim=embed_dim,
                    dim_out=dim_out,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_rate=self.drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    kernel_q=pool_q[i] if len(pool_q) > i else [],
                    kernel_kv=pool_kv[i] if len(pool_kv) > i else [],
                    kernel_skip=pool_skip[i] if len(pool_skip) > i else [],
                    stride_q=stride_q[i] if len(stride_q) > i else [],
                    stride_kv=stride_kv[i] if len(stride_kv) > i else [],
                    stride_skip=stride_skip[i] if len(stride_skip) > i else [],
                    mode=mode,
                    has_cls_embed=self.cls_embed_on,
                )
            )

        embed_dim = dim_out
        self.norm = norm_layer(embed_dim)

        self.index_select = IndexSelect()

        self.head = TransformerBasicHead(
            embed_dim,
            num_classes,
            dropout_rate=cfg.MODEL.DROPOUT_RATE,
            act_func=cfg.MODEL.HEAD_ACT,
        )
        if self.sep_pos_embed:
            trunc_normal_(self.pos_embed_spatial, std=0.02)
            trunc_normal_(self.pos_embed_temporal, std=0.02)
            trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_embed_on:
            trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.out_thw = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        if self.cfg.MVIT.ZERO_DECAY_POS_CLS:
            if self.sep_pos_embed:
                if self.cls_embed_on:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                        "cls_token",
                    }
                else:
                    return {
                        "pos_embed_spatial",
                        "pos_embed_temporal",
                        "pos_embed_class",
                    }
            else:
                if self.cls_embed_on:
                    return {"pos_embed", "cls_token"}
                else:
                    return {"pos_embed"}
        else:
            return {}

    def forward(self, x):
        # x = x[0]
        x = self.patch_embed(x)

        T = self.cfg.DATA.NUM_FRAMES // self.patch_stride[0]
        H = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[1]
        W = self.cfg.DATA.TRAIN_CROP_SIZE // self.patch_stride[2]
        B, N, C = x.shape

        if self.cls_embed_on:
            cls_tokens = self.cls_token.expand(
                B, -1, -1
            )  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.patch_dims[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.patch_dims[1] * self.patch_dims[2],
                dim=1,
            )
            pos_embed_cls = torch.cat([self.pos_embed_class, pos_embed], 1)
            # x = x + pos_embed_cls
            x = self.add([x, pos_embed_cls])
        else:
            # x = x + self.pos_embed
            x = self.add([x, self.pos_embed])

        if self.drop_rate:
            x = self.pos_drop(x)

        if self.norm_stem:
            x = self.norm_stem(x)

        thw = [T, H, W]
        for i, blk in enumerate(self.blocks):
            x, thw = blk(x, thw)
        self.out_thw = thw
        x = self.norm(x)
        if self.cls_embed_on:
            # x = x[:, 0]
            x = self.index_select(x, dim=1, indices=torch.tensor(0, device=x.device))
            x = x.squeeze(1)
        else:
            x = x.mean(1)

        x = self.head(x)
        return x

    def relprop(self, cam, method="full", **kwargs):
        print('input cam shape=', cam.shape)
        cam = self.head.relprop(cam, **kwargs)
        print('after self.head.relprop(cam, **kwargs), shape=', cam.shape)
        if self.cls_embed_on:
            cam = cam.unsqueeze(1)
            print('after cam.unsqueeze(1) shape=', cam.shape)
            cam = self.index_select.relprop(cam, **kwargs)
            print('after self.index_select.relprop(cam, **kwargs), shape=', cam.shape)
        cam = self.norm.relprop(cam, **kwargs)
        print('after self.norm.relprop(cam, **kwargs), shape=', cam.shape)
        thw = self.out_thw
        for i, blk in enumerate(reversed(self.blocks)):
            print('blk = -', i)
            cam, thw = blk.relprop(cam, thw=thw, **kwargs)
            print('after self.blk.relprop(cam, **kwargs), shape=', cam.shape)

        # print(cam.shape)
        # cam = cam.unsqueeze(1)
        # cam = self.pool.relprop(cam, **kwargs)
        # cam = self.norm.relprop(cam, **kwargs)
        # for blk in reversed(self.blocks):
        #     cam = blk.relprop(cam, **kwargs)

        if method == 'full':
            (cam, _) = self.add.relprop(cam, **kwargs)
            if self.cls_embed_on:
                cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam
        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=0)
            cam = cam[:, 0, 1:]
            return cam
        elif method == "reverse_rollout":
            # cam rollout
            attn_cams = []
            for blk in reversed(self.blocks):
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=0)
            cam = cam[:, 0, 1:]
            return cam

        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                # cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                cam = cam[0].reshape(-1, cam.shape[-2], cam.shape[-1])
                # grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-2], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0) #channel-wise mean
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=0)
            cam = rollout[:, 0, 1:]
            return cam

        elif method == 'last_layer':
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            # if is_ablation:
            #     grad = self.blocks[-1].attn.get_attn_gradients()
            #     grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
            #     cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam
        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam