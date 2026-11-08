# Modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/activation.py
from typing import Optional, Tuple, List

import torch
from torch.onnx.symbolic_helper import _optional_input_placeholder_tensor, parse_args
from torch import Tensor
from torch.nn import Linear, Module
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.parameter import Parameter

from torch.nn import functional as F

from AR.modules.patched_mha_with_cache_onnx import multi_head_attention_forward_patched


class SharedKVAttention(torch.autograd.Function):
    """CPU ONNX attention with aliased, head-major past and present caches."""

    @staticmethod
    def forward(ctx, qkv, key, value, past_length, total_length, heads, head_dim):
        q, k, v = qkv.reshape(1, 1, 3, heads, head_dim).unbind(2)
        q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        past = past_length[0]
        key = torch.cat((key[:, :, :past], k, key[:, :, past + 1:]), dim=2)
        value = torch.cat((value[:, :, :past], v, value[:, :, past + 1:]), dim=2)
        attended = F.scaled_dot_product_attention(q, key[:, :, :past + 1], value[:, :, :past + 1])
        return attended.transpose(1, 2).reshape(1, 1, heads * head_dim), key, value

    @staticmethod
    @parse_args("v", "v", "v", "v", "v", "i", "i")
    def symbolic(graph, qkv, key, value, past_length, total_length, heads, head_dim):
        unused = _optional_input_placeholder_tensor(graph)
        output, present_key, present_value = graph.op(
            "com.microsoft::GroupQueryAttention", qkv, unused, unused, key, value,
            past_length, total_length, num_heads_i=heads, kv_num_heads_i=heads,
            outputs=3,
        )
        return (
            output.setType(qkv.type().with_sizes([1, 1, heads * head_dim])),
            present_key.setType(key.type()),
            present_value.setType(value.type()),
        )


class MultiheadAttention(Module):
    __constants__ = ["batch_first"]
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        linear1_cls=Linear,
        linear2_cls=Linear,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * \
            num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(
                (1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty(
                (1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        if linear1_cls == Linear:
            if not self._qkv_same_embed_dim:
                self.q_proj_weight = Parameter(
                    torch.empty((embed_dim, embed_dim), **factory_kwargs)
                )
                self.k_proj_weight = Parameter(
                    torch.empty((embed_dim, self.kdim), **factory_kwargs)
                )
                self.v_proj_weight = Parameter(
                    torch.empty((embed_dim, self.vdim), **factory_kwargs)
                )
                self.register_parameter("in_proj_weight", None)
            else:
                self.in_proj_weight = Parameter(
                    torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
                )
                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

            if bias:
                self.in_proj_bias = Parameter(
                    torch.empty(3 * embed_dim, **factory_kwargs))
            else:
                self.register_parameter("in_proj_bias", None)
            self.out_proj = NonDynamicallyQuantizableLinear(
                embed_dim, embed_dim, bias=bias, **factory_kwargs)

            self._reset_parameters()
        else:
            if not self._qkv_same_embed_dim:
                raise NotImplementedError
            else:
                self.in_proj_linear = linear1_cls(
                    embed_dim,
                    3 * embed_dim,
                    bias=bias,
                    **factory_kwargs,
                )
                self.in_proj_weight = self.in_proj_linear.weight

                self.register_parameter("q_proj_weight", None)
                self.register_parameter("k_proj_weight", None)
                self.register_parameter("v_proj_weight", None)

                if bias:
                    self.in_proj_bias = self.in_proj_linear.bias
                else:
                    self.register_parameter("in_proj_bias", None)

            self.out_proj = linear2_cls(
                embed_dim,
                embed_dim,
                bias=bias,
                **factory_kwargs,
            )

            if self.bias_k is not None:
                xavier_normal_(self.bias_k)
            if self.bias_v is not None:
                xavier_normal_(self.bias_v)

        self.add_zero_attn = add_zero_attn

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        k_cache: Optional[Tensor] = None,
        v_cache: Optional[Tensor] = None,
        first_infer: bool = True,
        cache_lengths: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, Tensor]:
        batch_size = 1 # ONLY SUPPORT 1 NOW
        projected = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        if cache_lengths is not None:
            attn, key, value = SharedKVAttention.apply(
                projected.unsqueeze(1), k_cache, v_cache, *cache_lengths,
                self.num_heads, self.head_dim,
            )
            return F.linear(attn.squeeze(1), self.out_proj.weight, self.out_proj.bias), key, value
        q, k, v = projected.split(self.embed_dim, dim=-1)

        # Update key and value with cache
        if k_cache is not None and v_cache is not None and not first_infer:
            k_cache = torch.cat([k_cache, k], dim=1)
            v_cache = torch.cat([v_cache, v], dim=1)
        else:
            k_cache = k
            v_cache = v
        q_len = q.shape[1]
        kv_len = k_cache.shape[1]
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_cache.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_cache.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        if first_infer:
            attn = F.scaled_dot_product_attention(q, k, v,  (~attn_mask))
        else:
            attn = F.scaled_dot_product_attention(q, k, v)

        attn = attn.transpose(1, 2).reshape(batch_size, q_len, -1)
        attn = F.linear(attn, self.out_proj.weight, self.out_proj.bias)

        return attn, k_cache, v_cache
