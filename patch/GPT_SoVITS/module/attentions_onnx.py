import math
import torch
from torch import nn
from torch.nn import functional as F
from module import commons
from module.modules import LayerNorm
from typing import Optional

class Encoder(nn.Module):
    """Transformer encoder with multi-head attention and feed-forward layers."""
    def __init__(
        self,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int = 1,
        p_dropout: float = 0.0,
        window_size: int = 4,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.window_size = window_size

        self.attn_layers = nn.ModuleList([
            MultiHeadAttention(
                hidden_channels,
                hidden_channels,
                n_heads,
                p_dropout=p_dropout,
                window_size=window_size,
            ) for _ in range(n_layers)
        ])
        self.norm_layers_1 = nn.ModuleList([LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.ffn_layers = nn.ModuleList([
            FFN(
                hidden_channels,
                hidden_channels,
                filter_channels,
                kernel_size,
                p_dropout=p_dropout,
            ) for _ in range(n_layers)
        ])
        self.norm_layers_2 = nn.ModuleList([LayerNorm(hidden_channels) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_mask = None
        if x_mask is not None:
            attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
            x = x * x_mask
        for attn, norm1, ffn, norm2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            residual = x
            x = norm1(residual + attn(x, x, attn_mask))
            residual = x
            x = norm2(residual + ffn(x, x_mask))
        return x if x_mask is None else x * x_mask

class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional relative positional embeddings."""
    def __init__(
        self,
        channels: int,
        out_channels: int,
        n_heads: int,
        p_dropout: float = 0.0,
        window_size: Optional[int] = None,
        proximal_init: bool = False,
    ):
        super().__init__()
        assert channels % n_heads == 0, f"Channels ({channels}) must be divisible by n_heads ({n_heads})"

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.k_channels = channels // n_heads
        self.window_size = window_size

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        self.register_buffer("packed_weight", None)
        self.register_buffer("packed_bias", None)

        if window_size is not None:
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(1, window_size * 2 + 1, self.k_channels) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def pack_projections_for_export(self):
        # Relative attention is self-attention in these encoders. MRTE shares only K/V.
        layers = (self.conv_k, self.conv_v)
        if self.window_size is not None:
            layers = (self.conv_q, *layers)
        self.packed_weight = torch.cat([layer.weight.detach() for layer in layers], dim=0)
        self.packed_bias = torch.cat([layer.bias.detach() for layer in layers], dim=0)

    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.packed_weight is None:
            q, k, v = self.conv_q(x), self.conv_k(c), self.conv_v(c)
        elif self.window_size is not None:
            q, k, v = F.conv1d(x, self.packed_weight, self.packed_bias).split(self.channels, dim=1)
        else:
            q = self.conv_q(x)
            k, v = F.conv1d(c, self.packed_weight, self.packed_bias).split(self.channels, dim=1)
        x, _ = self.attention(q, k, v, attn_mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask: Optional[torch.Tensor] = None):
        b, d, t_s  = (*key.size(),)
        # Reshape directly to avoid multiple view operations
        query = query.view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, -1).transpose(2, 3)

        query = query * (self.k_channels ** -0.5)
        scores = torch.matmul(query, key.transpose(-2, -1))

        if self.window_size is not None:
            positions = torch.arange(t_s, device=query.device)
            offsets = torch.arange(-self.window_size, self.window_size + 1, device=query.device)
            neighbors = positions[:, None] + offsets
            valid = (neighbors >= 0) & (neighbors < t_s)
            neighbors = neighbors.clamp(0, t_s - 1).expand(b, self.n_heads, -1, -1)
            # Each relative embedding contributes only inside its fixed window.
            relative_logits = query @ self.emb_rel_k.unsqueeze(0).transpose(-2, -1)
            scores = scores.scatter_add(-1, neighbors, relative_logits.masked_fill(~valid, 0))

        if mask is not None:
            scores.masked_fill_(mask == 0, -1e4)  # In-place operation

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        if self.window_size is not None:
            relative_weights = p_attn.gather(-1, neighbors).masked_fill(~valid, 0)
            output = output + relative_weights @ self.emb_rel_v.unsqueeze(0)

        output = output.transpose(2, 3).contiguous().view(b, d, -1)
        return output, p_attn

class FFN(nn.Module):
    """Feed-forward network with convolutional layers."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float = 0.0,
        activation: str = "relu",
        causal: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.activation = activation

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=self._get_padding())
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=self._get_padding())
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x: torch.Tensor, x_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if x_mask is not None:
            x = x * x_mask
        x = self.conv_1(x)
        x = F.gelu(x) if self.activation == "gelu" else torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x)
        return x if x_mask is None else x * x_mask

    def _get_padding(self) -> int:
        if self.kernel_size == 1:
            return 0
        return (self.kernel_size - 1) // 2 if not self.causal else self.kernel_size - 1
class MRTE(nn.Module):
    def __init__(
        self,
        content_enc_channels: int = 192,
        hidden_size: int = 512,
        out_channels: int = 192,
        n_heads: int = 4,
    ):
        super().__init__()
        self.cross_attention = MultiHeadAttention(hidden_size, hidden_size, n_heads)
        self.c_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.text_pre = nn.Conv1d(content_enc_channels, hidden_size, 1)
        self.c_post = nn.Conv1d(hidden_size, out_channels, 1)

    def forward(self, ssl_enc: torch.Tensor, ssl_mask: Optional[torch.Tensor], text: torch.Tensor, text_mask: Optional[torch.Tensor], ge: torch.Tensor) -> torch.Tensor:
        if ssl_mask is None and text_mask is None:
            ssl_enc = self.c_pre(ssl_enc)
            text_enc = self.text_pre(text)
            return self.c_post(self.cross_attention(ssl_enc, text_enc) + ssl_enc + ge)
        attn_mask = text_mask.unsqueeze(2) * ssl_mask.unsqueeze(-1)
        ssl_enc = self.c_pre(ssl_enc * ssl_mask)
        text_enc = self.text_pre(text * text_mask)
        x = self.cross_attention(ssl_enc * ssl_mask, text_enc * text_mask, attn_mask) + ssl_enc + ge
        return self.c_post(x * ssl_mask)
