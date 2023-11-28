import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from collections import namedtuple

from einops import rearrange, repeat

from entmax import entmax15

# constants

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])


# helpers

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


# classes


# attention.

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=DEFAULT_DIM_HEAD,
            heads=8,
            causal=False,
            mask=None,
            talking_heads=False,
            collab_heads=False,
            collab_compression=.3,
            sparse_topk=None,
            use_entmax15=False,
            num_mem_kv=0,
            dropout=0.,
            on_attn=False,
            gate_values=False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.causal = causal
        self.mask = mask

        qk_dim = v_dim = dim_head * heads

        # collaborative heads
        self.collab_heads = collab_heads
        if self.collab_heads:
            qk_dim = int(collab_compression * qk_dim)
            self.collab_mixing = nn.Parameter(torch.randn(heads, qk_dim))

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, v_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, v_dim)
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 1)

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        self.attn_fn = entmax15 if use_entmax15 else F.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(v_dim, dim * 2), nn.GLU()) if on_attn else nn.Linear(v_dim, dim)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            rel_pos=None,
            sinusoidal_emb=None,
            rotary_pos_emb=None,
            prev_attn=None,
            mem=None
    ):
        b, n, _, h, talking_heads, collab_heads, device, has_context = *x.shape, self.heads, self.talking_heads, self.collab_heads, x.device, exists(
            context)
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        if not collab_heads:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        else:
            q = torch.einsum('b i d, h d -> b h i d', q, self.collab_mixing)
            k = rearrange(k, 'b n d -> b () n d')
            v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        if exists(rotary_pos_emb) and not has_context:
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
            ql, kl, vl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl))
            q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device=device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)

        if collab_heads:
            k = k.expand(-1, h, -1, -1)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots.clone()

        if talking_heads:
            dots = einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj).contiguous()

        if exists(rel_pos):
            dots = rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if self.causal:
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j - i, 0), value=False)
            dots.masked_fill_(mask, mask_value)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        if talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj).contiguous()

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if exists(self.to_v_gate):
            gates = self.gate_v(x)
            out = out * gates.sigmoid()

        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn
        )

        return self.to_out(out), intermediates
