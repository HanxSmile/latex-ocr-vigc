""" PyTorch LLaMA model."""
import math
from typing import Optional, Tuple

import torch
# from apex.normalization.fused_layer_norm import MixedFusedRMSNorm as LlamaRMSNorm
import rotary_emb
import torch.utils.checkpoint
from einops import rearrange
from flash_attn.layers.rotary import ApplyRotaryEmbQKV_ as LegacyApplyRotaryEmbQKV_
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.utils import logging

# from flash_attn.modules.mha import FlashSelfAttention

logger = logging.get_logger(__name__)

from vigc.models.intern_models.lora import LoRALinear


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    """
    ApplyRotaryEmbQKV_
    """

    @staticmethod
    def forward(ctx, qkv, cos, sin, cos_k=None, sin_k=None):
        """
            qkv: (total, 3, nheads, headdim)
            cos, sin: (seqlen, rotary_dim / 2)
            cos_k, sin_k: (seqlen, rotary_dim / 2), optional
        rotary_dim must be <= headdim
        Apply rotary embedding *inplace* to the first rotary_dim of q and k.
        """
        _, three, _, headdim = qkv.shape
        assert three == 3
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        assert sin.shape == cos_k.shape == sin_k.shape == (rotary_seqlen,
                                                           rotary_dim // 2)
        q1, q2 = qkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(q1, q2, rearrange(cos, "s d -> s 1 d"),
                                rearrange(sin, "s d -> s 1 d"), q1, q2, False)
        k1, k2 = qkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(k1, k2, rearrange(cos_k, "s d -> s 1 d"),
                                rearrange(sin_k, "s d -> s 1 d"), k1, k2,
                                False)
        ctx.save_for_backward(cos, sin, cos_k, sin_k)
        return qkv

    @staticmethod
    def backward(ctx, dqkv):
        cos, sin, cos_k, sin_k = ctx.saved_tensors
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        dq1, dq2 = dqkv[:, 0, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(dq1, dq2, rearrange(cos, "s d -> s 1 d"),
                                rearrange(sin, "s d -> s 1 d"), dq1, dq2, True)
        dk1, dk2 = dqkv[:, 1, :, :rotary_dim].chunk(2, dim=-1)
        rotary_emb.apply_rotary(dk1, dk2, rearrange(cos_k, "s d -> s 1 d"),
                                rearrange(sin_k, "s d -> s 1 d"), dk1, dk2,
                                True)
        return dqkv, None, None, None, None


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, base=10000, scale_base=0, device=None):
        """ """
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (
                torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.scale_base = scale_base
        scale = ((torch.arange(0, dim, 2, device=device, dtype=torch.float32) +
                  0.4 * dim) / (1.4 * dim) if scale_base > 0 else None)
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _update_cos_sin_cache(self, x, indexes):
        """x: (batch, seqlen, nheads, headdim) or (batch, seqlen, 3, nheads, headdim)"""
        if not isinstance(indexes, int):
            seqlen = indexes.max().item() + 1
        else:
            seqlen = indexes + 1  # eval_forward
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached or self._cos_cached.device != x.device or self._cos_cached.dtype != x.dtype:
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen,
                             device=x.device,
                             dtype=self.inv_freq.dtype)
            # Don't do einsum, it converts fp32 to fp16
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, self.inv_freq.to(device=t.device))
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(x.dtype)
                self._sin_cached = torch.sin(freqs).to(x.dtype)
            else:
                power = (torch.arange(
                    seqlen, dtype=self.scale.dtype, device=self.scale.device) -
                         seqlen // 2) / self.scale_base
                scale = self.scale.to(device=power.device) ** rearrange(
                    power, "s -> s 1")
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(x.dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(x.dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(x.dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(x.dtype)

    def forward(self,
                qkv: torch.Tensor,
                indexes=0) -> Tuple[torch.Tensor, torch.Tensor]:
        self._update_cos_sin_cache(qkv, indexes)
        if self.scale is None:
            return apply_rotary_emb_qkv_(qkv, self._cos_cached[indexes],
                                         self._sin_cached[indexes]).to(qkv.dtype)
        else:
            return apply_rotary_emb_qkv_(
                qkv,
                self._cos_cached[indexes],
                self._sin_cached[indexes],
                self._cos_k_cached[indexes],
                self._sin_k_cached[indexes],
            ).to(qkv.dtype)

    def eval_forward(self, qkv, seqlen_offset=0):
        """
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(qkv, seqlen_offset + qkv.shape[1])
        if self.scale is None:
            return legacy_apply_rotary_embed_qkv(
                qkv, self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:])
        else:
            return legacy_apply_rotary_embed_qkv(
                qkv,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self._cos_k_cached[seqlen_offset:],
                self._sin_k_cached[seqlen_offset:],
            )


apply_rotary_emb_qkv_ = ApplyRotaryEmbQKV_.apply
legacy_apply_rotary_embed_qkv = LegacyApplyRotaryEmbQKV_.apply


class ConvertedLlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads}).")
        if config.lora_cfg is None:
            self.q_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    bias=config.kqvo_bias)
            self.k_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    bias=config.kqvo_bias)
            self.v_proj = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim,
                                    bias=config.kqvo_bias)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                    self.hidden_size,
                                    bias=config.kqvo_bias)

        else:
            lora_cfg = config.lora_cfg
            if 'q' in lora_cfg.learn_param:
                self.q_proj = ConvertedLoRALinear(self.hidden_size,
                                                  self.num_heads * self.head_dim,
                                                  bias=config.kqvo_bias,
                                                  **lora_cfg)
            else:
                self.q_proj = nn.Linear(self.hidden_size,
                                        self.num_heads * self.head_dim,
                                        bias=config.kqvo_bias, )
            if 'k' in lora_cfg.learn_param:
                self.k_proj = ConvertedLoRALinear(self.hidden_size,
                                                  self.num_heads * self.head_dim,
                                                  bias=config.kqvo_bias,
                                                  **lora_cfg)
            else:
                self.k_proj = nn.Linear(self.hidden_size,
                                        self.num_heads * self.head_dim,
                                        bias=config.kqvo_bias, )
            if 'v' in lora_cfg.learn_param:
                self.v_proj = ConvertedLoRALinear(self.hidden_size,
                                                  self.num_heads * self.head_dim,
                                                  bias=config.kqvo_bias,
                                                  **lora_cfg)
            else:
                self.v_proj = nn.Linear(self.hidden_size,
                                        self.num_heads * self.head_dim,
                                        bias=config.kqvo_bias, )

            if 'o' in lora_cfg.learn_param:
                self.o_proj = LoRALinear(self.num_heads * self.head_dim,
                                         self.hidden_size,
                                         bias=config.kqvo_bias,
                                         **lora_cfg)
            else:
                self.o_proj = nn.Linear(self.num_heads * self.head_dim,
                                        self.hidden_size,
                                        bias=config.kqvo_bias, )

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        q = query_states
        k = key_states
        v = value_states

        dim = q.shape[-1]
        split_dim = dim // 2
        q1 = q[..., :split_dim]
        q2 = q[..., split_dim:]
        q1 = q1.contiguous().view(bsz, -1, 1, self.head_dim)
        q2 = q2.contiguous().view(bsz, -1, 1, self.head_dim)
        k1 = k[..., :split_dim]
        k2 = k[..., split_dim:]
        k1 = k1.contiguous().view(bsz, -1, 1, self.head_dim)
        k2 = k2.contiguous().view(bsz, -1, 1, self.head_dim)
        v1 = v[..., :split_dim]
        v2 = v[..., split_dim:]
        v1 = v1.contiguous().view(bsz, -1, 1, self.head_dim)
        v2 = v2.contiguous().view(bsz, -1, 1, self.head_dim)
        qkv1 = torch.cat([q1, k1, v1], dim=2).contiguous()
        qkv2 = torch.cat([q2, k2, v2], dim=2).contiguous()
        qkv1 = qkv1.view(hidden_states.size(0), hidden_states.size(1), -1)
        qkv2 = qkv2.view(hidden_states.size(0), hidden_states.size(1), -1)
        qkv1 = rearrange(qkv1,
                         "b s (h three d) -> b s three h d",
                         three=3,
                         d=self.head_dim)
        qkv2 = rearrange(qkv2,
                         "b s (h three d) -> b s three h d",
                         three=3,
                         d=self.head_dim)
        qkv1[:, :,
        0] = torch.cat([qkv1[..., 0, :, ::2], qkv1[..., 0, :, 1::2]],
                       dim=-1)
        qkv1[:, :,
        1] = torch.cat([qkv1[..., 1, :, ::2], qkv1[..., 1, :, 1::2]],
                       dim=-1)
        qkv2[:, :,
        0] = torch.cat([qkv2[..., 0, :, ::2], qkv2[..., 0, :, 1::2]],
                       dim=-1)
        qkv2[:, :,
        1] = torch.cat([qkv2[..., 1, :, ::2], qkv2[..., 1, :, 1::2]],
                       dim=-1)
        qkv = torch.cat([qkv1, qkv2], -2)

        if past_key_value is not None:
            qkv = self.rotary_emb.eval_forward(qkv, seqlen_offset=past_key_value[0].shape[2])
        else:
            qkv = self.rotary_emb.eval_forward(qkv)

        query_states, key_states, value_states = qkv.unbind(2)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        # cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        # query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(
            2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}")

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights,
                torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
            query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}")

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class ConvertedLoRALinear(nn.Linear):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 lora_r=8,
                 lora_alpha=16,
                 lora_dropout=0.05, **kwargs) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        self.lora_scaling = self.lora_alpha / self.lora_r

        self.lora_A = nn.Linear(in_features,
                                self.lora_r,
                                bias=False,
                                device=device,
                                dtype=dtype)
        self.lora_B = nn.Linear(self.lora_r,
                                out_features,
                                bias=False,
                                device=device,
                                dtype=dtype)

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            # print ("lora weight init {} {}".format(torch.mean(self.lora_A.weight), torch.mean(self.lora_B.weight)))

    def forward(self, x):
        orig_type = x.dtype
        res = super().forward(x)

        dim = int(res.shape[-1] // 2)

        r1 = res[..., :dim]
        r2 = res[..., dim:]

        r1 = r1.float()
        r2 = r2.float()
        x_ = x.float()

        tmp = self.lora_B(self.lora_A(
            self.lora_dropout(x_))) * self.lora_scaling
        tmp1 = tmp[..., ::2]
        tmp2 = tmp[..., 1::2]

        r1 += tmp1
        r2 += tmp2

        r1 = r1.to(orig_type)
        r2 = r2.to(orig_type)

        res = torch.cat([r1, r2], -1)

        # res += self.lora_B(self.lora_A(
        #     self.lora_dropout(x))) * self.lora_scaling
        return res
