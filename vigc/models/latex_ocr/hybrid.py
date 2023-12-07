import torch
import numpy as np
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer_hybrid import HybridEmbed
from timm.models.resnetv2 import ResNetV2
from timm.models.layers import StdConv2dSame
from einops import repeat


class CustomVisionTransformer(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, multi_scale=False, *args, **kwargs):
        super(CustomVisionTransformer, self).__init__(img_size=img_size, patch_size=patch_size, *args, **kwargs)
        self.height, self.width = img_size
        self.patch_size = patch_size
        self.multi_scale = multi_scale
        self.cls_pos_embedding = torch.nn.Parameter(torch.zeros(1, self.embed_dim))

    def forward_features(self, x):
        B, c, h, w = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        h, w = h // self.patch_size, w // self.patch_size
        if self.multi_scale:
            x = self.position_encoding(x, h, w)
        else:
            pos_emb_ind = repeat(torch.arange(h) * (self.width // self.patch_size - w), 'h -> (h w)',
                                 w=w) + torch.arange(
                h * w)
            pos_emb_ind = torch.cat((torch.zeros(1), pos_emb_ind + 1), dim=0).long()
            x += self.pos_embed[:, pos_emb_ind]
            # x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def position_encoding(self, x, h, w):
        bs, _, chs = x.shape
        h_pos_emb_ind = torch.arange(h) / h * 2000.
        w_pos_emb_ind = torch.arange(w) / w * 2000.

        half_dim = chs // 4
        emb = -np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * emb)

        h_pos_emb = h_pos_emb_ind[:, None] * emb[None, :]
        h_pos_emb = torch.cat([torch.cos(h_pos_emb), torch.sin(h_pos_emb)], dim=1)
        h_pos_emb = repeat(h_pos_emb, "h d -> (h w) d", w=w)

        w_pos_emb = w_pos_emb_ind[:, None] * emb[None, :]
        w_pos_emb = torch.cat([torch.cos(w_pos_emb), torch.sin(w_pos_emb)], dim=1)
        w_pos_emb = repeat(w_pos_emb, "w d -> (h w) d", h=h)

        pos_emb = torch.cat([self.cls_pos_embedding, torch.cat([h_pos_emb, w_pos_emb], dim=1)], dim=0)[None]
        return x + pos_emb.to(x.dtype).to(x.device)


def get_encoder(args):
    backbone_layers = [int(_) for _ in args.backbone_layers]
    backbone = ResNetV2(
        layers=backbone_layers, num_classes=0, global_pool='', in_chans=args.channels,
        preact=False, stem_type='same', conv_layer=StdConv2dSame)
    min_patch_size = 2 ** (len(backbone_layers) + 1)

    def embed_layer(**x):
        ps = x.pop('patch_size', min_patch_size)
        assert ps % min_patch_size == 0 and ps >= min_patch_size, 'patch_size needs to be multiple of %i with current backbone configuration' % min_patch_size
        return HybridEmbed(**x, patch_size=ps // min_patch_size, backbone=backbone)

    encoder = CustomVisionTransformer(
        img_size=(args.max_height, args.max_width),
        multi_scale=args.multi_scale,
        patch_size=args.patch_size,
        in_chans=args.channels,
        num_classes=0,
        embed_dim=args.dim,
        depth=args.encoder_depth,
        num_heads=args.heads,
        embed_layer=embed_layer
    )
    return encoder
