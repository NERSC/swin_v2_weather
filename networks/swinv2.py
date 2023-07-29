import logging
import math
from typing import Tuple, Optional, List, Union, Any, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.swin_transformer_v2_cr import PatchEmbed, SwinTransformerV2CrStage
from timm.layers import ClassifierHead, to_2tuple, Mlp


# Adapted from timm v0.9.2:
# https://github.com/huggingface/pytorch-image-models/blob/v0.9.2/timm/models/swin_transformer_v2_cr.py


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C). """
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W). """
    return x.permute(0, 3, 1, 2)


def swinv2net(params):
    return SwinV2(
                  img_size=params.img_size,
                  patch_size=params.patch_size,
                  depths = (params.depth//3, params.depth//3, params.depth//3),   
                  num_heads=(params.num_heads, params.num_heads, params.num_heads),
                  in_chans=params.n_channels,
                  out_chans=params.n_channels,
                  embed_dim=params.embed_dim
    )
                  

class SwinV2(nn.Module):
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        window_size: Optional[int] = None,
        img_window_ratio: int = 32,
        in_chans: int = 3,
        out_chans: int = 3,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0.,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
        sequential_attn: bool = False,
        global_pool: str = 'avg',
        weight_init='skip',
        **kwargs: Any
    ) -> None:
        super(SwinV2, self).__init__()
        img_size = to_2tuple(img_size)
        window_size = tuple([
            s // img_window_ratio for s in img_size]) if window_size is None else to_2tuple(window_size)

        self.patch_size: int = patch_size
        self.img_size: Tuple[int, int] = img_size
        self.window_size: int = window_size
        self.num_features: int = int(embed_dim)
        self.out_chans: int = out_chans
        self.feature_info = []

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        patch_grid_size: Tuple[int, int] = self.patch_embed.grid_size

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        in_dim = embed_dim
        in_scale = 1
        for stage_idx, (depth, num_heads) in enumerate(zip(depths, num_heads)):
            stages += [SwinTransformerV2CrStage(
                embed_dim=in_dim,
                depth=depth,
                downscale=False,
                feat_size=(
                    patch_grid_size[0] // in_scale,
                    patch_grid_size[1] // in_scale
                ),
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                drop_attn=attn_drop_rate,
                drop_path=dpr[stage_idx],
                extra_norm_period=extra_norm_period,
                extra_norm_stage=extra_norm_stage or (stage_idx + 1) == len(depths),  # last stage ends w/ norm
                sequential_attn=sequential_attn,
                norm_layer=norm_layer,
            )]
            self.feature_info += [dict(num_chs=in_dim, reduction=4 * in_scale, module=f'stages.{stage_idx}')]
            
        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size*self.patch_size, bias=False)
        # current weight init skips custom init and uses pytorch layer defaults, seems to work well
        # FIXME more experiments needed
        if weight_init != 'skip':
            named_apply(init_weights, self)
            
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.stages(x)
        return x
    
    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        B, _, h, w = x.shape
        x = bchw_to_bhwc(x)
        x = self.head(x)
        
        x = x.reshape(shape=(B, h, w, self.patch_size, self.patch_size, self.out_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_chans, self.img_size[0], self.img_size[1]))    
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



