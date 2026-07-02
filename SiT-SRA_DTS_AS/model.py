# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from torch.nn.init import trunc_normal_


class SimpleHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleHead, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim + out_dim)
        self.linear2 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(self.act(x))
        return x


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def modulate_per_token(x, shift, scale):
    """Per-token modulation for (N, T, D) conditioning."""
    return x * (1 + scale) + shift


#################################################################################
#                          Group Attention Separation                           #
#################################################################################
class GroupSeparatedAttention(nn.Module):
    """
    Attention with pairwise token-group separation.

    separation_mask: (N, T, T), bool
        True  -> allow attention
        False -> block attention

    This prevents attention interaction across token groups.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        fused_attn=False,
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # kept for compatibility
        self.fused_attn = fused_attn

    def forward(self, x, separation_mask=None):
        """
        x: (N, T, D)
        separation_mask: (N, T, T), bool
        """
        N, T, D = x.shape

        qkv = self.qkv(x).reshape(N, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, N, H, T, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (N, H, T, Hd)

        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (N, H, T, T)

        if separation_mask is not None:
            # (N, T, T) -> (N, 1, T, T)
            separation_mask = separation_mask[:, None, :, :]
            mask_value = torch.finfo(attn.dtype).min
            attn = attn.masked_fill(~separation_mask, mask_value)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (N, H, T, Hd)
        out = out.transpose(1, 2).reshape(N, T, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def positional_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                  These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        self.timestep_embedding = self.positional_embedding
        t_freq = self.timestep_embedding(t, dim=self.frequency_embedding_size).to(t.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    Also handles label dropout for classifier-free guidance.

    Supports:
        - labels: (N,)
        - labels: (N, T)

    CFG dropout is per-sample:
        if one sample is dropped, all its token labels are dropped together.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.

        labels:
            - (N,)
            - (N, T)

        force_drop_ids:
            - for (N,): shape (N,)
            - for (N, T): shape (N,) preferred, will be broadcast to full sequence
        """
        if labels.ndim == 1:
            if force_drop_ids is None:
                drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            else:
                drop_ids = force_drop_ids == 1
            labels = torch.where(drop_ids, self.num_classes, labels)
            return labels

        elif labels.ndim == 2:
            batch_size = labels.shape[0]
            if force_drop_ids is None:
                drop_ids = torch.rand(batch_size, device=labels.device) < self.dropout_prob
            else:
                drop_ids = force_drop_ids == 1

            # per-sample CFG: one decision for the whole sequence
            drop_ids = drop_ids[:, None].expand_as(labels)
            labels = torch.where(drop_ids, self.num_classes, labels)
            return labels

        else:
            raise ValueError(f"Labels must be 1D or 2D, got shape {labels.shape}")

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)

        embeddings = self.embedding_table(labels)
        return embeddings, labels


#################################################################################
#                                 Core SiT Model                                #
#################################################################################
class SiTBlock(nn.Module):
    """
    A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Optionally separates attention across token groups.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attention_separation = block_kwargs.get("attention_separation", False)

        if self.attention_separation:
            self.attn = GroupSeparatedAttention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=block_kwargs.get("qk_norm", False),
                fused_attn=block_kwargs.get("fused_attn", False),
            )
        else:
            self.attn = Attention(
                hidden_size,
                num_heads=num_heads,
                qkv_bias=True,
                qk_norm=block_kwargs.get("qk_norm", False),
            )
            if "fused_attn" in block_kwargs:
                self.attn.fused_attn = block_kwargs["fused_attn"]

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, separation_mask=None):
        """
        Args:
            x: (N, T, D) tokens
            c: (N, T, D) per-token conditioning
            separation_mask: (N, T, T) bool, True=allow attend, False=block
        """
        batch_size, seq_len, hidden_dim = c.shape
        c_flat = c.reshape(-1, hidden_dim)
        modulation_flat = self.adaLN_modulation(c_flat)
        modulation = modulation_flat.reshape(batch_size, seq_len, -1)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)

        attn_input = modulate_per_token(self.norm1(x), shift_msa, scale_msa)
        if self.attention_separation:
            attn_output = self.attn(attn_input, separation_mask=separation_mask)
        else:
            attn_output = self.attn(attn_input)
        x = x + gate_msa * attn_output
        x = x + gate_mlp * self.mlp(
            modulate_per_token(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of SiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        Args:
            x: (N, T, D) tokens
            c: (N, T, D) per-token conditioning
        """
        batch_size, seq_len, hidden_dim = c.shape
        c_flat = c.reshape(-1, hidden_dim)
        modulation_flat = self.adaLN_modulation(c_flat)
        modulation = modulation_flat.reshape(batch_size, seq_len, -1)

        shift, scale = modulation.chunk(2, dim=-1)
        x = modulate_per_token(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.

    Supports mixed-token input using mask:
    - mixed groups share the same sequence by default
    - if attention_separation=True, cross-group attention is blocked
    """

    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        use_alignment_head=False,
        attention_separation=False,
        **block_kwargs
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.use_alignment_head = use_alignment_head
        self.attention_separation = attention_separation
        block_kwargs = {**block_kwargs, "attention_separation": attention_separation}

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches

        # fixed sin-cos embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, hidden_size),
            requires_grad=False
        )

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs)
            for _ in range(depth)
        ])

        if self.use_alignment_head:
            self.ap_head = SimpleHead(hidden_size, hidden_size)

        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear:
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in SiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def patchify(self, imgs, patch_size=None):
        """
        imgs: (N, C, H, W)
        returns: (N, T, p*p*C)
        """
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        N, C, H, W = imgs.shape
        assert H == W
        assert H % p == 0

        h = w = H // p
        x = imgs.reshape(N, C, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(N, h * w, p * p * C)
        return x

    def unpatchify(self, x, patch_size=None):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def build_attention_separation_mask(self, group_mask):
        """
        group_mask: (N, T)
            bool: two-group mask, True selects group 0 and False selects group 1
            long: group ids in [0, num_groups)

        returns:
            separation_mask: (N, T, T) bool
            True if two tokens belong to the same group, else False
        """
        return group_mask[:, :, None] == group_mask[:, None, :]

    def normalize_group_mask(self, mask):
        if mask.ndim == 4:
            mask = mask.squeeze(-1).squeeze(-1)
        elif mask.ndim == 3:
            mask = mask.squeeze(-1)

        if mask.dtype == torch.bool:
            return torch.where(mask, torch.zeros_like(mask, dtype=torch.long), torch.ones_like(mask, dtype=torch.long))

        return mask.long()

    def mix_group_tokens(self, tokens, group_mask):
        num_groups, batch_size, seq_len, hidden_dim = tokens.shape
        tokens = tokens.permute(1, 2, 0, 3)
        gather_index = group_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, seq_len, 1, hidden_dim)
        return tokens.gather(2, gather_index).squeeze(2)

    def forward(self, x, t, y, ad=4, mask=None, gt=None, return_token=False):
        """
        Forward pass of SiT.

        Args:
            x:
                - normal mode: (N, C, H, W)
                - mixed mode:  (2N, C, H, W), first N are x_small, last N are x_large
            t:
                - normal mode: (N,) or (N, T)
                - mixed mode:  (N,) or (N, T), aligned with merged tokens
            y:
                - normal mode: (N,) or (N, T)
                - mixed mode:  (N,) or (N, T), aligned with merged tokens
            ad: number of layers to use for self-alignment
            mask:
                - None, or
                - (N, T), (N, T, 1), or (N, T, 1, 1)
                - bool for the legacy two-group mode
                - long for group ids in the multi-group mode
            gt:
                - None
                - normal mode: (N, C, H, W)
                - mixed mode:  (2N, C, H, W), same split rule as x
            return_token:
                - if True, return token prediction
                - if False, return image prediction after unpatchify

        Returns:
            pred:
                - if return_token=True:  (N, T, patch_dim)
                - else:                  (N, C, H, W)
            xr:
                alignment feature
            gt_token:
                - None if gt is None
                - else tokenized and mixed target: (N, T, patch_dim)
        """
        x = self.x_embedder(x) + self.pos_embed  # (N_or_2N, T, D)

        separation_mask = None
        gt_token = None

        # Mixed-token mode:
        if mask is not None and x.shape[0] % mask.shape[0] == 0 and x.shape[0] != mask.shape[0]:
            group_mask = self.normalize_group_mask(mask)
            bsz = group_mask.shape[0]
            num_groups = x.shape[0] // bsz

            if group_mask.max().item() >= num_groups or group_mask.min().item() < 0:
                raise ValueError(f"group ids should be in [0, {num_groups}), got min={group_mask.min().item()} max={group_mask.max().item()}")

            x_groups = x.reshape(num_groups, bsz, x.shape[1], x.shape[2])
            x = self.mix_group_tokens(x_groups, group_mask)

            if self.attention_separation:
                separation_mask = self.build_attention_separation_mask(group_mask)

            # process gt if provided
            if gt is not None:
                assert gt.shape[0] == num_groups * bsz, f"gt batch size {gt.shape[0]} != {num_groups} * {bsz}"
                gt_groups = self.patchify(gt).reshape(num_groups, bsz, x.shape[1], -1)
                gt_token = self.mix_group_tokens(gt_groups, group_mask)

            batch_size, seq_len, hidden_dim = x.shape

            # timestep embedding
            if t.ndim == 1:
                t_embed = self.t_embedder(t).unsqueeze(1).expand(-1, seq_len, -1)
            elif t.ndim == 2:
                t_flat = t.reshape(-1)
                t_emb_flat = self.t_embedder(t_flat)
                t_embed = t_emb_flat.reshape(batch_size, seq_len, -1)
            else:
                raise ValueError(f"Timesteps must be 1D or 2D, got shape {t.shape}")

            # label embedding
            if y.ndim == 1:
                y_embed, labels_train = self.y_embedder(y, self.training)
                y_embed = y_embed.unsqueeze(1).expand(-1, seq_len, -1)
            elif y.ndim == 2:
                y_embed, labels_train = self.y_embedder(y, self.training)  # (B, T, D)
            else:
                raise ValueError(f"Labels must be 1D or 2D, got shape {y.shape}")

            c = t_embed + y_embed  # (B, T, D)

        else:
            # Normal mode
            batch_size, seq_len, hidden_dim = x.shape

            if gt is not None:
                assert gt.shape[0] == batch_size, f"gt batch size {gt.shape[0]} != x batch size {batch_size}"
                gt_token = self.patchify(gt)

            if t.ndim == 1:
                t_embed = self.t_embedder(t).unsqueeze(1).expand(-1, seq_len, -1)
            elif t.ndim == 2:
                t_flat = t.reshape(-1)
                t_emb_flat = self.t_embedder(t_flat)
                t_embed = t_emb_flat.reshape(batch_size, seq_len, -1)
            else:
                raise ValueError(f"Timesteps must be 1D or 2D, got shape {t.shape}")

            if y.ndim == 1:
                y_embed, labels_train = self.y_embedder(y, self.training)
                y_embed = y_embed.unsqueeze(1).expand(-1, seq_len, -1)
            elif y.ndim == 2:
                y_embed, labels_train = self.y_embedder(y, self.training)  # (N, T, D)
            else:
                raise ValueError(f"Labels must be 1D or 2D, got shape {y.shape}")

            c = t_embed + y_embed  # (N, T, D)

        xr = 0

        for i, block in enumerate(self.blocks):
            x = block(x, c, separation_mask=separation_mask)  # (N, T, D)
            if (i + 1) == ad:
                if self.training and self.use_alignment_head:
                    xr = self.ap_head(x)
                else:
                    xr = x

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)

        if return_token:
            return x, xr, gt_token

        x_img = self.unpatchify(x)  # (N, out_channels, H, W)
        return x_img, xr


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
               or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)        # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2)

    emb_sin = np.sin(out)        # (M, D/2)
    emb_cos = np.cos(out)        # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   SiT Configs                                 #
#################################################################################
def SiT_XL_2(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def SiT_XL_4(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def SiT_XL_8(**kwargs):
    return SiT(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def SiT_L_2(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def SiT_L_4(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def SiT_L_8(**kwargs):
    return SiT(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def SiT_B_2(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def SiT_B_4(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def SiT_B_8(**kwargs):
    return SiT(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def SiT_S_2(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def SiT_S_4(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def SiT_S_8(**kwargs):
    return SiT(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_models = {
    'SiT-XL/2': SiT_XL_2, 'SiT-XL/4': SiT_XL_4, 'SiT-XL/8': SiT_XL_8,
    'SiT-L/2': SiT_L_2,  'SiT-L/4': SiT_L_4,  'SiT-L/8': SiT_L_8,
    'SiT-B/2': SiT_B_2,  'SiT-B/4': SiT_B_4,  'SiT-B/8': SiT_B_8,
    'SiT-S/2': SiT_S_2,  'SiT-S/4': SiT_S_4,  'SiT-S/8': SiT_S_8,
}
