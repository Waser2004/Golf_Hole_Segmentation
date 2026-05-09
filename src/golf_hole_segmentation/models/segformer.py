import torch
import torch.nn.functional as F


class SegFormer(torch.nn.Module):
    """SegFormer implementation with selectable MiT-B0..MiT-B5 variants."""

    _VARIANT_CONFIGS = {
        "B0": {"embed_dims": [32, 64, 160, 256], "depths": [2, 2, 2, 2], "decoder_embed_dim": 256},
        "B1": {"embed_dims": [64, 128, 320, 512], "depths": [2, 2, 2, 2], "decoder_embed_dim": 256},
        "B2": {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 6, 3], "decoder_embed_dim": 768},
        "B3": {"embed_dims": [64, 128, 320, 512], "depths": [3, 4, 18, 3], "decoder_embed_dim": 768},
        "B4": {"embed_dims": [64, 128, 320, 512], "depths": [3, 8, 27, 3], "decoder_embed_dim": 768},
        "B5": {"embed_dims": [64, 128, 320, 512], "depths": [3, 6, 40, 3], "decoder_embed_dim": 768},
    }

    class _DropPath(torch.nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.drop_prob == 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor

    class _DWConv(torch.nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dwconv = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=True)

        def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
            batch_size, _, channels = x.shape
            x = x.transpose(1, 2).reshape(batch_size, channels, height, width)
            x = self.dwconv(x)
            return x.flatten(2).transpose(1, 2)

    class _Mlp(torch.nn.Module):
        def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
            super().__init__()
            self.fc1 = torch.nn.Linear(in_features, hidden_features)
            self.dwconv = SegFormer._DWConv(hidden_features)
            self.act = torch.nn.GELU()
            self.fc2 = torch.nn.Linear(hidden_features, in_features)
            self.drop = torch.nn.Dropout(drop)

        def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
            x = self.fc1(x)
            x = self.dwconv(x, height, width)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class _Attention(torch.nn.Module):
        def __init__(self, dim: int, num_heads: int, sr_ratio: int, qkv_bias: bool, attn_drop: float, proj_drop: float):
            super().__init__()
            if dim % num_heads != 0:
                raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

            self.num_heads = num_heads
            self.scale = (dim // num_heads) ** -0.5
            self.sr_ratio = sr_ratio

            self.q = torch.nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = torch.nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.attn_drop = torch.nn.Dropout(attn_drop)
            self.proj = torch.nn.Linear(dim, dim)
            self.proj_drop = torch.nn.Dropout(proj_drop)

            if sr_ratio > 1:
                self.sr = torch.nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = torch.nn.LayerNorm(dim, eps=1e-6)

        def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
            batch_size, num_tokens, channels = x.shape

            q = self.q(x).reshape(batch_size, num_tokens, self.num_heads, channels // self.num_heads)
            q = q.permute(0, 2, 1, 3)

            if self.sr_ratio > 1:
                x_reduced = x.permute(0, 2, 1).reshape(batch_size, channels, height, width)
                x_reduced = self.sr(x_reduced).reshape(batch_size, channels, -1).permute(0, 2, 1)
                x_reduced = self.norm(x_reduced)
                kv = self.kv(x_reduced)
            else:
                kv = self.kv(x)

            kv = kv.reshape(batch_size, -1, 2, self.num_heads, channels // self.num_heads)
            kv = kv.permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, channels)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x

    class _Block(torch.nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float,
            sr_ratio: int,
            qkv_bias: bool,
            drop: float,
            attn_drop: float,
            drop_path: float,
        ):
            super().__init__()
            self.norm1 = torch.nn.LayerNorm(dim, eps=1e-6)
            self.attn = SegFormer._Attention(
                dim=dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.drop_path = SegFormer._DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
            self.norm2 = torch.nn.LayerNorm(dim, eps=1e-6)
            self.mlp = SegFormer._Mlp(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                drop=drop,
            )

        def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
            x = x + self.drop_path(self.attn(self.norm1(x), height, width))
            x = x + self.drop_path(self.mlp(self.norm2(x), height, width))
            return x

    class _OverlapPatchEmbed(torch.nn.Module):
        def __init__(self, patch_size: int, stride: int, in_channels: int, embed_dim: int):
            super().__init__()
            self.proj = torch.nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                padding=patch_size // 2,
            )
            self.norm = torch.nn.LayerNorm(embed_dim, eps=1e-6)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
            x = self.proj(x)
            _, _, height, width = x.shape
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            return x, height, width

    class _LinearEmbed(torch.nn.Module):
        def __init__(self, input_dim: int, embed_dim: int):
            super().__init__()
            self.proj = torch.nn.Linear(input_dim, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.flatten(2).transpose(1, 2)
            return self.proj(x)

    def __init__(
        self,
        variant: str = "B0",
        in_channels: int = 3,
        num_classes: int = 13,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        decoder_dropout: float = 0.1,
    ):
        super().__init__()

        variant_key = variant.upper()
        if variant_key not in self._VARIANT_CONFIGS:
            allowed = ", ".join(self._VARIANT_CONFIGS.keys())
            raise ValueError(f"Unknown SegFormer variant '{variant}'. Supported variants: {allowed}")

        cfg = self._VARIANT_CONFIGS[variant_key]
        embed_dims = cfg["embed_dims"]
        depths = cfg["depths"]
        decoder_embed_dim = cfg["decoder_embed_dim"]

        num_heads = [1, 2, 5, 8]
        mlp_ratios = [4.0, 4.0, 4.0, 4.0]
        sr_ratios = [8, 4, 2, 1]
        patch_sizes = [7, 3, 3, 3]
        strides = [4, 2, 2, 2]

        self.patch_embeds = torch.nn.ModuleList()
        input_dims = [in_channels] + embed_dims[:-1]
        for in_dim, out_dim, patch_size, stride in zip(input_dims, embed_dims, patch_sizes, strides):
            self.patch_embeds.append(
                self._OverlapPatchEmbed(
                    patch_size=patch_size,
                    stride=stride,
                    in_channels=in_dim,
                    embed_dim=out_dim,
                )
            )

        total_depth = sum(depths)
        dpr = torch.linspace(0, drop_path_rate, total_depth).tolist()
        dpr_index = 0

        self.blocks = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()
        for stage_idx in range(4):
            stage_blocks = []
            for layer_idx in range(depths[stage_idx]):
                stage_blocks.append(
                    self._Block(
                        dim=embed_dims[stage_idx],
                        num_heads=num_heads[stage_idx],
                        mlp_ratio=mlp_ratios[stage_idx],
                        sr_ratio=sr_ratios[stage_idx],
                        qkv_bias=True,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        drop_path=dpr[dpr_index + layer_idx],
                    )
                )
            self.blocks.append(torch.nn.ModuleList(stage_blocks))
            self.norms.append(torch.nn.LayerNorm(embed_dims[stage_idx], eps=1e-6))
            dpr_index += depths[stage_idx]

        self.decoder_mlps = torch.nn.ModuleList(
            [self._LinearEmbed(input_dim=dim, embed_dim=decoder_embed_dim) for dim in embed_dims]
        )
        self.linear_fuse = torch.nn.Sequential(
            torch.nn.Conv2d(decoder_embed_dim * 4, decoder_embed_dim, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(decoder_embed_dim),
            torch.nn.ReLU(inplace=True),
        )
        self.dropout = torch.nn.Dropout2d(decoder_dropout) if decoder_dropout > 0.0 else torch.nn.Identity()
        self.classifier = torch.nn.Conv2d(decoder_embed_dim, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, torch.nn.BatchNorm2d):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

    def _forward_backbone(self, x: torch.Tensor) -> list[torch.Tensor]:
        outputs = []
        for patch_embed, stage_blocks, norm in zip(self.patch_embeds, self.blocks, self.norms):
            x, height, width = patch_embed(x)
            for block in stage_blocks:
                x = block(x, height, width)
            x = norm(x)
            x = x.reshape(x.shape[0], height, width, -1).permute(0, 3, 1, 2).contiguous()
            outputs.append(x)
        return outputs

    def _forward_decode(self, features: list[torch.Tensor]) -> torch.Tensor:
        c1, c2, c3, c4 = features
        target_size = c1.shape[-2:]

        projected = []
        for feat, mlp in zip((c1, c2, c3, c4), self.decoder_mlps):
            batch_size, _, height, width = feat.shape
            token_feat = mlp(feat).permute(0, 2, 1).reshape(batch_size, -1, height, width)
            if (height, width) != target_size:
                token_feat = F.interpolate(token_feat, size=target_size, mode="bilinear", align_corners=False)
            projected.append(token_feat)

        x = torch.cat((projected[3], projected[2], projected[1], projected[0]), dim=1)
        x = self.linear_fuse(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        features = self._forward_backbone(x)
        logits = self._forward_decode(features)
        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
