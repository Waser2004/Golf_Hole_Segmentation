import torch


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(torch.nn.Module):
    """Simple U-Net implementation that returns raw logits."""

    def __init__(self, in_channels: int = 3, num_classes: int = 13, depth: int = 5, base_channels: int = 16):
        super().__init__()

        if depth < 2:
            raise ValueError("UNet depth must be >= 2")

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        encoder_channels = [base_channels * (2**i) for i in range(depth)]

        self.down_blocks = torch.nn.ModuleList()
        current_in = in_channels
        for out_ch in encoder_channels:
            self.down_blocks.append(DoubleConv(current_in, out_ch))
            current_in = out_ch

        self.bottleneck = DoubleConv(encoder_channels[-1], encoder_channels[-1] * 2)

        self.up_transpose = torch.nn.ModuleList()
        self.up_blocks = torch.nn.ModuleList()

        current_channels = encoder_channels[-1] * 2
        for out_ch in reversed(encoder_channels):
            self.up_transpose.append(
                torch.nn.ConvTranspose2d(current_channels, out_ch, kernel_size=2, stride=2)
            )
            self.up_blocks.append(DoubleConv(out_ch * 2, out_ch))
            current_channels = out_ch

        self.final_conv = torch.nn.Conv2d(encoder_channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        for down_block in self.down_blocks:
            x = down_block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, up_block, skip in zip(self.up_transpose, self.up_blocks, reversed(skips)):
            x = upconv(x)

            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            x = torch.cat([skip, x], dim=1)
            x = up_block(x)

        return self.final_conv(x)
