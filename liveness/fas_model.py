# -*- coding: utf-8 -*-
"""
fas_model.py
============
Face Anti-Spoofing (FAS) models:
1. MultiFTNet: MiniFASNetV2SE backbone trained from scratch with Fourier Auxiliary Supervision.
   Source: minivision-ai/Silent-Face-Anti-Spoofing (Apache-2.0).
   Note: Architecture code only; no pretrained weights are used for MultiFTNet.
2. MobileNetV3FAS: MobileNetV3-Large backbone with ImageNet pretrained weights for full fine-tuning
   (transfer learning with fine-tuning allowed per project specifications) combined
   with Fourier Auxiliary Supervision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import (
    Linear,
    Conv2d,
    BatchNorm1d,
    BatchNorm2d,
    PReLU,
    ReLU,
    Sigmoid,
    AdaptiveAvgPool2d,
    Sequential,
    Module,
)
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def get_kernel(height: int, width: int) -> tuple[int, int]:
    """Calculate conv6 depthwise kernel size based on input resolution."""
    return ((height + 15) // 16, (width + 15) // 16)


class L2Norm(Module):
    def forward(self, input):
        return F.normalize(input)


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Conv_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Conv_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)
        self.prelu = PReLU(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class Linear_block(Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super(Linear_block, self).__init__()
        self.conv = Conv2d(
            in_c,
            out_channels=out_c,
            kernel_size=kernel,
            groups=groups,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = BatchNorm2d(out_c)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Depth_Wise(Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
    ):
        super(Depth_Wise, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(
            c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_dw = Conv_block(
            c2_in,
            c2_out,
            groups=c2_in,
            kernel=kernel,
            padding=padding,
            stride=stride,
        )
        self.project = Linear_block(
            c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.residual = residual

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            output = short_cut + x
        else:
            output = x
        return output


class Residual(Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        num_block,
        groups,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
    ):
        super(Residual, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            modules.append(
                Depth_Wise(
                    c1_tuple,
                    c2_tuple,
                    c3_tuple,
                    residual=True,
                    kernel=kernel,
                    padding=padding,
                    stride=stride,
                    groups=groups,
                )
            )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=False
        )
        self.bn1 = BatchNorm2d(channels // reduction)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=False
        )
        self.bn2 = BatchNorm2d(channels)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.sigmoid(x)
        return module_input * x


class Depth_Wise_SE(Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        residual=False,
        kernel=(3, 3),
        stride=(2, 2),
        padding=(1, 1),
        groups=1,
        se_reduct=8,
    ):
        super(Depth_Wise_SE, self).__init__()
        c1_in, c1_out = c1
        c2_in, c2_out = c2
        c3_in, c3_out = c3
        self.conv = Conv_block(
            c1_in, out_c=c1_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.conv_dw = Conv_block(
            c2_in,
            c2_out,
            groups=c2_in,
            kernel=kernel,
            padding=padding,
            stride=stride,
        )
        self.project = Linear_block(
            c3_in, c3_out, kernel=(1, 1), padding=(0, 0), stride=(1, 1)
        )
        self.residual = residual
        self.se_module = SEModule(c3_out, se_reduct)

    def forward(self, x):
        if self.residual:
            short_cut = x
        x = self.conv(x)
        x = self.conv_dw(x)
        x = self.project(x)
        if self.residual:
            x = self.se_module(x)
            output = short_cut + x
        else:
            output = x
        return output


class ResidualSE(Module):
    def __init__(
        self,
        c1,
        c2,
        c3,
        num_block,
        groups,
        kernel=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        se_reduct=4,
    ):
        super(ResidualSE, self).__init__()
        modules = []
        for i in range(num_block):
            c1_tuple = c1[i]
            c2_tuple = c2[i]
            c3_tuple = c3[i]
            if i == num_block - 1:
                modules.append(
                    Depth_Wise_SE(
                        c1_tuple,
                        c2_tuple,
                        c3_tuple,
                        residual=True,
                        kernel=kernel,
                        padding=padding,
                        stride=stride,
                        groups=groups,
                        se_reduct=se_reduct,
                    )
                )
            else:
                modules.append(
                    Depth_Wise(
                        c1_tuple,
                        c2_tuple,
                        c3_tuple,
                        residual=True,
                        kernel=kernel,
                        padding=padding,
                        stride=stride,
                        groups=groups,
                    )
                )
        self.model = Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class MiniFASNet(Module):
    def __init__(
        self,
        keep,
        embedding_size,
        conv6_kernel=(7, 7),
        drop_p=0.0,
        num_classes=3,
        img_channel=3,
    ):
        super(MiniFASNet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = Conv_block(
            img_channel, keep[0], kernel=(3, 3), stride=(2, 2), padding=(1, 1)
        )
        self.conv2_dw = Conv_block(
            keep[0],
            keep[1],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            groups=keep[1],
        )

        c1 = [(keep[1], keep[2])]
        c2 = [(keep[2], keep[3])]
        c3 = [(keep[3], keep[4])]

        self.conv_23 = Depth_Wise(
            c1[0],
            c2[0],
            c3[0],
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=keep[3],
        )

        c1 = [
            (keep[4], keep[5]),
            (keep[7], keep[8]),
            (keep[10], keep[11]),
            (keep[13], keep[14]),
        ]
        c2 = [
            (keep[5], keep[6]),
            (keep[8], keep[9]),
            (keep[11], keep[12]),
            (keep[14], keep[15]),
        ]
        c3 = [
            (keep[6], keep[7]),
            (keep[9], keep[10]),
            (keep[12], keep[13]),
            (keep[15], keep[16]),
        ]

        self.conv_3 = Residual(
            c1,
            c2,
            c3,
            num_block=4,
            groups=keep[4],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        c1 = [(keep[16], keep[17])]
        c2 = [(keep[17], keep[18])]
        c3 = [(keep[18], keep[19])]

        self.conv_34 = Depth_Wise(
            c1[0],
            c2[0],
            c3[0],
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=keep[19],
        )

        c1 = [
            (keep[19], keep[20]),
            (keep[22], keep[23]),
            (keep[25], keep[26]),
            (keep[28], keep[29]),
            (keep[31], keep[32]),
            (keep[34], keep[35]),
        ]
        c2 = [
            (keep[20], keep[21]),
            (keep[23], keep[24]),
            (keep[26], keep[27]),
            (keep[29], keep[30]),
            (keep[32], keep[33]),
            (keep[35], keep[36]),
        ]
        c3 = [
            (keep[21], keep[22]),
            (keep[24], keep[25]),
            (keep[27], keep[28]),
            (keep[30], keep[31]),
            (keep[33], keep[34]),
            (keep[36], keep[37]),
        ]

        self.conv_4 = Residual(
            c1,
            c2,
            c3,
            num_block=6,
            groups=keep[19],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        c1 = [(keep[37], keep[38])]
        c2 = [(keep[38], keep[39])]
        c3 = [(keep[39], keep[40])]

        self.conv_45 = Depth_Wise(
            c1[0],
            c2[0],
            c3[0],
            kernel=(3, 3),
            stride=(2, 2),
            padding=(1, 1),
            groups=keep[40],
        )

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]

        self.conv_5 = Residual(
            c1,
            c2,
            c3,
            num_block=2,
            groups=keep[40],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        self.conv_6_sep = Conv_block(
            keep[46], keep[47], kernel=(1, 1), stride=(1, 1), padding=(0, 0)
        )
        self.conv_6_dw = Linear_block(
            keep[47],
            keep[48],
            groups=keep[48],
            kernel=conv6_kernel,
            stride=(1, 1),
            padding=(0, 0),
        )
        self.conv_6_flatten = Flatten()
        self.linear = Linear(512, embedding_size, bias=False)
        self.bn = BatchNorm1d(embedding_size)
        self.drop = nn.Dropout(p=drop_p)
        self.prob = Linear(embedding_size, num_classes, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2_dw(out)
        out = self.conv_23(out)
        out = self.conv_3(out)
        out = self.conv_34(out)
        out = self.conv_4(out)
        out = self.conv_45(out)
        out = self.conv_5(out)
        out = self.conv_6_sep(out)
        out = self.conv_6_dw(out)
        out = self.conv_6_flatten(out)
        if self.embedding_size != 512:
            out = self.linear(out)
        out = self.bn(out)
        out = self.drop(out)
        out = self.prob(out)
        return out


class MiniFASNetSE(MiniFASNet):
    def __init__(
        self,
        keep,
        embedding_size,
        conv6_kernel=(7, 7),
        drop_p=0.75,
        num_classes=4,
        img_channel=3,
    ):
        super(MiniFASNetSE, self).__init__(
            keep=keep,
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
            drop_p=drop_p,
            num_classes=num_classes,
            img_channel=img_channel,
        )

        c1 = [
            (keep[4], keep[5]),
            (keep[7], keep[8]),
            (keep[10], keep[11]),
            (keep[13], keep[14]),
        ]
        c2 = [
            (keep[5], keep[6]),
            (keep[8], keep[9]),
            (keep[11], keep[12]),
            (keep[14], keep[15]),
        ]
        c3 = [
            (keep[6], keep[7]),
            (keep[9], keep[10]),
            (keep[12], keep[13]),
            (keep[15], keep[16]),
        ]

        self.conv_3 = ResidualSE(
            c1,
            c2,
            c3,
            num_block=4,
            groups=keep[4],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        c1 = [
            (keep[19], keep[20]),
            (keep[22], keep[23]),
            (keep[25], keep[26]),
            (keep[28], keep[29]),
            (keep[31], keep[32]),
            (keep[34], keep[35]),
        ]
        c2 = [
            (keep[20], keep[21]),
            (keep[23], keep[24]),
            (keep[26], keep[27]),
            (keep[29], keep[30]),
            (keep[32], keep[33]),
            (keep[35], keep[36]),
        ]
        c3 = [
            (keep[21], keep[22]),
            (keep[24], keep[25]),
            (keep[27], keep[28]),
            (keep[30], keep[31]),
            (keep[33], keep[34]),
            (keep[36], keep[37]),
        ]

        self.conv_4 = ResidualSE(
            c1,
            c2,
            c3,
            num_block=6,
            groups=keep[19],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        c1 = [(keep[40], keep[41]), (keep[43], keep[44])]
        c2 = [(keep[41], keep[42]), (keep[44], keep[45])]
        c3 = [(keep[42], keep[43]), (keep[45], keep[46])]
        self.conv_5 = ResidualSE(
            c1,
            c2,
            c3,
            num_block=2,
            groups=keep[40],
            kernel=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )


keep_dict = {
    "1.8M_": [
        32, 32, 103, 103, 64, 13, 13, 64, 13, 13, 64, 13,
        13, 64, 13, 13, 64, 231, 231, 128, 231, 231, 128, 52,
        52, 128, 26, 26, 128, 77, 77, 128, 26, 26, 128, 26, 26,
        128, 308, 308, 128, 26, 26, 128, 26, 26, 128, 512, 512,
    ]
}


def MiniFASNetV2SE(
    embedding_size=128,
    conv6_kernel=(5, 5),
    drop_p=0.75,
    num_classes=2,
    img_channel=3,
):
    return MiniFASNetSE(
        keep_dict["1.8M_"],
        embedding_size,
        conv6_kernel,
        drop_p,
        num_classes,
        img_channel,
    )


class FTGenerator(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super(FTGenerator, self).__init__()
        self.ft = Sequential(
            Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            BatchNorm2d(128),
            ReLU(inplace=True),
            Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            BatchNorm2d(64),
            ReLU(inplace=True),
            Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            BatchNorm2d(out_channels),
            ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    def __init__(
        self,
        img_channel=3,
        num_classes=2,
        embedding_size=128,
        conv6_kernel=(5, 5),
    ):
        super(MultiFTNet, self).__init__()
        self.img_channel = img_channel
        self.num_classes = num_classes
        self.model = MiniFASNetV2SE(
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
            num_classes=num_classes,
            img_channel=img_channel,
        )
        self.FTGenerator = FTGenerator(in_channels=128)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        if self.model.embedding_size != 512:
            x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)

        if self.training:
            ft = self.FTGenerator(x)
            return cls, ft
        else:
            return cls


# ═════════════════════════════════════════════════════════════════════════════
#  MobileNetV3-Large with Fourier Auxiliary Supervision (Transfer Learning)
# ═════════════════════════════════════════════════════════════════════════════

class MobileNetV3FAS(nn.Module):
    """
    MobileNetV3-Large backbone (pretrained ImageNet) fine-tuned for Face Anti-Spoofing.
    Combines binary classification head with Fourier Transform (FT) auxiliary supervision
    attached at stride-8 feature map (stem_to_s8: 40 channels, 10x10 resolution for 80x80 input).

    Normalization:
        Input tensor x is assumed to be in [-1.0, 1.0] (RGB, from fas_common.to_tensor).
        Internal forward converts to [0, 1] then normalizes via standard ImageNet mean & std.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = False,
        ft_channels: int = 40,
        dropout: float = 0.2,
    ):
        super().__init__()
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = mobilenet_v3_large(weights=weights).features
        self.stem_to_s8 = backbone[:5]   # Output stride 8: 40 channels, (B, 40, 10, 10) for 80x80
        self.rest = backbone[5:]         # Output: 960 channels, (B, 960, 3, 3) for 80x80

        # Assert stride 8 shape with dummy input at initialization (eval mode preserves BatchNorm stats)
        was = self.stem_to_s8.training
        self.stem_to_s8.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 80, 80)
            dummy_s8 = self.stem_to_s8(dummy)
            assert dummy_s8.shape == (1, 40, 10, 10), (
                f"Expected stem_to_s8 output shape (1, 40, 10, 10), got {tuple(dummy_s8.shape)}"
            )
        self.stem_to_s8.train(was)

        self.head = Sequential(
            AdaptiveAvgPool2d(1),
            Flatten(),
            Linear(960, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            Linear(128, num_classes),
        )
        self.FTGenerator = FTGenerator(in_channels=ft_channels)

        # ImageNet normalization buffers (registered to follow model device and state_dict)
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor):
        # Convert [-1.0, 1.0] -> [0.0, 1.0] -> ImageNet normalized
        x = ((x + 1.0) / 2.0 - self.mean) / self.std
        f8 = self.stem_to_s8(x)
        f = self.rest(f8)
        cls = self.head(f)

        if self.training:
            ft = self.FTGenerator(f8)
            return cls, ft
        else:
            return cls


def build_model(model_cfg: dict, pretrained: bool = False) -> nn.Module:
    """
    Factory function for FAS models based on configuration dictionary.

    Args:
        model_cfg: Dictionary with at least 'arch' and 'num_classes'.
        pretrained: Whether to initialize with pretrained ImageNet weights
                    (only applicable for MobileNetV3Large-FT during training).
    """
    arch = model_cfg.get("arch")
    if arch == "MultiFTNet/MiniFASNetV2SE":
        num_classes = int(model_cfg.get("num_classes", 2))
        embedding_size = int(model_cfg.get("embedding_size", 128))
        conv6_kernel = tuple(model_cfg.get("conv6_kernel", (5, 5)))
        return MultiFTNet(
            img_channel=3,
            num_classes=num_classes,
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
        )
    elif arch == "MobileNetV3Large-FT":
        num_classes = int(model_cfg.get("num_classes", 2))
        return MobileNetV3FAS(
            num_classes=num_classes,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

