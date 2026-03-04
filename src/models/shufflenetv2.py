import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__()
        self.stride = stride
        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),

            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),

            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i: int, o: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    def __init__(self, net_size: float = 1.0, num_classes: int = 100):
        super().__init__()
        out_channels = {
            0.5: [48, 96, 192, 1024],
            1.0: [116, 232, 464, 1024],
            1.5: [176, 352, 704, 1024],
            2.0: [244, 488, 976, 2048],
        }
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = out_channels[net_size]

        input_channel = 24
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU(inplace=True),
        )

        stages = []
        for repeats, output_channel in zip(self.stage_repeats, self.stage_out_channels[:-1]):
            seq = [InvertedResidual(input_channel, output_channel, 2)]
            for _ in range(repeats - 1):
                seq.append(InvertedResidual(output_channel, output_channel, 1))
            stages.append(nn.Sequential(*seq))
            input_channel = output_channel
        self.stages = nn.Sequential(*stages)

        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channel, self.stage_out_channels[-1], 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.stage_out_channels[-1]),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(self.stage_out_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def shufflenetv2(num_classes: int = 100) -> ShuffleNetV2:
    return ShuffleNetV2(net_size=1.0, num_classes=num_classes)