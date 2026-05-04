from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNetCifar(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet32_cifar(num_classes=100):
    return ResNetCifar(BasicBlock, [5, 5, 5], num_classes=num_classes)


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    return x.view(batchsize, -1, height, width)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
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
            nn.Conv2d(inp if self.stride > 1 else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2Cifar(nn.Module):
    def __init__(self, net_size=1.0, num_classes=100):
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

    def forward(self, x):
        x = self.conv1(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)


def shufflenetv2_cifar(num_classes=100):
    return ShuffleNetV2Cifar(net_size=1.0, num_classes=num_classes)


def _replace_classifier(model: nn.Module, name: str, num_classes: int):
    if name.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("mobilenet"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name.startswith("shufflenet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported torchvision model head for {name}")
    return model


def build_model(name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    name = name.lower()
    if name in {"resnet32", "resnet32_cifar"}:
        return resnet32_cifar(num_classes=num_classes)
    if name in {"shufflenetv2", "shufflenetv2_cifar"}:
        return shufflenetv2_cifar(num_classes=num_classes)

    # Torchvision models for ImageNet-100 / Tiny-ImageNet / CUB.
    weights = None
    if pretrained:
        # Use DEFAULT weights when torchvision supports it; fall back to random if unavailable.
        try:
            weights = "DEFAULT"
        except Exception:
            weights = None
    if name == "resnet18":
        model = tvm.resnet18(weights=weights)
    elif name == "resnet34":
        model = tvm.resnet34(weights=weights)
    elif name == "mobilenet_v2":
        model = tvm.mobilenet_v2(weights=weights)
    elif name == "shufflenet_v2_x1_0":
        model = tvm.shufflenet_v2_x1_0(weights=weights)
    else:
        raise ValueError(f"Unknown model: {name}")
    return _replace_classifier(model, name, num_classes)


def build_peer_models(model_a: str, model_b: str, num_classes: int, pretrained: bool = False) -> Tuple[nn.Module, nn.Module]:
    return build_model(model_a, num_classes, pretrained), build_model(model_b, num_classes, pretrained)
