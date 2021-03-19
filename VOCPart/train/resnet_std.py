import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import math
import torch
from torch.distributions import Bernoulli
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class LearnableMaskLayer(nn.Module):
    def __init__(self, feature_dim, num_classes):
        super(LearnableMaskLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mask = torch.nn.Parameter(torch.full((feature_dim,num_classes),0.5))

    def get_channel_mask(self):
        c_mask = self.mask
        return c_mask

    def get_density(self):
        return torch.norm(self.mask, p=1)/torch.numel(self.mask)

    def _icnn_mask(self, x, labels):
        if self.training:
            index_mask = torch.zeros(x.shape, device=x.device)
            for idx, la in enumerate(labels):
                index_mask[idx, :, :, :] = self.mask[:, la].view(-1, self.mask.shape[0], 1, 1)
            return index_mask * x
        else:
            return x

    def loss_function(self):
        l1_reg = torch.norm(self.mask, p=1)
        l1_reg = torch.relu(l1_reg - torch.numel(self.mask) * 0.2)
        return l1_reg

    def clip_lmask(self):

        lmask = self.mask
        lmask = lmask / torch.max(lmask, dim=1)[0].view(-1, 1)
        lmask = torch.clamp(lmask, min=0, max=1)
        self.mask.data = lmask

    def forward(self, x, labels, last_layer_mask=None):
        if (last_layer_mask is not None):
            self.last_layer_mask = last_layer_mask

        x = self._icnn_mask(x, labels)

        return x, self.loss_function()


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1) #  nn.AvgPool2d(32)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, labels=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        if labels is not None:
            x, reg = self.lmask(x, labels)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if labels  is not None:
            return x, reg
        else:
            return x

    def get_feature_map(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


def resnet18(pretrained=False, num_classes=1000, ifmask=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    block = BasicBlock
    model = ResNet(block, [2, 2, 2, 2], num_classes=1000, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    model.fc = nn.Linear(512 * block.expansion, num_classes)
    if ifmask:
        model.lmask = LearnableMaskLayer(feature_dim=512* block.expansion, num_classes=num_classes)
    return model



def resnet34(pretrained=False, num_classes=1000, ifmask=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    block = BasicBlock
    model = ResNet(block, [3, 4, 6, 3], num_classes=1000, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    model.fc = nn.Linear(512 * block.expansion, num_classes)
    if ifmask:
        model.lmask = LearnableMaskLayer(feature_dim=512* block.expansion, num_classes=num_classes)
    return model



def resnet50(pretrained=False, num_classes=1000, ifmask=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    block = Bottleneck
    model = ResNet(block, [3, 4, 6, 3], num_classes=1000, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    model.fc = nn.Linear(512 * block.expansion, num_classes)
    if ifmask:
        model.lmask = LearnableMaskLayer(feature_dim=512* block.expansion, num_classes=num_classes)
    return model



def resnet101(pretrained=False, num_classes=1000, ifmask=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    block = Bottleneck
    model = ResNet(block, [3, 4, 23, 3], num_classes=1000, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    model.fc = nn.Linear(512 * block.expansion, num_classes)
    if ifmask:
        model.lmask = LearnableMaskLayer(feature_dim=512* block.expansion, num_classes=num_classes)
    return model



def resnet152(pretrained=False, num_classes=1000, ifmask=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    block = Bottleneck
    model = ResNet(block, [3, 8, 36, 3],  num_classes=1000, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    model.fc = nn.Linear(512 * block.expansion, num_classes)
    if ifmask:
        model.lmask = LearnableMaskLayer(feature_dim=512* block.expansion, num_classes=num_classes)
    return model

def resent(depth, **kwargs):
    if depth == 18:
        return resnet18(**kwargs)
    elif depth == 34:
        return resnet34(**kwargs)
    elif depth == 50:
        return resnet50(**kwargs)
    elif depth == 101:
        return resnet101(**kwargs)
    elif depth == 152:
        return resnet152(**kwargs)
    else:
        raise

