import torch
import torch.nn as nn

try: from torch.hub import load_state_dict_from_url
except ImportError: from torch.utils.model_zoo import load_url as load_state_dict_from_url

torch.manual_seed(0)


# Model
# - ImageNet_resnet18, ImageNet_resnet34, ImageNet_resnet50, ImageNet_resnet101, ImageNet_resnet152
# - CIFAR10_resnet20, CIFAR10_resnet32, CIFAR10_resnet44, CIFAR10_resnet56, CIFAR10_resnet110

# Pretrained model weights url (pretrained on ImageNet)
pretrained_model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        if self.stride == 1:
            self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=self.stride, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(inplanes//2, inplanes, kernel_size=3, stride=self.stride, padding=1, bias=False)
            self.downsample = nn.Conv2d(inplanes//2, inplanes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inplanes)

    def forward(self, x):
        if self.stride == 1:
            identity = x
        else:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, stride=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, mode, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.mode = mode
        
        if self.mode == 'ImageNet':
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        elif self.mode == 'CIFAR10':
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=True)
            self.layer1 = self._make_layer(block, 16, layers[0])
            self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        self.inplanes = planes * block.expansion
        layers = [block(self.inplanes, stride=stride)]
        for _ in range(1, blocks):
            layers.append(block(self.inplanes))

        return nn.Sequential(*layers)

    def _forward_ImageNet(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def _forward_CIFAR10(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        if self.mode == 'ImageNet':
            return self._forward_ImageNet(x)
        
        elif self.mode == 'CIFAR10':
            return self._forward_CIFAR10(x)

        
# Model info
cfgs = {
    # ImageNet Model
    18 : ['ImageNet', BasicBlock, [2, 2, 2, 2]],
    34 : ['ImageNet', BasicBlock, [3, 4, 6, 3]],
    50 : ['ImageNet', Bottleneck, [3, 4, 6, 3]],
    101 : ['ImageNet', Bottleneck, [3, 4, 23, 3]],
    152 : ['ImageNet', Bottleneck, [3, 8, 36, 3]],
    
    # CIFAR-10 Model
    20 : ['CIFAR10', BasicBlock, [3, 3, 3]],
    32 : ['CIFAR10', BasicBlock, [5, 5, 5]],
    44 : ['CIFAR10', BasicBlock, [7, 7, 7]],
    56 : ['CIFAR10', BasicBlock, [9, 9, 9]],
    110 : ['CIFAR10', BasicBlock, [18, 18, 18]]
}
       

def resnet(depth, num_classes=1000, pretrained=False):
    
    model = ResNet(mode=cfgs[depth][0], block=cfgs[depth][1], layers=cfgs[depth][2], num_classes=num_classes)
    arch = 'resnet'+str(depth)
    
    if pretrained and (num_classes == 1000) and (arch in pretrained_model_urls):
        state_dict = load_state_dict_from_url(pretrained_model_urls[arch], progress=True)
        model.load_state_dict(state_dict)
    elif pretrained:
        raise ValueError('No pretrained model in resnet {} model with class number {}'.format(depth, num_classes))
            
    return model