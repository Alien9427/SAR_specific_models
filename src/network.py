import torch.nn as nn
import torch.nn.init as init


class AlexNet_TSX(nn.Module):
    def __init__(self, num_class=7):
        super(AlexNet_TSX, self).__init__()
        # input image size: 160```
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, (11, 11), (4, 4)),
            nn.BatchNorm2d(96),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2), 1, 2),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(384),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.BatchNorm2d(384),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*3*3, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_class),
            # nn.LeakyReLU(),
            # nn.Dropout(0.7),
            # nn.Linear(128, num_class)
            # nn.Linear(256*3*3, num_class) # 256*3*3 is for input of 128*128, to be modified
        )
        self._initialize_weights()
        self.layer_dict = {'conv1': 'features.0', 'conv2': 'features.4', 'conv3': 'features.8', 'conv4': 'features.11',
                           'conv5': 'features.14', 'maxpool5': 'features.17', 'fc1': 'classifier.0'
            , 'fc2': 'classifier.3'#, 'fc3': 'classifier.4'
        }

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        """
            use He's initializer
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class AlexNet_OpenSAR(nn.Module):
    def __init__(self, num_class=3):
        super(AlexNet_OpenSAR, self).__init__()
        # input image size: 160```
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, (11, 11), (4, 4)),
            nn.BatchNorm2d(96),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (1, 1), (2, 2), 1, 2),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 384, (3, 3), (1, 1), (1, 1)),
            nn.BatchNorm2d(384),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(384, 384, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.BatchNorm2d(384),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.Conv2d(384, 256, (3, 3), (1, 1), (1, 1), 1, 2),
            nn.BatchNorm2d(256),
            # nn.ReLU(),
            nn.LeakyReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*2*2, num_class) # for input of 88*88
        )
        self._initialize_weights()
        self.layer_dict = {'conv1': 'features.0', 'conv2': 'features.4', 'conv3': 'features.8', 'conv4': 'features.11',
                           'conv5': 'features.14', 'maxpool5': 'features.17', 'fc1': 'classifier.0'}

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def forward_features(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        return x

    def _initialize_weights(self):
        """
            use He's initializer
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

""" ResNet """
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.5)
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
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1) # for input of 128*128
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=0.5),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet18():
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=1000)

    return model

def ResNet_TSX(tsx_num_class):
    model = ResNet(BasicBlock, [1,1,1,1], num_classes=tsx_num_class)

    return model

def ResNet18_TSX(tsx_num_class):
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=tsx_num_class)
    return model

def ResNet18_opti_rs(tsx_num_class):
    model = ResNet(BasicBlock, [2,2,2,2], num_classes=tsx_num_class)
    return model

def get_features_by_name(model, x, layer_name):
    layer_dict = model.layer_dict
    layer_num = layer_dict[layer_name]
    for i, l in enumerate(model.features):
        x = l(x)
        if 'features.' + str(i) == layer_num:
            break

    return x.data[0]