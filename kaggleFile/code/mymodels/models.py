import torch.nn as nn


C5L3_MNIST = nn.Sequential(
    nn.Conv2d(1, 128, 3, padding=1),
    nn.BatchNorm2d(128),  # Batch Normalization after the first convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(128, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the second convolution
    nn.ReLU(inplace=True),
    # nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(256, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the third convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the fourth convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the fifth convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Flatten(),
    nn.Linear(256 * 7 * 7, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(128, 10),
)

C5L3_cifar10 = nn.Sequential(
    nn.Conv2d(3, 128, 3, padding=1),
    nn.BatchNorm2d(128),  # Batch Normalization after the first convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(128, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the second convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Conv2d(256, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the third convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 512, 3, padding=1),
    nn.BatchNorm2d(512),  # Batch Normalization after the fourth convolution
    nn.ReLU(inplace=True),
    
    nn.Conv2d(512, 256, 3, padding=1),
    nn.BatchNorm2d(256),  # Batch Normalization after the fifth convolution
    nn.ReLU(inplace=True),
    nn.MaxPool2d(2),
    # nn.Dropout(0.3),
    
    nn.Flatten(),
    nn.Linear(256 * 4 * 4, 512),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(256, 128),
    nn.ReLU(inplace=True),
    nn.Dropout(0.3),
    
    nn.Linear(128, 10),
)

C3L2_MNIST = nn.Sequential(
    nn.Conv2d(1, 32, 3,padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3,padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace = True),
    # nn.MaxPool2d(2),

    nn.Conv2d(64,128,3,padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(128*7*7, 128),
    nn.ReLU(inplace = True),
    nn.Dropout(p=0.2),

    nn.Linear(128, 10)
    
)

C3L2_cifar10 = nn.Sequential(
    nn.Conv2d(3, 32, 3,padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3,padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Conv2d(64,128,3,padding=1),
    nn.BatchNorm2d(128),
    nn.ReLU(inplace = True),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(128*4*4, 128),
    nn.ReLU(inplace = True),
    nn.Dropout(p=0.2),

    nn.Linear(128, 10)
    
)


# ============= resnet ==============
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, img_channels = 3):
        super(ResNet, self).__init__()
        self.img_channels = img_channels
        self.in_channels = 64
        
        # 初始卷积层 (CIFAR-10 适配)
        self.conv1 = nn.Conv2d(self.img_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 构建残差层
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        
        # 分类器
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 创建 ResNet-20 实例 (适用于 CIFAR-10)
ResNet20_MNIST =  ResNet(num_blocks=[3, 3, 3], img_channels=1)
ResNet20_cifar10 =  ResNet(num_blocks=[3, 3, 3], img_channels=3)
