import torch
import torch.nn as nn
from config import CHARSET, IMAGE_SIZE, CODE_MAX_LENGTH

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        #  shortcut 连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

class CaptchaModel(nn.Module):
    def __init__(self):
        super(CaptchaModel, self).__init__()
        self.image_width, self.image_height = IMAGE_SIZE
        self.charset_size = len(CHARSET)
        
        # 初始卷积层
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 残差块（简化架构）
        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 64)
        self.layer4 = ResidualBlock(64, 128, stride=2)
        self.layer5 = ResidualBlock(128, 128)
        
        # 计算卷积后的输出大小
        conv_output_size = self._get_conv_output_size()
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, CODE_MAX_LENGTH * self.charset_size)
        )

    def _get_conv_output_size(self):
        """计算卷积层输出的大小"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.image_height, self.image_width)
            x = self.init_conv(dummy_input)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.layer5(x)
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc_layers(x)
        x = x.view(x.size(0), CODE_MAX_LENGTH, self.charset_size)  # 重塑为 (批量大小, 6, 字符集大小)
        return x