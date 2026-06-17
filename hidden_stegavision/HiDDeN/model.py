
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    HiDDeN 编码器：将秘密信息嵌入载体图片
    """
    def __init__(self, img_size=224, message_len=64):
        super(Encoder, self).__init__()
        self.img_size = img_size
        self.message_len = message_len
        
        # 秘密信息预处理层：将比特流扩展到图片尺寸
        self.message_pre = nn.Sequential(
            nn.Linear(message_len, img_size * img_size),
            nn.ReLU(),
        )
        
        # 编码器主体：CNN架构
        self.conv_layers = nn.Sequential(
            # 输入：3通道图片 + 1通道扩展后的秘密信息 = 4通道
            nn.Conv2d(4, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, cover, message):
        """
        Args:
            cover: 载体图片 [B, 3, H, W]
            message: 秘密比特流 [B, message_len]
        Returns:
            steg: 隐写图片 [B, 3, H, W]
        """
        # 扩展秘密信息到图片尺寸
        message_expanded = self.message_pre(message)
        message_expanded = message_expanded.view(-1, 1, self.img_size, self.img_size)
        
        # 拼接载体图片和扩展后的秘密信息
        x = torch.cat([cover, message_expanded], dim=1)
        
        # 通过编码器生成残差
        residual = self.conv_layers(x)
        
        # 残差加到载体图片上，裁剪到[0,1]范围
        steg = torch.clamp(cover + residual, 0, 1)
        
        return steg


class Decoder(nn.Module):
    """
    HiDDeN 解码器：从隐写图片中还原秘密信息
    """
    def __init__(self, img_size=224, message_len=64):
        super(Decoder, self).__init__()
        self.img_size = img_size
        self.message_len = message_len
        
        # 解码器主体：CNN架构
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        # 全局平均池化 + 全连接层
        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, message_len),
            nn.Sigmoid()
        )
    
    def forward(self, steg):
        """
        Args:
            steg: 隐写图片 [B, 3, H, W]
        Returns:
            decoded_message: 解码后的秘密信息 [B, message_len]
        """
        x = self.conv_layers(steg)
        decoded_message = self.fc_layers(x)
        return decoded_message


class NoiseLayer(nn.Module):
    """
    HiDDeN 噪声层：模拟信道干扰
    """
    def __init__(self, noise_std=0.01, jpeg_quality=75):
        super(NoiseLayer, self).__init__()
        self.noise_std = noise_std
        self.jpeg_quality = jpeg_quality
    
    def forward(self, x):
        """
        Args:
            x: 输入图片 [B, 3, H, W]
        Returns:
            加入噪声后的图片 [B, 3, H, W]
        """
        # 添加高斯噪声
        noise = torch.randn_like(x) * self.noise_std
        x_noisy = torch.clamp(x + noise, 0, 1)
        
        return x_noisy


class HiDDeN(nn.Module):
    """
    HiDDeN 完整网络：编码器 + 噪声层 + 解码器
    """
    def __init__(self, img_size=224, message_len=64, noise_std=0.01):
        super(HiDDeN, self).__init__()
        self.encoder = Encoder(img_size, message_len)
        self.decoder = Decoder(img_size, message_len)
        self.noise_layer = NoiseLayer(noise_std)
        self.message_len = message_len
    
    def encode(self, cover, message):
        """
        仅编码：生成隐写图片
        """
        return self.encoder(cover, message)
    
    def decode(self, steg):
        """
        仅解码：从隐写图片中还原信息
        """
        return self.decoder(steg)
    
    def forward(self, cover, message):
        """
        完整前向：编码 -> 加噪声 -> 解码
        """
        steg = self.encoder(cover, message)
        steg_noisy = self.noise_layer(steg)
        decoded_message = self.decoder(steg_noisy)
        return steg, steg_noisy, decoded_message

