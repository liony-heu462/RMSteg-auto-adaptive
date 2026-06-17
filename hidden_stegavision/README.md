
# Hidden StegoVision 项目

本项目包含两种经典的图像隐写算法实现：HiDDeN 和 StegaVision。

## 项目结构

```
hidden_stegavision/
├── HiDDeN/
│   ├── model.py          # HiDDeN 模型实现
│   ├── test.py           # HiDDeN 批量测试脚本
│   ├── util/             # 工具函数（从 src 复制）
│   ├── config.yaml       # 配置文件
│   └── result/           # 结果保存目录
├── StegaVision/
│   ├── model.py          # StegaVision 模型实现
│   ├── test.py           # StegaVision 批量测试脚本
│   ├── util/             # 工具函数（从 src 复制）
│   ├── config.yaml       # 配置文件
│   └── result/           # 结果保存目录
├── test_img/             # 测试图片文件夹
│   ├── 0.png
│   ├── 1.png
│   └── ...
└── README.md             # 本文件
```

## 算法说明

### HiDDeN (2018)

基于 CNN 的早期端到端图像隐写经典算法。

**核心特点：**
- 分为编码器、解码器、噪声层三部分
- 支持任意长度二进制信息嵌入
- 端到端可微训练

**固有缺陷：**
- 纯 CNN 感受野有限，背景存在可见伪影
- 未专门适配复杂失真，真实场景鲁棒性弱
- 嵌入容量有限

### StegaVision

在 HiDDeN 基础上改进的面向视觉保真的算法。

**核心特点：**
- 沿用编码器-解码器架构
- 使用残差块提升视觉质量
- 减少嵌入区域的明显噪点

**固有缺陷：**
- CNN 全局建模能力不足，存在轻微色彩偏移
- 未使用 QR 码自带纠错机制，重度失真下提取成功率低
- 纯色、平缓区域仍可能察觉嵌入痕迹

## 快速开始

### 测试 HiDDeN

```bash
cd HiDDeN
python test.py
```

### 测试 StegaVision

```bash
cd StegaVision
python test.py
```

## 批量测试说明

两种算法的测试脚本都会自动处理 `test_img/` 文件夹中的所有图片：

- 支持的图片格式：`.png`, `.jpg`, `.jpeg`, `.bmp`
- 结果保存在各自的 `result/` 文件夹中
- 结果文件命名格式：`{图片名称}_{类型}.png`
  - 例如：`0_host.png`, `0_steg.png`, `0_decode_qr.png`

## 依赖

- PyTorch
- torchvision
- PIL
- qrcode
- pyzbar (可选，用于 QR 码解码)

所有依赖与 src 文件夹保持一致。

## 测试图片

将测试图片放置在 `test_img/` 文件夹中，测试脚本会自动识别并处理。

## 注意事项

- 两种算法均直接实现其固有缺陷，未进行改进
- 测试脚本会遍历 test_img 文件夹中的所有图片
- 结果文件会以图片名称为前缀保存，便于区分

