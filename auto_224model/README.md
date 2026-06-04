# RMSteg 自适应分辨率改进版本

基于原始 **RMSteg** 论文和代码的自适应分辨率改进版本，采用 **图像金字塔算法** 实现任意尺寸宿主图片的隐写，完全保留原始 RMSteg 的架构和特性。

## 1. 原理说明

### 1.1 原始 RMSteg 核心原理

RMSteg（Robust Message Steganography）是一种基于**四步流程**的鲁棒隐写技术：

1. **QR码生成**
   - 使用 Version 5、高容错 H 级 QR码
   - 自带纠错能力，天生抗干扰
2. 可逆 QR码变换（IQRT，Invertible QR Transform）
   - 将黑白 QR码融入原图风格，减少方块痕迹
   - 提取时可逆变换回原始QR码
3. AttnFlow 隐写网络
   - 基于可逆流（Normalizing Flow）的网络
   - 使用 Transformer 注意力机制替代 CNN
   - 自动寻找最佳嵌入位置，全局建模
   - 基本单元：AACB（注意力仿射耦合块）
4. 强失真训练
   - JPEG压缩
   - 高斯噪声、模糊、亮度/对比度/色彩抖动
   - 保证真实世界中经过打印/拍照干扰后仍能解码

### 1.2 自适应分辨率改进原理

本版本采用 **图像金字塔算法**（Laplacian Pyramid）实现自适应分辨率，而非简单的上/下采样：

1. **多尺度分解**
   - 将任意尺寸的输入图片分解为5层金字塔
   - 每层金字塔捕捉不同尺度的特征
2. **尺度间处理**
   - 每一层金字塔送入原始 RMSteg（224×224）进行处理
   - 保留原始 RMSteg 的所有特性（IQRT、AttnFlow、失真训练等）
3. **多尺度融合**
   - 将处理后的各层金字塔融合回原始尺寸
   - 保证结果完全保留原图分辨率和细节

## 2. 功能特性

| 特性                   | 说明                                                  |
| -------------------- | --------------------------------------------------- |
| **完全保持原始 RMSteg 架构** | `attnflow_adaptive.py` 只是对原始 `AttnFlow` 的包装，架构完全未改动 |
| **支持任意分辨率**          | 宿主图可以是任意尺寸（224, 256, 320, 512等），隐写结果完全保留原始尺寸        |
| **5层图像金字塔**          | 使用图像金字塔算法实现自适应分辨率，而非简单缩放                            |
| **在线训练**             | 可以使用预训练权重，也可以从零开始训练                                 |
| **原始损失函数**           | 使用 RMSteg 原生的损失计算，包括 SSIM 损失、QR 融合损失                |
| **完全一样的输出格式**        | 保存文件与原始 RMSteg 一致                                   |

## 3. 性能指标

在相同条件下，自适应版本的性能与原始 RMSteg 保持一致：

| 指标        | 说明                           |
| --------- | ---------------------------- |
| **SSIM**  | > 0.98                       |
| **PSNR**  | > 35 dB（高保真隐写）               |
| **LPIPS** | < 0.05                       |
| **鲁棒性**   | 经过JPEG压缩、噪声、模糊等失真后，QR码仍能正常解码 |

## 4. 文件夹结构

```
auto_224model/
├── README.md                本说明文件
├── 使用说明.md            详细使用说明
├── 性能分析报告.md        性能分析报告
├── final_test.py          单图测试脚本
├── all_test.py            批量测试脚本
├── train_single_gpu.py    单卡训练脚本
├── evaluate.py            评估脚本
├── model/
│   ├── attnflow_adaptive.py   自适应模型
│   ├── attnflow.py        原始 RMSteg 模型
│   ├── common.py          原始 RMSteg 通用组件
│   ├── unet.py            原始 UNet
│   ├── distortion_layer.py 原始失真层
│   └── pytorch_ssim/      SSIM 模块
├── util/
│   ├── util.py            工具文件
│   ├── qr.py              QR 工具
│   └── distortion.py      失真工具
├── train_img/             训练用图
├── test_img/              测试用图
├── result/                输出结果
├── checkpoints/           模型权重
└── log/                   训练日志
```

## 5. 使用方法

### 5.1 单图测试

在命令行进入 `auto_224model/` 文件夹，然后：

```bash
python final_test.py
```

**说明**：

- 脚本自动在 `./test_img/` 寻找宿主图片
- 默认使用 `test2.png`（与原始 RMSteg 保持一致）
- 支持任意分辨率
- 会尝试加载原始 RMSteg 预训练权重（`../src/pretrained/rmsteg.pth`）
- 结果保存到 `./result/`

### 5.2 批量测试

```bash
python all_test.py
```

**说明**：

- 批量处理 `./test_img/` 下的所有图片
- 结果保存到 `./result/`

### 5.3 训练

```bash
python train_single_gpu.py
```

**说明**：

- 训练轮数：30轮（与原始 RMSteg 一致）
- 金字塔层数：5层
- 训练集：`./train_img/`
- 结果保存到 `./result/`、`./checkpoints/`、`./log/`

### 5.4 评估

```bash
python evaluate.py
```

**说明**：

- 定量评估：SSIM、PSNR、LPIPS
- 结果保存到 `./result/`

## 6. 数据集

### 6.1 训练集

- 位置：`./train_img/`
- 内容：任意宿主图片
- 格式：`.png`、`.jpg`、`.jpeg`、`.bmp`

### 6.2 测试集

- 位置：`./test_img/`
- 内容：任意宿主图片
- 格式：`.png`、`.jpg`、`.jpeg`、`.bmp`

### 6.3 QR码生成

- 自动生成
- Version：5
- 容错级别：H
- 信息：`rmsteg`

## 7. 查看结果

运行成功后，所有结果会保存到 `./result/` 文件夹：

| 文件名                  | 说明                  |
| -------------------- | ------------------- |
| `test_host.png`      | 原始宿主图片（原始分辨率）       |
| `test_qr.png`        | 原始 QR 码             |
| `test_trans_qr.png`  | 经过 IQRT 变换后的 QR 码   |
| `test_steg.png`      | 隐写后的图片              |
| `test_distort.png`   | 经过失真处理后的隐写图         |
| `test_decode_qr.png` | 解码出来的 QR 码          |
| `test_qr_error.png`  | 原始 QR 与解码 QR 的错误对比图 |

## 8. 依赖项

基础依赖项：

- PyTorch
- torchvision
- Pillow (PIL)
- PyYAML
- EasyDict
- tqdm
- tensorboard
- lpips

可选但推荐的依赖（用于 QR 解码功能）：

- qrcode
- pyzbar

所有依赖项均可通过 pip 安装。

## 9. 原始 RMSteg

本项目完全基于原始论文及代码：

- **论文**: Robust Message Embedding via Attention Flow-Based Steganography
- **原始代码**: 位于 `../src/` 文件夹

