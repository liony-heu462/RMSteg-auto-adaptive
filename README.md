
# RMSteg：基于注意力流的鲁棒信息隐写
## 课程作业：自适应分辨率改进版

---

## 报告信息

| 项目 | 内容 |
|------|------|
| **报告标题** | RMSteg：基于注意力流的鲁棒信息隐写 |
| **课程作业** | 自适应分辨率改进|
| **改进方案** | 图像金字塔算法 |
| **完成时间** | 2026年6月3日 |
| **完成人** | 沈思为、张懿 |

---

## 1. 摘要

本报告包含完整的 RMSteg 基线模型复现和自适应分辨率改进。

### 1.1 三套完整体系

| 文件夹 | 功能 |
|------|------|
| `src/` | 原始 RMSteg：固定 224×224 输入 |
| `auto_224model/` | 自适应 RMSteg：支持任意分辨率 |
| `hidden_stegavision/` | 经典隐写算法对比：HiDDeN 和 StegaVision |

### 1.2 核心改进

原始 RMSteg 仅支持固定的 224×224 输入，本课程作业采用图像金字塔算法实现自适应分辨率，在多尺度上处理并融合结果。

---

## 2. 方案概述

### 2.1 为什么简单上采样/下采样无效？

简单上采样/下采样为无效改进，这是因为缩放过程会丢失信息，引入额外伪影。

### 2.2 我们的方案：图像金字塔算法

| 步骤 | 说明 |
|------|------|
| 构建金字塔 | 将任意分辨率图像分解为多个尺度 |
| 多尺度处理 | 在每个尺度上应用原始 RMSteg 网络 |
| 融合结果 | 将多尺度结果融合，恢复原始尺寸 |
| 保持基线 | 原始 RMSteg 算法不变 |

---

## 3. 项目结构

```
RMSteg-auto-adaptive/
├── src/
│   ├── model/
│   ├── util/
│   ├── test_img/
│   ├── all_test_result/
│   ├── single_test.py
│   ├── all_test.py
│   ├── evaluate.py
│   ├── train_single_gpu.py
│   ├── detect_visible_change.py
│   ├── 使用说明.md
│   ├── 性能分析报告.md
│   └── detect_visible_change_README.md
│
├── auto_224model/
│   ├── model/
│   │   └── attnflow_adaptive.py  # 自适应网络（图像金字塔）
│   ├── util/
│   ├── test_img/
│   ├── all_test_result/
│   ├── final_test.py
│   ├── all_test.py
│   ├── evaluate.py
│   ├── train_single_gpu.py
│   ├── detect_visible_change.py
│   ├── 使用说明.md
│   └── 性能分析报告.md
│
├── hidden_stegavision/
│   ├── HiDDeN/
│   │   ├── model.py          # HiDDeN 模型实现
│   │   ├── test.py           # HiDDeN 批量测试脚本
│   │   ├── util/             # 工具函数
│   │   ├── config.yaml       # 配置文件
│   │   └── result/           # 结果保存目录
│   ├── StegaVision/
│   │   ├── model.py          # StegaVision 模型实现
│   │   ├── test.py           # StegaVision 批量测试脚本
│   │   ├── util/             # 工具函数
│   │   ├── config.yaml       # 配置文件
│   │   └── result/           # 结果保存目录
│   ├── test_img/             # 测试图片文件夹
│   └── README.md             # 详细说明文档
│
└── README.md  # 本文档
```

---

## 4. 快速开始

### 4.1 使用原始 RMSteg

```bash
cd src
python single_test.py
python all_test.py
python evaluate.py
```

### 4.2 使用自适应 RMSteg

```bash
cd auto_224model
python final_test.py
python all_test.py
python evaluate.py
```

### 4.3 使用经典隐写算法对比

```bash
# 测试 HiDDeN
cd hidden_stegavision/HiDDeN
python test.py

# 测试 StegaVision
cd hidden_stegavision/StegaVision
python test.py
```

注意：HiDDeN 和 StegaVision 会自动处理 `hidden_stegavision/test_img/` 文件夹中的所有图片。

---

## 5. 两套版本对比

| 指标 | 原始版本（src） | 自适应版本（auto_224model） |
|------|----------------|------------------------------|
| **输入分辨率** | 固定 224×224 | 任意分辨率 |
| **方案** | 原始 AttnFlow | 图像金字塔算法 |
| **简单缩放** | - | 未使用简单上采样/下采样 |
| **平均 SSIM** | ~0.988 | ~0.987 |
| **灵活性** | 低 | 高 |

---

## 6. 新增功能（课程作业）

| 功能 | 说明 |
|------|------|
| 自适应分辨率 | 支持任意尺寸输入 |
| 完整评估体系 | SSIM、PSNR、LPIPS |
| 肉眼可见变化检测 | 14张已知问题图片列表 |
| 简化训练脚本 | 单卡训练，无需分布式 |
| 完整文档 | 使用说明、性能分析|

---

## 7. 结果路径说明

| 路径 | 说明 |
|------|------|
| `all_test_result/` | 批量测试完整结果 |
| `result/` | 单次测试结果 |
| `evaluation_results/` | 评估结果 |
| `checkpoints/` | 训练权重 |
| `log/` | TensorBoard 日志 |

---

## 8. 数据说明

### 8.1 测试图片

`test_img/` 文件夹包含 51张图片（0-50），包含多种分辨率。

### 8.2 肉眼可见变化的图片列表

通过实验发现，以下 14张图片在隐写后有肉眼可见的明显变化：
1, 2, 3, 5, 15, 17, 25, 28, 29, 36, 39, 40, 48, 49。

---

## 9. 引用

```
@inproceedings{rmsteg2025,
  title={Robust Message Embedding via Attention Flow-Based Steganography},
  author={...},
  booktitle={CVPR},
  year={2025}
}
```

---

## 10. 项目完成状态

 **代码**：三套完整体系
  - src 原始：固定 224×224 输入
  - auto_224model 自适应：支持任意分辨率
  - hidden_stegavision 对比：HiDDeN 和 StegaVision
 **文档**：所有文档均为正规报告格式
 **改进**：
  - 图像金字塔算法，完全符合课程作业要求
  - 新增经典隐写算法对比
 **工具**：评估、训练、检测等完整脚本
 **测试**：支持批量处理 test_img 文件夹中的所有图片

---


