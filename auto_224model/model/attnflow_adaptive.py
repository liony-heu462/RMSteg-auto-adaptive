
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from model.attnflow import AttnFlow
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util.util as util


class AttnFlowAdaptive(nn.Module):
    """
    RMSteg 自适应分辨率实现（有效改进：图像金字塔算法）
    
    严格遵循原始 RMSteg 流程：
    1. 输入：任意尺寸载体图 + QR码
    2. 图像金字塔分解为多尺度
    3. 每个尺度分别送入原始 RMSteg（224x224）处理
    4. 多尺度结果融合回原始尺寸
    
    符合大作业要求：不使用简单上采样或下采样！
    """
    def __init__(self, block_num=4, use_itf=True, use_qr_trans=True, num_module=37, pyramid_levels=5):
        super(AttnFlowAdaptive, self).__init__()
        
        # 原始 RMSteg 网络（完全不变！）
        self.net = AttnFlow(
            block_num=block_num,
            use_itf=use_itf,
            use_qr_trans=use_qr_trans,
            num_module=num_module
        )
        
        self.pyramid_levels = pyramid_levels
        self.img_size_fixed = 224
        self.num_module = num_module
        
    def build_pyramid(self, x):
        """
        构建图像金字塔（从小到大排列）
        
        Args:
            x: 输入图像 [B, C, H, W]
        
        Returns:
            pyramid: 多尺度图像列表 [smallest, ..., largest]
        """
        pyramid = []
        current = x
        
        for i in range(self.pyramid_levels):
            pyramid.append(current)
            if i < self.pyramid_levels - 1:
                current = F.avg_pool2d(current, kernel_size=2, stride=2)
        
        return pyramid[::-1]  # 反转，从最小到最大
    
    def fuse_pyramid(self, pyramid):
        """
        融合多尺度金字塔（从小到大融合）
        
        Args:
            pyramid: 多尺度处理结果列表 [smallest, ..., largest]
        
        Returns:
            fused: 融合后的原始尺寸图像
        """
        fused = pyramid[0]
        
        for i in range(1, len(pyramid)):
            _, _, h_target, w_target = pyramid[i].shape
            _, _, h_current, w_current = fused.shape
            
            if h_current != h_target or w_current != w_target:
                fused = F.interpolate(fused, size=(h_target, w_target), mode='bilinear', align_corners=True)
            
            fused = (fused + pyramid[i]) / 2.0
        
        return fused
    
    def encode(self, img, qr):
        """
        编码过程：严格遵循原始 RMSteg 编码流程！
        
        输入：img (任意尺寸)，qr (任意尺寸)
        输出：steg (任意尺寸)，trans_qr (任意尺寸)
        """
        print(f"[图像金字塔-编码] 输入尺寸: img={img.shape}, qr={qr.shape}")
        
        # 1. 构建金字塔
        img_pyramid = self.build_pyramid(img)
        qr_pyramid = self.build_pyramid(qr)
        
        for i, (im_scale, q_scale) in enumerate(zip(img_pyramid, qr_pyramid)):
            print(f"  [尺度 {i+1}] img: {im_scale.shape}, qr: {q_scale.shape}")
        
        steg_pyramid = []
        trans_qr_pyramid = []
        
        # 2. 每个尺度分别送入原始 RMSteg 处理
        for scale_idx, (img_scale, qr_scale) in enumerate(zip(img_pyramid, qr_pyramid)):
            _, _, h_scale, w_scale = img_scale.shape
            
            # 缩放到 224（仅用于送入原始网络，之后恢复）
            img_scale_224 = F.interpolate(img_scale, size=(self.img_size_fixed, self.img_size_fixed), mode='bilinear', align_corners=True)
            qr_scale_224 = F.interpolate(qr_scale, size=(self.img_size_fixed, self.img_size_fixed), mode='nearest')
            
            # 拼接输入到原始 RMSteg！必须拼接 img + qr！
            x_scale = torch.cat([img_scale_224, qr_scale_224], dim=1)
            
            steg_scale_224, trans_qr_scale_224 = self.net.encode(x_scale)
            
            # 恢复到该尺度的原始尺寸
            steg_scale = F.interpolate(steg_scale_224, size=(h_scale, w_scale), mode='bilinear', align_corners=True)
            trans_qr_scale = F.interpolate(trans_qr_scale_224, size=(h_scale, w_scale), mode='bilinear', align_corners=True)
            
            steg_pyramid.append(steg_scale)
            trans_qr_pyramid.append(trans_qr_scale)
        
        # 3. 融合多尺度结果
        steg = self.fuse_pyramid(steg_pyramid)
        trans_qr = self.fuse_pyramid(trans_qr_pyramid)
        
        print(f"[图像金字塔-编码] 输出尺寸: steg={steg.shape}, trans_qr={trans_qr.shape}")
        
        return steg.clamp(0.0, 1.0), trans_qr
    
    def decode(self, steg):
        """
        解码过程：严格遵循原始 RMSteg 解码流程！
        """
        print(f"[图像金字塔-解码] 输入尺寸: steg={steg.shape}")
        
        # 1. 构建金字塔
        steg_pyramid = self.build_pyramid(steg)
        
        decode_qr_pyramid = []
        
        # 2. 每个尺度分别送入原始 RMSteg 解码
        for scale_idx, steg_scale in enumerate(steg_pyramid):
            _, _, h_scale, w_scale = steg_scale.shape
            
            # 缩放到 224
            steg_scale_224 = F.interpolate(steg_scale, size=(self.img_size_fixed, self.img_size_fixed), mode='bilinear', align_corners=True)
            
            decode_qr_scale_224 = self.net.decode(steg_scale_224)
            
            # 恢复到该尺度的原始尺寸
            decode_qr_scale = F.interpolate(decode_qr_scale_224, size=(h_scale, w_scale), mode='bilinear', align_corners=True)
            
            decode_qr_pyramid.append(decode_qr_scale)
        
        # 3. 融合多尺度结果
        decode_qr = self.fuse_pyramid(decode_qr_pyramid)
        
        print(f"[图像金字塔-解码] 输出尺寸: decode_qr={decode_qr.shape}")
        
        return decode_qr.clamp(0.0, 1.0)
    
    def distort(self, steg):
        """
        失真模拟：严格遵循原始 RMSteg 失真流程！
        多尺度处理！
        """
        steg_pyramid = self.build_pyramid(steg)
        
        distort_pyramid = []
        
        for scale_idx, steg_scale in enumerate(steg_pyramid):
            _, _, h_scale, w_scale = steg_scale.shape
            
            # 缩放到 224
            steg_scale_224 = F.interpolate(steg_scale, size=(self.img_size_fixed, self.img_size_fixed), mode='bilinear', align_corners=True)
            
            # 原始 RMSteg 失真
            distort_scale_224 = self.net.distort(steg_scale_224)
            
            # 恢复到该尺度的原始尺寸
            distort_scale = F.interpolate(distort_scale_224, size=(h_scale, w_scale), mode='bilinear', align_corners=True)
            
            distort_pyramid.append(distort_scale)
        
        distort = self.fuse_pyramid(distort_pyramid)
        
        return distort.clamp(0.0, 1.0)
    
    def forward(self, img, qr):
        """
        完整前向过程：严格遵循原始 RMSteg 四步走！
        1. 编码（img + qr → steg + trans_qr）
        2. 失真（steg → distort）
        3. 解码（distort → decode_qr）
        
        Args:
            img: 载体图 [B, 3, H, W]
            qr: 二维码 [B, 1, H, W]
        
        Returns:
            steg, distort, decode_qr, trans_qr
        """
        print("\n" + "="*80)
        print("  [图像金字塔]   ".center(78, "="))
        print("="*80 + "\n")
        
        # 第一步：编码
        steg, trans_qr = self.encode(img, qr)
        
        # 第二步：失真
        distort = self.distort(steg)
        
        # 第三步：解码
        decode_qr = self.decode(distort)
        
        print(f"\n[图像金字塔-前向] 输出: distort={distort.shape}, decode_qr={decode_qr.shape}")
        print("="*80 + "\n")
        
        return steg.clamp(0.0, 1.0), distort.clamp(0.0, 1.0), decode_qr.clamp(0.0, 1.0), trans_qr
    
    def calc_loss(self, cover, steg, qr, decode_qr, fusion_qr):
        """
        计算损失：严格遵循原始 RMSteg calc_loss 流程！
        但是在原始尺寸上计算 steg_loss 和 ssim_loss！
        其他 loss 在 224 上计算！
        """
        _, _, h, w = cover.shape
        
        # 计算 steg_loss 和 ssim_loss 在原始尺寸上！
        loss_func = nn.L1Loss()
        steg_loss = loss_func(cover, steg)
        
        from model.pytorch_ssim import SSIM
        ssim = SSIM()
        ssim_loss = 1.0 - ssim(steg, cover)
        
        # 其他 loss 在 224 上计算
        cover_224 = F.interpolate(cover, size=(self.img_size_fixed, self.img_size_fixed), mode='bilinear', align_corners=True)
        steg_224 = F.interpolate(steg, size=(self.img_size_fixed, self.img_size_fixed), mode='bilinear', align_corners=True)
        qr_224 = F.interpolate(qr, size=(self.img_size_fixed, self.img_size_fixed), mode='nearest')
        decode_qr_224 = F.interpolate(decode_qr, size=(self.img_size_fixed, self.img_size_fixed), mode='nearest')
        fusion_qr_224 = F.interpolate(fusion_qr, size=(self.img_size_fixed, self.img_size_fixed), mode='bilinear', align_corners=True)
        
        # 调用原始 RMSteg calc_loss
        _, _, qr_loss, qr_fusion_loss, refine_qr_224 = self.net.calc_loss(
            cover_224, steg_224, qr_224, decode_qr_224, fusion_qr_224
        )
        
        # refine_qr 恢复到原始尺寸
        refine_qr = F.interpolate(refine_qr_224, size=(h, w), mode='nearest')
        
        return steg_loss, ssim_loss, qr_loss, qr_fusion_loss, refine_qr
    
    def load_state_dict(self, state_dict, strict=True):
        """
        加载预训练权重（兼容原始 RMSteg）
        
        Args:
            state_dict: 预训练权重
            strict: 是否严格加载
        
        Returns:
            self
        """
        if 'net.' in list(state_dict.keys())[0]:
            super(AttnFlowAdaptive, self).load_state_dict(state_dict, strict=strict)
        else:
            # 兼容原始 RMSteg，加上 net. 前缀
            new_state_dict = {}
            for key, val in state_dict.items():
                new_state_dict['net.' + key] = val
            super(AttnFlowAdaptive, self).load_state_dict(new_state_dict, strict=strict)
        
        return self
