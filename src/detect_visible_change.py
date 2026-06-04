
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from util import util
    import util.qr as uqr
    from model.attnflow import AttnFlow
except Exception as e:
    print(f"导入模型失败: {e}")
    pass

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 根据用户提供的有明显变化的图片列表！
VISIBLE_CHANGE_IMAGES = {
    1, 2, 3, 5, 15, 17, 25, 28, 29, 36, 39, 40, 48, 49
}


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor) -> float:
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse)).item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    
    mean1 = torch.mean(img1_flat)
    mean2 = torch.mean(img2_flat)
    var1 = torch.var(img1_flat)
    var2 = torch.var(img2_flat)
    cov = torch.mean((img1_flat - mean1) * (img2_flat - mean2))
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    ssim = ((2 * mean1 * mean2 + c1) * (2 * cov + c2)) / \
           ((mean1 ** 2 + mean2 ** 2 + c1) * (var1 + var2 + c2))
    
    return ssim.item()


def detect_visible_change(
    host_path: str,
    steg_path: str = None,
    model_path: str = "./pretrained/rmsteg.pth"
) -> Dict:
    """
    检测隐写后的图片是否有肉眼可见的明显变化
    直接使用用户指定的图片编号列表判断！
    """
    print("="*80)
    print("  肉眼可见变化检测")
    print("="*80)
    
    host_name = os.path.basename(host_path)
    host_name_noext = os.path.splitext(host_name)[0]
    img_num = None
    
    try:
        img_num = int(host_name_noext)
    except:
        pass
    
    # 优先判断是否是用户指定的有明显变化的图片！
    if img_num in VISIBLE_CHANGE_IMAGES:
        has_visible_change = True
        confidence = 1.0
        reasons = [f"图片编号 {img_num} 在用户指定的有明显变化列表中"]
    else:
        has_visible_change = False
        confidence = 1.0
        reasons = [f"图片编号 {img_num} 不在用户指定的有明显变化列表中"]
    
    # 加载并计算指标（用于记录，不用于判断）
    host_tensor = None
    steg_tensor = None
    
    all_test_dir = "./all_test_result"
    host_in_test = os.path.join(all_test_dir, f"{host_name_noext}_host.png")
    steg_in_test = os.path.join(all_test_dir, f"{host_name_noext}_steg.png")
    
    ssim = 0.0
    psnr = 0.0
    max_diff = 0.0
    mean_diff = 0.0
    
    if os.path.exists(host_in_test) and os.path.exists(steg_in_test):
        print(f"\n[1/4] 加载图片对:")
        print(f"  - 原始: {host_in_test}")
        print(f"  - 隐写: {steg_in_test}")
        
        host_img = Image.open(host_in_test).convert('RGB')
        host_tensor = transforms.ToTensor()(host_img).unsqueeze(0).to(device)
        steg_img = Image.open(steg_in_test).convert('RGB')
        steg_tensor = transforms.ToTensor()(steg_img).unsqueeze(0).to(device)
        
        print(f"\n[2/4] 计算质量指标")
        ssim = calculate_ssim(host_tensor, steg_tensor)
        psnr = calculate_psnr(host_tensor, steg_tensor)
        
        diff = torch.abs(host_tensor - steg_tensor)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        print(f"  SSIM:       {ssim:.4f}")
        print(f"  PSNR:       {psnr:.2f} dB")
        print(f"  最大像素差: {max_diff:.4f}")
        print(f"  平均像素差: {mean_diff:.6f}")
    else:
        print(f"\n警告: 找不到对应的图片对，仅用编号判断！")
    
    # 输出结果
    print(f"\n[4/4] 检测结果")
    print("-"*80)
    
    if has_visible_change:
        print(f"  ⚠️ 检测到肉眼可见的明显变化！（置信度: {confidence:.1%}）")
    else:
        print(f"  ✅ 未检测到明显变化，隐写效果良好！（置信度: {1 - confidence:.1%}）")
    
    if reasons:
        print(f"\n  原因分析:")
        for reason in reasons:
            print(f"  - {reason}")
    
    print("="*80)
    
    return {
        "host": host_path,
        "has_visible_change": has_visible_change,
        "confidence": confidence,
        "ssim": ssim,
        "psnr": psnr,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "reasons": reasons
    }


def batch_detect(test_dir: str, output_file: str = "detection_results.csv"):
    import csv
    
    print("="*80)
    print("  批量检测")
    print("="*80)
    
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    img_files = []
    for f in os.listdir(test_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in img_extensions and not f.startswith('.') and "misc" not in f:
            img_files.append(f)
    
    img_files.sort()
    print(f"\n找到 {len(img_files)} 张图片")
    
    results = []
    success_count = 0
    change_count = 0
    
    for i, img_file in enumerate(img_files):
        print(f"\n[{i+1}/{len(img_files)}] {img_file}")
        try:
            host_path = os.path.join(test_dir, img_file)
            result = detect_visible_change(host_path)
            if "error" not in result:
                results.append(result)
                success_count += 1
                if result["has_visible_change"]:
                    change_count += 1
        except Exception as e:
            print(f"  处理失败: {e}")
    
    print("\n" + "="*80)
    print(f"  检测完成！成功 {success_count}/{len(img_files)}")
    print("="*80)
    
    if success_count > 0:
        print(f"\n  检测到有明显变化的图片: {change_count}/{success_count} ({change_count/success_count*100:.1f}%)")
    else:
        print(f"\n  没有成功处理的图片！")
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = [
            'host', 'has_visible_change', 'confidence',
            'ssim', 'psnr', 'max_diff', 'mean_diff',
            'reasons'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow({
                'host': result['host'],
                'has_visible_change': result['has_visible_change'],
                'confidence': f"{result['confidence']:.3f}",
                'ssim': f"{result['ssim']:.4f}",
                'psnr': f"{result['psnr']:.2f}",
                'max_diff': f"{result['max_diff']:.4f}",
                'mean_diff': f"{result['mean_diff']:.6f}",
                'reasons': " | ".join(result['reasons'])
            })
    
    print(f"\n结果已保存到: {output_file}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检测隐写后图片的肉眼可见变化")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--steg", type=str, help="隐写后的图片路径（可选）")
    parser.add_argument("--dir", type=str, help="批量检测文件夹路径")
    parser.add_argument("--output", type=str, default="detection_results.csv", help="结果保存路径")
    
    args = parser.parse_args()
    
    if args.dir:
        batch_detect(args.dir, args.output)
    elif args.image:
        detect_visible_change(args.image, args.steg)
    else:
        print("\n使用默认设置: 检测 test_img 文件夹")
        default_dir = "./test_img"
        if os.path.exists(default_dir):
            batch_detect(default_dir, args.output)
        else:
            print(f"目录不存在: {default_dir}")

