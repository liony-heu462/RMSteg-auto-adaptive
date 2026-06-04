
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from util.util import args
from util import util
import util.qr as uqr
from model.attnflow import AttnFlow
from model.pytorch_ssim import SSIM
from torchvision import transforms
import lpips
import numpy as np
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_module = 37
img_size = 224

# 初始化 SSIM 计算
ssim_model = SSIM()

def calculate_psnr(img1, img2):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))

def evaluate_single_image(net, img_path, qr_message='rmsteg', save_dir='./evaluation_results'):
    """评估单张图片"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载宿主图片
    host_image = util.image_to_tensor(img_path).to(device)
    host_image = transforms.Resize((img_size, img_size))(host_image)
    
    # 生成二维码
    qr_path = os.path.join(save_dir, 'eval_qr.png')
    uqr.save_qr_code(
        save_dir=qr_path,
        version=5,
        message=qr_message
    )
    qr = util.image_to_tensor(qr_path).to(device)
    qr = transforms.Resize((img_size, img_size))(qr)[:, :1, ...]
    
    # 前向传播
    with torch.no_grad():
        steg, trans_qr = net.encode(torch.cat([host_image, qr], dim=1))
        distort = net.distort(steg)
        decode_qr = net.decode(distort)
    
    # 计算指标
    ssim = ssim_model(steg, host_image).item()
    lpips_score = 0.0  # 稍后计算
    psnr = calculate_psnr(host_image, steg).item()
    
    # 保存结果
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    util.save_image_from_tensor(host_image, os.path.join(save_dir, f'{img_name}_host.png'))
    util.save_image_from_tensor(qr, os.path.join(save_dir, f'{img_name}_qr.png'))
    util.save_image_from_tensor(steg, os.path.join(save_dir, f'{img_name}_steg.png'))
    util.save_image_from_tensor(torch.abs(steg - host_image) * 5, os.path.join(save_dir, f'{img_name}_diff.png'))
    util.save_image_from_tensor(trans_qr, os.path.join(save_dir, f'{img_name}_trans_qr.png'))
    util.save_image_from_tensor(distort, os.path.join(save_dir, f'{img_name}_distort.png'))
    util.save_image_from_tensor(decode_qr, os.path.join(save_dir, f'{img_name}_decode_qr.png'))
    util.save_image_from_tensor(util.get_error_map(qr, decode_qr, num_module=num_module), 
                                os.path.join(save_dir, f'{img_name}_qr_error.png'))
    
    return {
        'ssim': ssim,
        'psnr': psnr,
        'lpips': lpips_score,
        'img_name': img_name
    }

def evaluate_folder(net, img_dir, save_dir='./evaluation_results'):
    """评估整个文件夹"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化 LPIPS 模型
    calc_lpips = lpips.LPIPS(net='vgg').to(device)
    
    # 获取所有图片
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    img_paths = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_extensions:
                img_paths.append(os.path.join(root, file))
    
    if len(img_paths) == 0:
        print(f'在 {img_dir} 中没有找到图片')
        return
    
    print(f'找到 {len(img_paths)} 张图片')
    
    # 评估所有图片
    all_metrics = []
    for img_path in tqdm(img_paths, desc='评估中'):
        try:
            metrics = evaluate_single_image(net, img_path, save_dir=save_dir)
            
            # 单独计算 LPIPS
            host_image = util.image_to_tensor(img_path).to(device)
            host_image = transforms.Resize((img_size, img_size))(host_image)
            qr_path = os.path.join(save_dir, 'eval_qr.png')
            qr = util.image_to_tensor(qr_path).to(device)
            qr = transforms.Resize((img_size, img_size))(qr)[:, :1, ...]
            with torch.no_grad():
                steg, _ = net.encode(torch.cat([host_image, qr], dim=1))
            lpips_score = calc_lpips(host_image, steg).reshape(-1).mean().item()
            metrics['lpips'] = lpips_score
            
            all_metrics.append(metrics)
            print(f'{metrics["img_name"]}: SSIM={metrics["ssim"]:.4f}, PSNR={metrics["psnr"]:.2f}, LPIPS={metrics["lpips"]:.4f}')
        except Exception as e:
            print(f'评估 {img_path} 时出错: {e}')
    
    # 计算平均指标
    if len(all_metrics) > 0:
        avg_ssim = np.mean([m['ssim'] for m in all_metrics])
        avg_psnr = np.mean([m['psnr'] for m in all_metrics])
        avg_lpips = np.mean([m['lpips'] for m in all_metrics])
        
        print('\n' + '='*60)
        print('平均指标:')
        print(f'  SSIM:  {avg_ssim:.4f}')
        print(f'  PSNR:  {avg_psnr:.2f} dB')
        print(f'  LPIPS: {avg_lpips:.4f}')
        print('='*60)
        
        # 保存结果到文件
        report_path = os.path.join(save_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('='*60 + '\n')
            f.write('RMSteg 评估报告\n')
            f.write('='*60 + '\n\n')
            f.write(f'评估图片数量: {len(all_metrics)}\n\n')
            f.write('平均指标:\n')
            f.write(f'  SSIM:  {avg_ssim:.4f}\n')
            f.write(f'  PSNR:  {avg_psnr:.2f} dB\n')
            f.write(f'  LPIPS: {avg_lpips:.4f}\n\n')
            f.write('单张图片详情:\n')
            for m in all_metrics:
                f.write(f'  {m["img_name"]}: SSIM={m["ssim"]:.4f}, PSNR={m["psnr"]:.2f}, LPIPS={m["lpips"]:.4f}\n')
        
        print(f'\n评估报告已保存到: {report_path}')
        
        return {
            'avg_ssim': avg_ssim,
            'avg_psnr': avg_psnr,
            'avg_lpips': avg_lpips,
            'all_metrics': all_metrics
        }

if __name__ == '__main__':
    # 加载模型
    print('加载模型...')
    net = AttnFlow(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module)
    net.load_state_dict(torch.load('./pretrained/rmsteg.pth', map_location=device))
    net = net.to(device)
    net.eval()
    print('模型加载成功!')
    
    # 找一张图片进行单张测试
    test_img_dir = './test_img'
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    test_img_path = None
    
    for file in os.listdir(test_img_dir):
        if os.path.splitext(file)[1].lower() in img_extensions:
            test_img_path = os.path.join(test_img_dir, file)
            break
    
    if test_img_path is None:
        print('未找到测试图片!')
    else:
        print(f'\n--- 评估单张图片: {os.path.basename(test_img_path)} ---')
        evaluate_single_image(net, test_img_path, save_dir='./evaluation_results')
        
        # 评估文件夹（如果有 test_img 下的多个图片）
        print('\n--- 评估文件夹 ---')
        evaluate_folder(net, './test_img', save_dir='./evaluation_results')

