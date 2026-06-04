import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import lpips
import numpy as np
from tqdm import tqdm
import sys

# 确保导入正确
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入自适应模型和工具
from model.attnflow_adaptive import AttnFlowAdaptive

try:
    from util import util
    import util.qr as uqr
except Exception as e:
    print(f"Importing util failed: {e}")
    from PIL import Image
    import numpy as np

    class SimpleUtil:
        @staticmethod
        def image_to_tensor(path):
            img = Image.open(path).convert('RGB')
            img_tensor = transforms.ToTensor()(img).unsqueeze(0)
            return img_tensor
        
        @staticmethod
        def save_image_from_tensor(tensor, path):
            img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
            img = np.clip(img, 0.0, 1.0) * 255.0
            img = img.astype(np.uint8)
            Image.fromarray(img).save(path)
        
        @staticmethod
        def get_error_map(gt, pred, num_module, img_size=224):
            red_path = None
            green_path = None
            possible_dirs = ['./test_img/misc/', '../src/test_img/misc/', './test_img/']
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    for root, dirs, files in os.walk(dir_path):
                        if 'red.png' in files:
                            red_path = os.path.join(root, 'red.png')
                        if 'green.png' in files:
                            green_path = os.path.join(root, 'green.png')
            
            if red_path is not None and green_path is not None:
                red = transforms.Resize((img_size, img_size))(SimpleUtil.image_to_tensor(red_path)).to(pred.device)
                green = transforms.Resize((img_size, img_size))(SimpleUtil.image_to_tensor(green_path)).to(pred.device)
                error = torch.abs(gt - pred)
                error = transforms.Resize((num_module, num_module))(error)
                error = transforms.Resize((img_size, img_size))(error)
                error_binary = (error > 0.2).float()
                error_map = error_binary * red + (1.0 - error_binary) * green
                return error_map
            else:
                return torch.clamp(torch.abs(gt - pred) * 10, 0, 1).repeat(1,3,1,1)
    
    util = SimpleUtil()
    
    class SimpleQr:
        @staticmethod
        def save_qr_code(save_dir, version, message):
            try:
                import qrcode
                qr = qrcode.QRCode(version=version, box_size=10, border=4)
                qr.add_data(message)
                qr.make(fit=True)
                img = qr.make_image(fill_color='black', back_color='white')
                img.save(save_dir)
            except Exception as e:
                from PIL import Image
                import numpy as np
                img = Image.new('L', (224,224), color=255)
                for i in range(0, 224, 32):
                    for j in range(0, 224, 32):
                        if (i//32 + j//32) % 2 == 0:
                            img.paste(0, (i, j, i+32, j+32))
                img.save(save_dir)
    
    uqr = SimpleQr()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_module = 37
img_size_fixed = 224


def calculate_psnr(img1, img2):
    """计算 PSNR (Peak Signal-to-Noise Ratio)"""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def evaluate_single_image(net, img_path, save_dir='./evaluation_results'):
    """评估单张图片"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载宿主图片（保持原始尺寸）
    host_image = util.image_to_tensor(img_path).to(device)
    orig_h = host_image.shape[2]
    orig_w = host_image.shape[3]
    
    # 生成二维码，缩放到与宿主图片相同大小
    qr_path = os.path.join(save_dir, 'eval_qr.png')
    uqr.save_qr_code(
        save_dir=qr_path,
        version=5,
        message='rmsteg'
    )
    qr_base = util.image_to_tensor(qr_path).to(device)
    qr_base = transforms.Resize((img_size_fixed, img_size_fixed))(qr_base)[:, :1, ...]
    qr = transforms.Resize((orig_h, orig_w))(qr_base)
    
    # 前向传播
    with torch.no_grad():
        steg, distort, decode_qr, trans_qr = net(host_image, qr)
    
    # 保存结果
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    util.save_image_from_tensor(host_image, os.path.join(save_dir, f'{img_name}_host.png'))
    util.save_image_from_tensor(qr, os.path.join(save_dir, f'{img_name}_qr.png'))
    util.save_image_from_tensor(steg, os.path.join(save_dir, f'{img_name}_steg.png'))
    util.save_image_from_tensor(torch.abs(steg - host_image) * 5, os.path.join(save_dir, f'{img_name}_diff.png'))
    util.save_image_from_tensor(trans_qr, os.path.join(save_dir, f'{img_name}_trans_qr.png'))
    util.save_image_from_tensor(distort, os.path.join(save_dir, f'{img_name}_distort.png'))
    util.save_image_from_tensor(decode_qr, os.path.join(save_dir, f'{img_name}_decode_qr.png'))
    util.save_image_from_tensor(util.get_error_map(qr_base, decode_qr, num_module=num_module), 
                                os.path.join(save_dir, f'{img_name}_qr_error.png'))
    
    return {
        'img_name': img_name,
        'orig_h': orig_h,
        'orig_w': orig_w
    }


def evaluate_folder(net, img_dir, save_dir='./evaluation_results'):
    """评估整个文件夹"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化 LPIPS 模型
    calc_lpips = lpips.LPIPS(net='vgg').to(device)
    
    # 获取所有图片
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    img_files = []
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_extensions and not file.startswith('.') and 'misc' not in file:
                img_files.append(os.path.join(root, file))
    
    # 按数字排序
    def get_sort_key(filepath):
        filename = os.path.basename(filepath)
        basename = os.path.splitext(filename)[0]
        if basename.isdigit():
            return (0, int(basename))
        return (1, basename)
    
    img_files.sort(key=get_sort_key)
    
    if len(img_files) == 0:
        print(f"在 {img_dir} 中没有找到图片")
        return
    
    print(f"找到 {len(img_files)} 张图片")
    
    # 评估所有图片
    all_metrics = []
    for img_path in tqdm(img_files, desc='评估中'):
        try:
            # 先保存图片
            eval_result = evaluate_single_image(net, img_path, save_dir=save_dir)
            
            # 单独计算指标
            host_image = util.image_to_tensor(img_path).to(device)
            orig_h = host_image.shape[2]
            orig_w = host_image.shape[3]
            
            qr_path = os.path.join(save_dir, 'eval_qr.png')
            qr_base = util.image_to_tensor(qr_path).to(device)
            qr_base = transforms.Resize((img_size_fixed, img_size_fixed))(qr_base)[:, :1, ...]
            qr = transforms.Resize((orig_h, orig_w))(qr_base)
            
            with torch.no_grad():
                steg, _, decode_qr, _ = net(host_image, qr)
            
            # 缩放到224计算指标
            host_224 = F.interpolate(host_image, size=(img_size_fixed, img_size_fixed), mode='bicubic', align_corners=True)
            steg_224 = F.interpolate(steg, size=(img_size_fixed, img_size_fixed), mode='bicubic', align_corners=True)
            
            # 先把 decode_qr 也缩放到224
            decode_qr_224 = F.interpolate(decode_qr, size=(img_size_fixed, img_size_fixed), mode='nearest')
            
            # 直接计算 SSIM、PSNR、LPIPS，不调用 calc_loss
            from model.pytorch_ssim import SSIM
            ssim_model = SSIM()
            ssim = ssim_model(steg_224, host_224).item()
            psnr = calculate_psnr(host_224, steg_224).item()
            lpips_score = calc_lpips(host_224, steg_224).reshape(-1).mean().item()
            
            metrics = {
                'img_name': eval_result['img_name'],
                'ssim': ssim,
                'psnr': psnr,
                'lpips': lpips_score,
                'orig_h': eval_result['orig_h'],
                'orig_w': eval_result['orig_w']
            }
            
            all_metrics.append(metrics)
            print(f"{metrics['img_name']} ({metrics['orig_h']}x{metrics['orig_w']}): SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}, LPIPS={metrics['lpips']:.4f}")
            
        except Exception as e:
            print(f"评估 {img_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
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
            f.write('RMSteg 自适应分辨率 - 评估报告\n')
            f.write('='*60 + '\n\n')
            f.write(f'评估图片数量: {len(all_metrics)}\n\n')
            f.write('平均指标:\n')
            f.write(f'  SSIM:  {avg_ssim:.4f}\n')
            f.write(f'  PSNR:  {avg_psnr:.2f} dB\n')
            f.write(f'  LPIPS: {avg_lpips:.4f}\n\n')
            f.write('单张图片详情:\n')
            for m in all_metrics:
                f.write(f'  {m["img_name"]} ({m["orig_h"]}x{m["orig_w"]}): SSIM={m["ssim"]:.4f}, PSNR={m["psnr"]:.2f}, LPIPS={m["lpips"]:.4f}\n')
        
        print(f'\n评估报告已保存到: {report_path}')
        
        return {
            'avg_ssim': avg_ssim,
            'avg_psnr': avg_psnr,
            'avg_lpips': avg_lpips,
            'all_metrics': all_metrics
        }


if __name__ == '__main__':
    save_dir = './evaluation_results'
    os.makedirs(save_dir, exist_ok=True)
    
    print('='*80)
    print(' RMSteg 自适应分辨率 - 评估 '.center(78, '='))
    print('='*80)
    
    # 初始化模型
    print(f'\n初始化自适应模型...')
    net = AttnFlowAdaptive(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module)
    net = net.to(device)
    
    # 尝试加载原始预训练权重
    pretrained_paths = [
        #'./pretrained/rmsteg.pth',
        #'../src/pretrained/rmsteg.pth'
        './checkpoints/rmsteg_adaptive_final.pth'
    ]
    pretrained_loaded = False
    for pp in pretrained_paths:
        if os.path.exists(pp):
            try:
                print(f"尝试加载预训练权重: {pp}")
                net.load_state_dict(torch.load(pp, map_location=device), strict=True)
                print("✅ 预训练权重加载成功！")
                pretrained_loaded = True
                break
            except Exception as e:
                print(f"加载失败: {e}")
                continue
    
    if not pretrained_loaded:
        print("⚠️ 未找到预训练权重，将使用随机初始化！")
    
    net.eval()
    
    # 找一张图片进行单张测试
    test_img_dir = './test_img'
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    test_img_path = './test_img'
    
    for file in os.listdir(test_img_dir):
        if os.path.splitext(file)[1].lower() in img_extensions and not file.startswith('.') and 'misc' not in file:
            test_img_path = os.path.join(test_img_dir, file)
            break
    
    if test_img_path is None:
        print('\n未找到测试图片！')
    else:
        print(f'\n--- 评估单张图片: {os.path.basename(test_img_path)} ---')
        evaluate_single_image(net, test_img_path, save_dir=save_dir)
        
        # 评估文件夹
        print(f'\n--- 评估文件夹 ---')
        evaluate_folder(net, test_img_dir, save_dir=save_dir)

