
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time
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
            possible_dirs = ['./train_img/misc/', '../src/test_img/misc/', './test_img/', './train_img/']
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


task_name = 'rmsteg_adaptive'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_module = 37
img_size_fixed = 224
train_epochs = 30  # 和 src/train.py 的 epoch_num: 30 保持一致！


def main():
    os.makedirs('./result/', exist_ok=True)
    os.makedirs('./log/', exist_ok=True)
    os.makedirs('./checkpoints/', exist_ok=True)
    
    print('='*80)
    print(' RMSteg 自适应分辨率版本 - 单卡训练'.center(78, '='))
    print('='*80)
    print(f"\n设置：训练轮数 = {train_epochs}，金字塔层数 = 5，训练集 = ./train_img/")
    
    # 找一张图片作为训练用
    train_img_dir = './train_img'
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    host_image_path = None
    
    for file in os.listdir(train_img_dir):
        if os.path.splitext(file)[1].lower() in img_extensions and not file.startswith('.') and 'misc' not in file:
            host_image_path = os.path.join(train_img_dir, file)
            break
    
    if host_image_path is None:
        print("找不到图片，使用随机生成的图片！")
        host_image = torch.rand(1, 3, 512, 512).to(device)
    else:
        host_image = util.image_to_tensor(host_image_path).to(device)
        print(f"使用宿主图片: {os.path.basename(host_image_path)}")
        print(f"原始尺寸: {host_image.shape[2]}x{host_image.shape[3]}")
    
    # 自适应：直接用原始尺寸！不缩放到224！
    target_size_h = host_image.shape[2]
    target_size_w = host_image.shape[3]
    print(f"自适应尺寸: {target_size_h}x{target_size_w}")
    
    # 生成二维码，和原始一致
    qr_path = './result/train_qr.png'
    uqr.save_qr_code(
        save_dir=qr_path,
        version=5,
        message='rmsteg'
    )
    qr = util.image_to_tensor(qr_path).to(device)
    qr = transforms.Resize((img_size_fixed, img_size_fixed))(qr)[:, :1, ...]
    qr_train = transforms.Resize((target_size_h, target_size_w))(qr)
    
    # 创建自适应模型（金字塔层数 = 5）
    print(f"\n初始化自适应模型...（金字塔层数 = 5）")
    net = AttnFlowAdaptive(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module, pyramid_levels=5)
    net = net.to(device)
    
    # 尝试加载原始预训练权重
    pretrained_paths = [
        '../src/pretrained/rmsteg.pth',
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
        print("⚠️ 未使用预训练权重，从零开始训练...")
    
    net.train()
    
    # 优化器和学习率调度器（和原始类似）
    optimizer = optim.Adam(net.parameters(), lr=1e-5, betas=(0.9, 0.999))
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1.0)
    
    # TensorBoard
    writer = SummaryWriter(log_dir=f'log/{task_name}/')
    
    # ========== 在线训练 ==========
    print('='*80)
    print(f'在线训练（{train_epochs}轮，和原始 RMSteg 一致）'.center(78, '='))
    print('='*80)
    
    for epoch in range(1, train_epochs + 1):
        optimizer.zero_grad()
        
        # 前向：严格遵循原始 RMSteg 流程
        steg, distort, decode_qr, trans_qr = net(host_image, qr_train)
        
        # 计算损失
        steg_loss, ssim_loss, qr_loss, qr_fusion_loss, refine_qr = net.calc_loss(
            host_image, steg, qr, decode_qr, trans_qr
        )
        
        # 总损失
        total_loss = steg_loss + ssim_loss + qr_loss + qr_fusion_loss
        
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # 记录到 TensorBoard
        writer.add_scalar('Loss/Steg Loss', steg_loss.item(), epoch)
        writer.add_scalar('Loss/QR Loss', qr_loss.item(), epoch)
        writer.add_scalar('Loss/QR Fusion Loss', qr_fusion_loss.item(), epoch)
        writer.add_scalar('Metrices/SSIM', 1.0 - ssim_loss.item(), epoch)
        
        # 打印
        print(f"Epoch {epoch:2d}/{train_epochs} | Loss: {total_loss.item():.4f} "
              f"(Steg: {steg_loss.item():.4f}, QR: {qr_loss.item():.4f}, SSIM: {1.0 - ssim_loss.item():.4f})")
    
    # ========== 保存最终结果 ==========
    print('='*80)
    print('保存最终结果'.center(78, '='))
    print('='*80)
    
    net.eval()
    with torch.no_grad():
        steg, distort, decode_qr, trans_qr = net(host_image, qr_train)
    
    # 保存图片（按原始格式命名）
    util.save_image_from_tensor(host_image, './result/final_host.png')
    util.save_image_from_tensor(qr_train, './result/final_qr.png')
    util.save_image_from_tensor(steg, './result/final_steg.png')
    util.save_image_from_tensor(torch.abs(steg - host_image), './result/final_steg_res.png')
    util.save_image_from_tensor(trans_qr, './result/final_trans_qr.png')
    util.save_image_from_tensor(distort, './result/final_distort.png')
    util.save_image_from_tensor(decode_qr, './result/final_decode_qr.png')
    util.save_image_from_tensor(util.get_error_map(qr, decode_qr, num_module=num_module), './result/final_qr_error.png')
    
    # 保存最终模型
    torch.save(net.state_dict(), f'./checkpoints/{task_name}_final.pth')
    
    print(f"\n训练完成！")
    print(f"最终模型已保存到: ./checkpoints/{task_name}_final.pth")
    print(f"结果图片已保存到: ./result/")
    print(f"TensorBoard 日志已保存到: ./log/{task_name}/")
    print('='*80)


if __name__ == '__main__':
    main()
