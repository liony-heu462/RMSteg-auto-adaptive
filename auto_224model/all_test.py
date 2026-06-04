
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
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
            # 尝试找 red 和 green 图片
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
                # 简单 resize，和原始一致！
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
train_epochs = 500


def main():
    save_dir = './all_test_result'
    os.makedirs(save_dir, exist_ok=True)
    
    print('='*80)
    print(' RMSteg 自适应分辨率 - 批量测试 '.center(78, '='))
    print('='*80)
    
    # 获取 test_img 文件夹里的所有图片
    test_img_dir = './test_img'
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    # 按数字顺序排序（0.png, 1.png, 2.png...）
    img_files = []
    for filename in os.listdir(test_img_dir):
        if os.path.splitext(filename)[1].lower() in img_extensions and not filename.startswith('.') and 'misc' not in filename:
            img_files.append(filename)
    
    def get_sort_key(filename):
        basename = os.path.splitext(filename)[0]
        if basename.isdigit():
            return (0, int(basename))
        return (1, basename)
    
    img_files.sort(key=get_sort_key)
    
    print(f'\n找到 {len(img_files)} 张图片:')
    for i, f in enumerate(img_files[:10]):
        print(f'  {i+1}. {f}')
    if len(img_files) > 10:
        print(f'  ... 还有 {len(img_files)-10} 张')
    print()
    
    # ========== 生成一次通用 QR 码 ==========
    qr_path = os.path.join(save_dir, 'common_qr.png')
    uqr.save_qr_code(
        save_dir=qr_path,
        version=5,
        message='rmsteg'
    )
    qr_base = util.image_to_tensor(qr_path).to(device)
    qr_base = transforms.Resize((img_size_fixed, img_size_fixed))(qr_base)[:, :1, ...]
    
    # ========== 初始化并准备模型（全局只做一次！）==========
    print(f"\n初始化自适应模型...")
    net = AttnFlowAdaptive(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module)
    net = net.to(device)
    
    # 尝试加载原始预训练权重
    pretrained_paths = [
        # './pretrained/rmsteg.pth',
        #'../src/pretrained/rmsteg.pth'
        './checkpoints/rmsteg_adaptive_final.pth'
    ]
    pretrained_loaded = False
    use_pretrained = False
    for pp in pretrained_paths:
        if os.path.exists(pp):
            try:
                print(f"尝试加载预训练权重: {pp}")
                net.load_state_dict(torch.load(pp, map_location=device), strict=True)
                print("✅ 预训练权重加载成功！")
                pretrained_loaded = True
                use_pretrained = True
                net.eval()
                break
            except Exception as e:
                print(f"加载失败: {e}")
                continue
    
    # ========== 如果没有预训练，先在第一张图上训练一次！ ==========
    if not use_pretrained and len(img_files) > 0:
        print("⚠️ 未使用预训练权重，先在第一张图上训练一次...")
        host_image_path = os.path.join(test_img_dir, img_files[0])
        host_image = util.image_to_tensor(host_image_path).to(device)
        
        target_size_h = host_image.shape[2]
        target_size_w = host_image.shape[3]
        
        qr = transforms.Resize((target_size_h, target_size_w))(qr_base)
        
        net.train()
        
        # 优化器
        optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999))
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.5)
        
        # 在线训练（只在第一张图上训练一次）
        print('='*80)
        print('在线训练'.center(78, '='))
        print('='*80)
        
        for epoch in range(1, train_epochs + 1):
            optimizer.zero_grad()
            
            steg, distort, decode_qr, trans_qr = net(host_image, qr)
            
            steg_loss, ssim_loss, qr_loss, qr_fusion_loss, refine_qr = net.calc_loss(
                host_image, steg, qr, decode_qr, trans_qr
            )
            
            total_loss = steg_loss + ssim_loss + qr_loss + qr_fusion_loss
            
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            print(f"Epoch {epoch:4d} | Loss: {total_loss.item():.4f}")
    else:
        print("✅ 直接用预训练权重推理！")
    
    # ========== 现在开始循环处理所有图片！ ==========
    print('\n' + '='*80)
    print('开始批量处理'.center(78, '='))
    print('='*80)
    
    net.eval()
    for idx, img_file in enumerate(img_files):
        try:
            img_path = os.path.join(test_img_dir, img_file)
            
            host_image = util.image_to_tensor(img_path).to(device)
            orig_h = host_image.shape[2]
            orig_w = host_image.shape[3]
            
            print(f'\n[{idx+1}/{len(img_files)}] {img_file} ({orig_h}x{orig_w})')
            
            # 将二维码缩放到与宿主图片相同大小
            qr = transforms.Resize((orig_h, orig_w))(qr_base)
            
            # 前向传播
            with torch.no_grad():
                steg, distort, decode_qr, trans_qr = net(host_image, qr)
            
            # 保存结果，按编号命名
            prefix = f'{idx}'
            util.save_image_from_tensor(host_image, os.path.join(save_dir, f'{prefix}_host.png'))
            util.save_image_from_tensor(qr, os.path.join(save_dir, f'{prefix}_qr.png'))
            util.save_image_from_tensor(steg, os.path.join(save_dir, f'{prefix}_steg.png'))
            util.save_image_from_tensor(torch.abs(steg - host_image), os.path.join(save_dir, f'{prefix}_steg_res.png'))
            util.save_image_from_tensor(trans_qr, os.path.join(save_dir, f'{prefix}_trans_qr.png'))
            util.save_image_from_tensor(distort, os.path.join(save_dir, f'{prefix}_distort.png'))
            util.save_image_from_tensor(decode_qr, os.path.join(save_dir, f'{prefix}_decode_qr.png'))
            util.save_image_from_tensor(util.get_error_map(qr, decode_qr, num_module=num_module), 
                                        os.path.join(save_dir, f'{prefix}_qr_error.png'))
            
            print(f'  结果保存成功！')
            
        except Exception as e:
            print(f'  处理 {img_file} 时出错: {e}')
            import traceback
            traceback.print_exc()
    
    print('\n' + '='*80)
    print(f'处理完成! 结果已保存到: {save_dir}')
    print('='*80)


if __name__ == '__main__':
    main()

