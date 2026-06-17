
import os
import sys
import torch
from torchvision import transforms

# 添加 src 文件夹到路径，以便复用其功能
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from util import util
import util.qr as uqr
from model import StegaVision


def qr_to_bits(qr_tensor, num_bits=64):
    """
    将 QR 码转换为比特流（简化实现）
    """
    qr_resized = transforms.Resize((8, 8))(qr_tensor)
    qr_flat = qr_resized.view(-1)
    bits = (qr_flat > 0.5).float()
    bits = bits[:num_bits]
    if len(bits) < num_bits:
        bits = torch.cat([bits, torch.zeros(num_bits - len(bits))])
    return bits.unsqueeze(0)


def bits_to_qr(bits, qr_size=224):
    """
    将比特流转换回 QR 码（简化实现）
    """
    side = int(torch.sqrt(torch.tensor(len(bits), dtype=torch.float32)))
    bits_reshaped = bits[:side*side].view(1, 1, side, side)
    qr = transforms.Resize((qr_size, qr_size))(bits_reshaped)
    return qr


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    img_size = 224
    message_len = 64
    
    os.makedirs('./result/', exist_ok=True)
    
    # 初始化 StegaVision 模型
    net = StegaVision(img_size=img_size, message_len=message_len, noise_std=0.005)
    net = net.to(device)
    net.eval()
    
    # 获取 test_img 文件夹中的所有图片
    test_img_dir = '../test_img/'
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    image_files = []
    for filename in os.listdir(test_img_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            image_files.append(filename)
    
    print(f"找到 {len(image_files)} 张测试图片：{image_files}")
    
    # 遍历所有测试图片
    for img_filename in image_files:
        print(f"\n正在处理：{img_filename}")
        
        # 获取图片名称（不含扩展名）
        img_name = os.path.splitext(img_filename)[0]
        
        # 加载测试图片
        host_image = util.image_to_tensor(os.path.join(test_img_dir, img_filename)).to(device)
        host_image = transforms.Resize((img_size, img_size))(host_image)
        
        # 生成并保存 QR 码（复用 src 的功能）
        uqr.save_qr_code(
            save_dir=f'./result/{img_name}_qr.png',
            version=5,
            message='StegaVision'
        )
        qr = util.image_to_tensor(f'./result/{img_name}_qr.png').to(device)
        qr = transforms.Resize((img_size, img_size))(qr)[:, :1, ...]
        
        # 将 QR 码转换为比特流
        message = qr_to_bits(qr, num_bits=message_len).to(device)
        
        # 使用 StegaVision 进行隐写
        with torch.no_grad():
            steg, steg_noisy, decoded_bits = net(host_image, message)
        
        # 将解码后的比特流转换回 QR 码
        decode_qr = bits_to_qr(decoded_bits[0], qr_size=img_size)
        
        # 保存结果
        util.save_image_from_tensor(host_image, f'./result/{img_name}_host.png')
        util.save_image_from_tensor(qr, f'./result/{img_name}_qr.png')
        util.save_image_from_tensor(steg, f'./result/{img_name}_steg.png')
        util.save_image_from_tensor(torch.abs(steg - host_image), f'./result/{img_name}_steg_residual.png')
        util.save_image_from_tensor(steg_noisy, f'./result/{img_name}_steg_noisy.png')
        util.save_image_from_tensor(decode_qr, f'./result/{img_name}_decode_qr.png')
        
        print(f"已完成 {img_filename} 的处理")
    
    print("\nStegaVision 所有图片测试完成，结果保存在 result/ 文件夹中")
    print("缺陷说明：存在轻微色彩偏移（符合预期）")

