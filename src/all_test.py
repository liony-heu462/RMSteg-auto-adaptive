
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from util.util import args
from util import util
import util.qr as uqr
from model.attnflow import AttnFlow
from torchvision import transforms
from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_module = 37
img_size = 224

if __name__ == '__main__':
    save_dir = './all_test_result'
    os.makedirs(save_dir, exist_ok=True)
    
    print('加载模型...')
    net = AttnFlow(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module)
    net.load_state_dict(torch.load('./pretrained/rmsteg.pth', map_location=device))
    net = net.to(device)
    net.eval()
    print('模型加载成功!\n')
    
    # 获取 test_img 文件夹里所有的图片
    test_img_dir = './test_img'
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']
    
    # 按数字顺序排序（0.png, 1.png, 2.png...）
    img_files = []
    for file in os.listdir(test_img_dir):
        if os.path.splitext(file)[1].lower() in img_extensions and not file.startswith('.'):
            img_files.append(file)
    
    # 尝试按数字排序
    def get_sort_key(filename):
        basename = os.path.splitext(filename)[0]
        if basename.isdigit():
            return (0, int(basename))
        return (1, basename)
    
    img_files.sort(key=get_sort_key)
    
    print(f'找到 {len(img_files)} 张图片:')
    for i, f in enumerate(img_files[:10]):
        print(f'  {i+1}. {f}')
    if len(img_files) > 10:
        print(f'  ... 还有 {len(img_files)-10} 张')
    print()
    
    # 生成一次二维码，重复使用
    qr_path = os.path.join(save_dir, 'common_qr.png')
    uqr.save_qr_code(
        save_dir=qr_path,
        version=5,
        message='rmsteg'
    )
    qr_base = util.image_to_tensor(qr_path).to(device)
    qr_base = transforms.Resize((img_size, img_size))(qr_base)[:, :1, ...]
    
    # 处理每张图片
    all_metrics = []
    
    for idx, img_file in enumerate(tqdm(img_files, desc='处理图片中')):
        try:
            img_path = os.path.join(test_img_dir, img_file)
            
            # 加载宿主图片
            host_image = util.image_to_tensor(img_path).to(device)
            host_image = transforms.Resize((img_size, img_size))(host_image)
            
            # 使用同一个二维码
            qr = qr_base.clone()
            
            # 前向传播
            with torch.no_grad():
                steg, trans_qr = net.encode(torch.cat([host_image, qr], dim=1))
                distort = net.distort(steg)
                decode_qr = net.decode(distort)
            
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
            
        except Exception as e:
            print(f'处理 {img_file} 时出错: {e}')
    
    print(f'\n处理完成! 结果已保存到: {save_dir}')

