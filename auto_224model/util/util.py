
import torch
from torchvision import transforms, utils
from PIL import Image
from easydict import EasyDict
import yaml
import matplotlib.pyplot as plt
import torch.nn as nn
import util.qr as uqr
from einops import rearrange
import os

try:
    from pyzbar import pyzbar
except:
    pyzbar = None


def resize_qr_unit(qr_unit, factor):
    unit_size = qr_unit.shape[-1]
    qr_unit = qr_unit.reshape(qr_unit.shape[0], qr_unit.shape[1], -1, 1)
    qr_unit = qr_unit.repeat(1, 1, 1, factor * factor)
    qr_unit = rearrange(qr_unit, 'b c (h w) (f1 f2) -> b c (h f1) (w f2)', h=unit_size, w=unit_size, f1=factor, f2=factor)
    return qr_unit


def save_image_from_tensor(t, filename):
    while len(t.shape) < 4:
        t = t.unsqueeze(0)
    if t.shape[-1] == 1 or t.shape[-1] == 3:
        t = t.permute(0, 3, 1, 2)
    t = t.clone().detach()
    t = t.to(torch.device('cpu'))
    utils.save_image(t, filename)


def image_to_tensor(img_path):
    rgb = transforms.ToTensor()(Image.open(img_path))
    rgb = rgb.unsqueeze(0)
    if rgb.shape[1] == 4:
        rgb = rgb[:, :3, ...]
    return rgb


def calc_psnr(tensor1, tensor2):
    l1_loss = nn.L1Loss()
    l1 = l1_loss(tensor1, tensor2)
    psnr = 20 * torch.log10(1 / l1)
    return psnr


def calc_emr(qr, decode_qr, num_module):
    qr_trans = transforms.Resize((num_module * 5, num_module * 5))
    qr = qr_trans(qr)[:, :1, ...]
    decode_qr = qr_trans(decode_qr)[:, :1, ...]
    gs_kernel = uqr.get_gaussian_kernel(5, 1.0)
    gs_conv = nn.Conv2d(1, 1, 5, padding=0, stride=5).requires_grad_(False)
    gs_conv.weight.data = gs_kernel
    gs_conv = gs_conv.to(qr.device)
    qr = gs_conv(qr.clamp(0.0, 1.0))
    decode_qr = gs_conv(decode_qr.clamp(0.0, 1.0))
    error_map = torch.round(torch.abs(qr - decode_qr))
    emr = error_map.mean().reshape(-1).item()
    return emr * 100 # in percentage


def calc_tra(decode_qr):
    if pyzbar is None:
        return 0
    decode_qr = torch.cat([decode_qr, decode_qr, decode_qr], dim=1).clamp(0.0, 1.0)[0].cpu()[:3, ...]
    decode_qr = transforms.ToPILImage()(decode_qr)
    try:
        decode_qr_msg = pyzbar.decode(decode_qr, symbols=[pyzbar.ZBarSymbol.QRCODE])[0].data.decode("utf-8")
        return 1
    except Exception as e:
        return 0


def get_error_map(gt, pred, num_module, img_size=None):
    # 自适应版本：支持任意分辨率，但原理和原始 RMSteg 完全一致！
    
    # 步骤1：找到 red.png 和 green.png
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
    
    if red_path is None or green_path is None:
        # 如果找不到 red 和 green，回退到简单的灰度图
        error = torch.abs(gt - pred)
        error = torch.clamp(error * 10, 0, 1)
        if error.shape[1] == 1:
            error = error.repeat(1, 3, 1, 1)
        return error
    
    # 步骤2：确定输出大小
    if img_size is None:
        # 如果没有指定大小，以 gt 的大小为准
        out_h, out_w = gt.shape[2], gt.shape[3]
    else:
        out_h, out_w = img_size, img_size
    
    # 步骤3：加载 red 和 green 并缩放到目标大小
    red = transforms.Resize((out_h, out_w))(image_to_tensor(red_path)).to(pred.device)
    green = transforms.Resize((out_h, out_w))(image_to_tensor(green_path)).to(pred.device)
    
    # 步骤4：计算差异
    # 先对齐 gt 和 pred 的尺寸
    if gt.shape != pred.shape:
        pred = torch.nn.functional.interpolate(pred, size=(gt.shape[2], gt.shape[3]), mode='nearest')
    
    error = torch.abs(gt - pred)
    
    # 步骤5：处理二维码（和原始 RMSteg 一致）
    error = transforms.Resize((num_module, num_module))(error)
    error = resize_qr_unit(error, factor=7)
    error = transforms.Resize((out_h, out_w))(error)
    
    # 步骤6：生成红色/绿色的误差图
    error_binary = torch.round(error)
    error_map = error_binary * red + (1.0 - error_binary) * green
    
    return error_map


# 加载配置文件，支持多个路径
config_path = None
for path in ['config.yaml', '../src/config.yaml']:
    if os.path.exists(path):
        config_path = path
        break

if config_path is None:
    # 使用默认配置
    args = EasyDict({
        'train': {
            'img_size': 224
        }
    })
else:
    with open(config_path, 'r', encoding='utf-8') as f:
        args = EasyDict(yaml.load(f, Loader=yaml.SafeLoader))


if __name__ == '__main__':
    print(args)

