
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from util.util import args
from util import util
import torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import StegDataset
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time
from model.gan import Discriminator
from util.qr import get_gaussian_kernel
from model.attnflow import AttnFlow
import lpips
from tqdm import tqdm

task_name = 'rmsteg_single_gpu'
num_module = 37
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
writer = SummaryWriter(log_dir=f'log/{task_name}/')

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs('./result/', exist_ok=True)
    os.makedirs('./log/', exist_ok=True)
    os.makedirs('./checkpoints/', exist_ok=True)
    
    # 初始化模型
    print('初始化模型...')
    net = AttnFlow(block_num=4, use_itf=True, use_qr_trans=True, num_module=num_module).to(device)
    
    # 如果有预训练权重，可以加载
    # net.load_state_dict(torch.load('./pretrained/rmsteg.pth', map_location=device))
    
    # 初始化 GAN
    use_gan = args.train.use_gan
    if use_gan:
        dis = Discriminator().to(device)
    
    calc_lpips = lpips.LPIPS(net='vgg').to(device)
    
    # 准备数据集 - 使用 test_img 作为演示
    print('准备数据集...')
    from torchvision import transforms
    from PIL import Image
    
    # 创建一个简单的数据集
    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.img_paths = []
            # 检查 test_img 文件夹
            test_img_dirs = ['./test_img', '../test_img', 'test_img']
            for test_img_dir in test_img_dirs:
                if os.path.exists(test_img_dir):
                    for root, dirs, files in os.walk(test_img_dir):
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')) and 'misc' not in root:
                                self.img_paths.append(os.path.join(root, file))
                    break
            
            # 如果没有找到，使用空图片
            if len(self.img_paths) == 0:
                print('未找到图片，使用生成的空图片')
                self.img_paths = [None]
            else:
                print(f'找到 {len(self.img_paths)} 张图片')
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        
        def __len__(self):
            return max(len(self.img_paths) * 10, 10)  # 至少10个样本
        
        def __getitem__(self, idx):
            idx = idx % len(self.img_paths)
            img_path = self.img_paths[idx]
            
            if img_path is None:
                # 生成随机图片
                img = torch.rand(3, 224, 224)
            else:
                img = Image.open(img_path).convert('RGB')
                img = self.transform(img)
            
            # 生成一个简单的二维码（模拟）
            qr = torch.zeros(1, 224, 224)
            for i in range(0, 224, 32):
                for j in range(0, 224, 32):
                    if (i // 32 + j // 32) % 2 == 0:
                        qr[:, i:i+32, j:j+32] = 1.0
            
            return {
                'img': img,
                'qr': qr
            }
    
    dataset = SimpleDataset()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    print(f'数据集大小: {len(dataset)}')
    
    # 优化器
    params_trainable_net = list(filter(lambda p: p.requires_grad, list(net.parameters())))
    optimizer_net = optim.AdamW(params_trainable_net,
                                args.train.lr,
                                betas=(args.train.betas1, args.train.betas2),
                                weight_decay=args.train.weight_decay)
    lr_scheduler_net = optim.lr_scheduler.StepLR(optimizer_net, args.train.optim_step, gamma=args.train.optim_gamma)
    
    if use_gan:
        params_trainable_gan = list(filter(lambda p: p.requires_grad, list(dis.parameters())))
        optimizer_gan = optim.AdamW(params_trainable_gan,
                                    args.train.lr,
                                    betas=(args.train.betas1, args.train.betas2),
                                    weight_decay=args.train.weight_decay)
        lr_scheduler_gan = optim.lr_scheduler.StepLR(optimizer_net, args.train.optim_step, gamma=args.train.optim_gamma)
    
    # 训练循环
    print('开始训练...')
    print('='*60)
    
    iter_idx = 1
    for epoch_idx in range(1, 31):  # 训练30个epoch，和原始论文一致
        net.train()
        epoch_steg_loss = 0.0
        epoch_qr_loss = 0.0
        epoch_qr_fusion_loss = 0.0
        epoch_loss_dis = 0.0
        epoch_loss_dis_steg = 0.0
        epoch_lpips = 0.0
        epoch_ssim = 0.0
        epoch_s_time = time.time()
        
        for data_idx, data in enumerate(tqdm(dataloader, desc=f'Epoch {epoch_idx}')):
            img = data['img'].to(device)
            qr = data['qr'].to(device)
            
            # 优化判别器
            if use_gan:
                steg, distort, decode_qr, fusion_qr = net(torch.cat([img, qr], dim=1))
                
                dis_result_real = dis(torch.cat([img], dim=1))
                dis_result_fake = dis(torch.cat([steg.detach()], dim=1))
                dis_loss = dis.calc_loss_dis(dis_result_real, dis_result_fake)
                epoch_loss_dis += dis_loss.item() * img.shape[0]
                
                optimizer_gan.zero_grad()
                dis_loss.backward()
                optimizer_gan.step()
            
            # 优化主网络
            steg, distort, decode_qr, fusion_qr = net(torch.cat([img, qr], dim=1))
            steg_loss, ssim_loss, qr_loss, qr_fusion_loss, refine_qr = net.calc_loss(img, steg, qr, decode_qr, fusion_qr)
            lpips_loss = calc_lpips(img, steg).reshape(-1).mean()
            
            total_loss = steg_loss * 5.0 + ssim_loss * 0.1 + qr_loss * 20.0 + lpips_loss * 4.0
            
            if use_gan:
                dis_result = dis(torch.cat([steg], dim=1))
                dis_loss_steg = dis.calc_loss_net(dis_result)
                epoch_loss_dis_steg += dis_loss_steg.item() * img.shape[0]
                total_loss += dis_loss_steg * 0.15
            
            optimizer_net.zero_grad()
            total_loss.backward()
            optimizer_net.step()
            
            epoch_steg_loss += steg_loss.item() * img.shape[0]
            epoch_qr_loss += qr_loss.item() * img.shape[0]
            epoch_lpips += lpips_loss.item() * img.shape[0]
            epoch_ssim += (1.0 - ssim_loss.item()) * img.shape[0]
            epoch_qr_fusion_loss += qr_fusion_loss.item() * img.shape[0]
            
            # 保存图片（只在最后一个 epoch 保存）
            if data_idx == 0 and epoch_idx == args.train.epoch_num:
                util.save_image_from_tensor(img, f'./result/final_cover.png')
                util.save_image_from_tensor(fusion_qr, f'./result/final_trans_qr.png')
                util.save_image_from_tensor(refine_qr, f'./result/final_refine_qr.png')
                util.save_image_from_tensor(steg, f'./result/final_steg.png')
                util.save_image_from_tensor(distort, f'./result/final_distort.png')
                util.save_image_from_tensor(qr, f'./result/final_qr.png')
                util.save_image_from_tensor(decode_qr , f'./result/final_decode_qr.png')
                util.save_image_from_tensor(util.get_error_map(qr, refine_qr, num_module=num_module) , 
                                            f'./result/final_qr_error_map.png')
            
            # TensorBoard 记录
            writer.add_scalar('Loss/Steg Loss', epoch_steg_loss / (data_idx * 2 + img.shape[0]), iter_idx)
            writer.add_scalar('Loss/QR Loss', epoch_qr_loss / (data_idx * 2 + img.shape[0]), iter_idx)
            writer.add_scalar('Loss/QR Fusion Loss', epoch_qr_fusion_loss / (data_idx * 2 + img.shape[0]), iter_idx)
            if use_gan:
                writer.add_scalar('Loss/Dis Loss', epoch_loss_dis / (data_idx * 2 + img.shape[0]), iter_idx)
                writer.add_scalar('Loss/Dis Steg Loss', epoch_loss_dis_steg / (data_idx * 2 + img.shape[0]), iter_idx)
            
            writer.add_scalar('Metrices/SSIM', epoch_ssim / (data_idx * 2 + img.shape[0]), iter_idx)
            writer.add_scalar('Metrices/LPIPS', epoch_lpips / (data_idx * 2 + img.shape[0]), iter_idx)
            
            iter_idx += 1
        
        lr_scheduler_net.step()
        if use_gan:
            lr_scheduler_gan.step()
        
        epoch_t_time = time.time()
        
        # 不保存中间 checkpoint，只保存最终模型
        
        # 打印训练信息
        print(f'epoch {epoch_idx} -- ' \
              f'lr: {optimizer_net.state_dict()["param_groups"][0]["lr"]:.8f}  ' \
              f'steg_loss: {epoch_steg_loss / len(dataset):.8f}  ' \
              f'ssim: {epoch_ssim / len(dataset):.8f}  ' \
              f'qr_loss: {epoch_qr_loss / len(dataset):.8f}  ' \
              f'lpips: {epoch_lpips / len(dataset):.8f}  ' \
              f'time: {epoch_t_time - epoch_s_time:.1f}')
    
    # 保存最终模型
    torch.save(net.state_dict(), f'./checkpoints/{task_name}_final.pth')
    print('='*60)
    print('训练完成!')
    print(f'最终模型已保存到: ./checkpoints/{task_name}_final.pth')
    print(f'TensorBoard 日志已保存到: ./log/{task_name}/')
    print('='*60)

