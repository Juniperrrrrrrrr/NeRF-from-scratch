import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.model import NeRFModel
from source.dataloaders import CustomDataloader
import torch
import torch.nn as nn    
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class TrainModel: 
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 参数说明：位置编码的L ===> 3D坐标的频带数，方向的频带数，隐层神经元数量
        self.model = NeRFModel(pos_freqs=10, dir_freqs=4, hidden_size=256).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                  lr=1e-4,  # Lower initial learning rate
                                  betas=(0.9, 0.999),
                                  eps=1e-8)
        
        self.batch_size = 1

        # 原来的
        # workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        # base_path = os.path.join(workspace_root, 'nerf_synthetic', 'chair')
        # data_path = os.path.join(base_path, 'train')
        # transforms_path = os.path.join(base_path, 'transforms_train.json')


        # # ==== 路径与数据加载 ====
        # # 取项目根目录： .../NeRF-from-scratch
        # 改成：
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ...\NeRF-from-scratch
        # 当前场景（chair）的数据根： .../NeRF-from-scratch/nerf_synthetic/chair
        base_path = os.path.join(project_root, 'nerf_synthetic', 'chair')  # ...\NeRF-from-scratch\nerf_synthetic\chair
        # 训练图片目录与变换文件（相机姿态 JSON）
        data_path = os.path.join(base_path, 'train')  # ...\chair\train
        transforms_path = os.path.join(base_path, 'transforms_train.json')  # ...\chair\transforms_train.json

        # 创建自定义 DataLoader：
        # - 读取 train/ 下的 PNG
        # - 读取 transforms_train.json 的相机矩阵
        # - __getitem__ 内部采样 N_rays 条射线并沿深度取样，返回 points / rays_d / z_vals / rgb_gt
        self.dataloader = CustomDataloader(self.batch_size, data_path, transforms_path)
        self.epochs = 200

        # 损失函数用均方误差，课里面讲了
        self.mse_loss = nn.MSELoss()
        

        # gamma=0.995 表示每个epoch后，学习率变为原来的 99.5%，后面学习率衰减用到
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)

        # ==== 断点续训与日志 ====
            # float('inf') 在 表示无穷大
            # 训练开始前：best_loss = inf（无限大）
            # 第一次计算损失：比如 current_loss = 0.5
            # 比较：if current_loss < best_loss: → 0.5 < inf → True
            # 更新：因为 0.5 肯定小于无限大，所以会执行更新操作：
        self.start_epoch = 0  # 从 0 开始；恢复训练时会改成 checkpoint 的下一轮
        self.best_loss = float('inf')  # 记录历史最优，用于保存 best_checkpoint.pth
        self.running_loss = []  # 保存最近 window 内的 loss，用于平滑打印
        self.window_size = 100  # 滑动窗口大小（打印更稳定）

# ============数据加载 → 前向传播 → 损失计算 → 反向传播 → 参数更新 → 模型保存===============
    def train(self):
        # try是处理异常的代码，用法如下：
        # try:
        #     # 可能会出错的代码
        #      risky_operation()
        # except Exception as e:
        #     # 如果出错了，执行这里的代码
        #     print(f"出错了: {e}")
        try:
            torch.cuda.empty_cache()  # 清理GPU内存，防止内存溢出

            # epoch循环
            for epoch in range(self.start_epoch, self.epochs): # self.epochs ===> 训练总轮数
                epoch_loss = 0.0 # 累计当前epoch的总损失
                batch_count = 0  # 计数当前epoch的batch数量

                # 数据加载
                for i, data in enumerate(self.dataloader):
                    try:
                        # 数据移动到gpu
                        points = data['points'].to(self.device)   # 3D采样点坐标 [batch, rays, n_samples, 3]
                        rays_d = data['rays_d'].to(self.device)   # 射线方向向量 [batch_size, n_rays, 3]
                        z_vals = data['z_vals'].to(self.device)   # 采样深度值 ===> o+td的t [batch_size, n_rays, n_samples]
                        rgb_gt = data['rgb_gt'].to(self.device)   # 真实颜色值（Ground Truth）[batch_size, n_rays, 3]
                        
                        batch_size, n_rays, n_samples = points.shape[0], points.shape[1], points.shape[2]
                        # batch_size ===> 一次迭代用多少张图；chunk_size ===> 在 GPU 上真正同时算多少条射线
                        chunk_size = 2048  # Reduced chunk size ===> 分块大小（防止GPU内存不足）
                        rgb_pred = torch.zeros((batch_size, n_rays, 3), device=self.device)  # 初始化预测结果容器 [B, n_rays, 3]
                        
                        self.optimizer.zero_grad()  # 清零梯度（重要！）

                        for chunk_start in range(0, n_rays, chunk_size): # 从0~n_rays中，每次取chunk_size个射线
                            chunk_end = min(chunk_start + chunk_size, n_rays) # 每个循环的开始更新chunk_end

                            # ========== 提取当前块的数据 ========
                            points_chunk = points[:, chunk_start:chunk_end].reshape(-1, n_samples, 3)
                            # 形状: [batch_size, chunk_size, n_samples, 3] → [batch_size * chunk_size, n_samples, 3]
                            rays_d_chunk = rays_d[:, chunk_start:chunk_end].reshape(-1, 3)
                            # 形状: [batch_size, chunk_size, 3] → [batch_size * chunk_size, 3]
                            z_vals_chunk = z_vals[:, chunk_start:chunk_end].reshape(-1, n_samples)
                            # 形状: [batch_size, chunk_size, n_samples] → [batch_size * chunk_size, n_samples]
                            

                            rays_d_norm = F.normalize(rays_d_chunk, dim=-1)# 标准化射线方向  [batch_size * chunk_size, 3]

                            # 重塑数据形状以适应模型输入 因为全连接
                            # 形状: [batch_size * chunk_size, n_samples, 3] → [batch_size * chunk_size * n_samples, 3]
                            # 可以理解为：batch_size * chunk_size * n_samples个点，的xyz坐标
                            points_flat = points_chunk.reshape(-1, 3)
                            # unsqueeze(1) 在第 1 个维度增加一个大小为 1 的新轴。
                            #   unsqueeze(1): [B*C, 3] → [B*C, 1, 3]
                            #   expand(-1, n_samples, -1): [B*C, 1, 3] → [B*C, n_samples, 3]
                            #   reshape(-1, 3): [B*C, n_samples, 3] → [B*C * n_samples, 3]
                            dirs_flat = rays_d_norm.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, 3)

                            # 神经网络前向传播 ===> 把点和方向送入网络
                            outputs = self.model(points_flat, dirs_flat)
                            # 输入: [B*C*n_samples, 3] 和 [B*C*n_samples, 3]
                            # 输出: [B*C*n_samples, 4] (RGB + sigma)
                            outputs = outputs.reshape(-1, n_samples, 4)
                            # 形状: [B*C*n_samples, 4] → [B*C, n_samples, 4]

                            rgb = outputs[..., :3]  # 提取rgb [B*C, n_samples, 3]
                            sigma = outputs[..., 3]  # 提取密度 [B*C, n_samples]，切片用数字不用冒号会降维，直接把最后一个维度挤掉

                            # ===============体渲染过程 ===> 利用很多个点生成的密度和rgb来计算出一个像素的颜色==============
                            # 1、计算dists
                                    # dists：相邻采样点之间的深度距离 ===>  [B*C, n_samples-1]
                            dists = z_vals_chunk[..., 1:] - z_vals_chunk[..., :-1]
                            # 在末尾添加一个很大的距离值，用于处理最后一个采样点 ===> [B*C, n_samples]
                            dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)

                            # 2、计算不透明度alpha
                                    # 输入: sigma [B*C, n_samples], dists [B*C, n_samples]
                                    # 输出: alpha [B*C, n_samples]
                                    # 目的: 计算每个采样点的不透明度（透光率）
                                    # 物理意义: 光线通过该点时被吸收的概率
                            alpha = 1 - torch.exp(-F.relu(sigma) * dists) # 用Relu是为了保证密度sigma>0
                            # 3、计算累积透明度（Transmittance） Ti = (1-a1)*，，，*(1-ai-1)
                                    # 输出: T [B*C, n_samples]
                                    # 目的: 计算光线到达每个采样点之前的累积透明度
                                    # 物理意义: 光线能够传播到该点的概率
                            # torch.cumprod() ===> dim=0：行累乘，dim=1：列累乘
                            T = torch.cumprod(torch.cat([
                                torch.ones_like(alpha[..., :1]),  # [B*C, 1]
                                (1 - alpha + 1e-10)[..., :-1]     # [B*C, n_samples-1]
                            ], dim=-1), dim=-1)

                            # wi表示第 i 个采样点对最终颜色的贡献权重
                            # w = alpha * T ===> 光线通过该点时被吸收的概率*光线能够传播到该点的概率
                            weights = alpha * T  # [B*C, n_samples]
                            #   weights.unsqueeze(-1): [B*C, n_samples] → [B*C, n_samples, 1]
                            #   rgb: [B*C, n_samples, 3]
                            #   相乘: [B*C, n_samples, 1] * [B*C, n_samples, 3] = [B*C, n_samples, 3]
                            #   sum(dim=1): [B*C, n_samples, 3] → [B*C, 3] ===> 因为要算一条射线上面采样点的和，所以在samples这个维度上面加
                            # 输出: rgb_chunk [B*C, 3]
                            # 目的: 将所有采样点的颜色按权重合成最终像素颜色
                            rgb_chunk = (weights.unsqueeze(-1) * rgb).sum(dim=1)

                            # rgb_pred ===> [B, n_rays, 3]：batch，射线数量，rgb的值
                            # 输入: rgb_chunk [B*C, 3] → 重塑为 [batch_size, chunk_size, 3]
                            # 输出: 存入 rgb_pred 的对应位置
                            # 目的: 将当前块的结果放回最终结果张量中
                            rgb_pred[:, chunk_start:chunk_end] = rgb_chunk.reshape(batch_size, -1, 3)

                        # 计算损失和反向传播
                        loss = self.mse_loss(rgb_pred, rgb_gt)  # 计算预测与真实的均方误差
                        loss.backward()   # 反向传播计算梯度
  
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)  # 梯度裁剪防爆炸
                        
                        self.optimizer.step()  # 更新模型参数

                        # 损失记录与日志
                        epoch_loss += loss.item()  # 累计损失
                        batch_count += 1  # 计数batch
                        
                        self.running_loss.append(loss.item()) # 加入最新的损失值
                        if len(self.running_loss) > self.window_size: # 如果超过窗口大小
                            self.running_loss.pop(0) # 移除最旧的损失值
                        
                        if i % 10 == 0:
                            avg_loss = sum(self.running_loss) / len(self.running_loss)
                            print(f'Epoch [{epoch}/{self.epochs}], Step [{i}], Loss: {avg_loss:.6f}')
                        
                        # Clear memory
                        # 清理内存，防止GPU内存泄漏
                        del points, rays_d, z_vals, rgb_gt, rgb_pred, outputs
                        torch.cuda.empty_cache()
                            
                    except Exception as e:  # 与try连用 如果出错了执行下面这行代码
                        print(f"Error in batch {i}: {str(e)}")
                        continue
                # 学习率调整与模型保存
                self.scheduler.step() # 调整学习率（指数衰减），将当前学习率乘以gamma（0.995），实现学习率逐渐减小

                avg_loss = epoch_loss / batch_count  # 计算epoch平均损失
                # 保存最佳模型 ===> loss值最小的那个
                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    torch.save({
                        'epoch': epoch,  # 当前训练到的epoch数
                        # 保存模型的所有参数权重，state_dict() 返回一个字典，包含所有可学习参数
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),  # 优化器的状态
                        'loss': avg_loss, # 当前的损失值
                    }, 'best_checkpoint.pth')  # 保存的文件名
                # 每50个epoch保存一次检查点（第50、100、150、200...个epoch）
                if (epoch + 1) % 50 == 0:
                    torch.save({
                        'epoch': epoch,  # epoch
                        'model_state_dict': self.model.state_dict(), # 权重
                        'optimizer_state_dict': self.optimizer.state_dict(), # 优化器
                        'loss': avg_loss, # 损失
                        # 文件名包含epoch数，例如：checkpoint_epoch_50.pth
                            # f-string 允许你在字符串中直接嵌入变量和表达式
                            # 传统：filename = "checkpoint_epoch_" + str(epoch + 1) + ".pth"
                            # f-string：filename = f"checkpoint_epoch_{epoch + 1}.pth"
                    }, f'checkpoint_epoch_{epoch+1}.pth')
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            
if __name__ == "__main__":
    train_model = TrainModel()
    train_model.train()