import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NeRF(nn.Module):
    # 参数：位置坐标编码后的维度，方向向量编码后的维度
    def __init__(self, pos_in_dims=63, dir_in_dims=27, hidden_size=256):
        # 输入：[num_points, 63]，[num_points, 27]
        super(NeRF, self).__init__()
        
        self.hidden_size = hidden_size

        # 位置编码输入的处理层（8层全连接网络）
        self.layer1 = nn.Linear(pos_in_dims, hidden_size)  # [num_points, 63] → [num_points, 256]
        self.layer2 = nn.Linear(hidden_size, hidden_size)  # # [num_points, 256] → [num_points, 256]
        self.layer3 = nn.Linear(hidden_size + pos_in_dims, hidden_size)  # [num_points, 256+63=319] → [num_points, 256]
        self.layer4 = nn.Linear(hidden_size, hidden_size)  # 256维 → 256维
        self.layer5 = nn.Linear(hidden_size + pos_in_dims, hidden_size)  # 256+63=319维 → 256维（带跳跃连接）
        self.layer6 = nn.Linear(hidden_size, hidden_size)  # 256维 → 256维
        self.layer7 = nn.Linear(hidden_size + pos_in_dims, hidden_size)  # 256+63=319维 → 256维（带跳跃连接）
        self.layer8 = nn.Linear(hidden_size, hidden_size)  # 256维 → 256维

        # 密度输出层：预测体积密度sigma（1个值）===> 看网络结构，第八层之后输出密度，并且加入了方向向量
        self.sigma_layer = nn.Linear(hidden_size, 1)  # [num_points, 256] → [num_points, 1]

        # 颜色输出分支：结合方向信息预测RGB颜色
        self.dir_layer1 = nn.Linear(hidden_size + dir_in_dims, hidden_size//2)  # [num_points, 256+27=283] → [num_points, 283]
        self.dir_layer2 = nn.Linear(hidden_size//2, hidden_size//2)  # 128维 → 128维
        self.rgb_layer = nn.Linear(hidden_size//2, 3)  # 128维 → 3维

    # x.shape = [num_points, 3]   # 3：(x, y, z)
    # d.shape = [num_points, 3]   # 3：(dx, dy, dz)
    def forward(self, x, d):
        # 位置编码分支的处理（8层全连接）
        h = F.relu(self.layer1(x))
        h = F.relu(self.layer2(h))
        h = F.relu(self.layer3(torch.cat([h, x], dim=-1)))
        h = F.relu(self.layer4(h))
        h = F.relu(self.layer5(torch.cat([h, x], dim=-1)))
        h = F.relu(self.layer6(h))
        h = F.relu(self.layer7(torch.cat([h, x], dim=-1)))
        h = F.relu(self.layer8(h))

        sigma = self.sigma_layer(h)

        dir_input = torch.cat([h, d], dim=-1)
        h = F.relu(self.dir_layer1(dir_input))
        h = F.relu(self.dir_layer2(h))
        rgb = torch.sigmoid(self.rgb_layer(h))

        # torch.cat 不增加维数，只增加“某一维的长度”
        # [num_points, 3] + [num_points, 1] = [num_points, 4]
        return torch.cat([rgb, sigma], dim=-1)


# 位置编码器：将低维输入映射到高维特征空间 ===> 目的为了引入高频信息，否则建模结果可能丢失细节信息
class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True):
        super(PositionalEncoding, self).__init__()
        self.num_freqs = num_freqs  # 使用的频率数量（L值），默认10个频率
        self.include_input = include_input  # 是否包含原始输入值，默认包含
        self.funcs = [torch.sin, torch.cos]  # 使用的周期函数：正弦和余弦

        # 生成频率序列：2^0, 2^1, 2^2, ..., 2^(num_freqs-1)
        # 例如num_freqs=10：[1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.freqs = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs)
        # 将频率乘以π：[π, 2π, 4π, 8π, ..., 512π]
        self.freqs = self.freqs * np.pi

    def forward(self, x):
        out = []  # 用于存储所有编码结果的列表
        if self.include_input:  # 如果设置为包含原始输入值
            out.append(x)  # 将原始输入值直接加入结果列表

        # 对每个频率生成sin和cos编码
        for freq in self.freqs: # 遍历每个频率：π, 2π, 4π, ..., 512π
            for func in self.funcs:  # 对每个频率，分别计算sin和cos
                # 计算 func(x * freq)
                # 例如：sin(x * π), cos(x * π), sin(x * 2π), cos(x * 2π), ...
                out.append(func(x * freq))

        # 沿着最后一个维度拼接所有编码结果
        # 例如：原始输入 + 所有sin/cos编码
        return torch.cat(out, dim=-1)

# 对外的接口类，将原始坐标和方向转换为颜色和密度
class NeRFModel(nn.Module):
    def __init__(self, pos_freqs=10, dir_freqs=4, hidden_size=256):
        super(NeRFModel, self).__init__()
        # 位置编码器：将3D坐标映射到高维特征空间
        # 输入：3维坐标 → 输出：63维特征
        self.pos_encoder = PositionalEncoding(num_freqs=pos_freqs)
        # 方向编码器：将3D方向映射到高维特征空间
        # 输入：3维方向 → 输出：27维特征
        self.dir_encoder = PositionalEncoding(num_freqs=dir_freqs)

        # 创建NeRF
        # 3 * (1 + 2 * pos_freqs) = 3 * (1 + 2 * 10) = 3 * 21 = 63
        # 3：原始xyz坐标的3个维度
        # 1：保留原始坐标值
        # 2：每个频率生成sin和cos两个值
        # 10：10个频率（pos_freqs = 10）
        self.nerf = NeRF(pos_in_dims=3*(1 + 2*pos_freqs), 
                        dir_in_dims=3*(1 + 2*dir_freqs),
                        hidden_size=hidden_size)

    def forward(self, points, view_dirs):
        # 对3D坐标进行位置编码
        # 输入形状: points = [num_points, 3]
        # 输出形状: points_encoded = [num_points, 63]
        points_encoded = self.pos_encoder(points)
        # 对视角方向进行方向编码
        # 输入形状: view_dirs = [num_points, 3]
        # 输出形状: dirs_encoded = [num_points, 27]
        dirs_encoded = self.dir_encoder(view_dirs)
        # 将编码后的特征输入NeRF
        # 输入: points_encoded = [num_points, 63], dirs_encoded = [num_points, 27]
        # 输出: [num_points, 4] (RGB + sigma)
        return self.nerf(points_encoded, dirs_encoded)

