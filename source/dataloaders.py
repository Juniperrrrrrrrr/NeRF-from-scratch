# 📁 数据加载：读取图像和相机参数
# 🎯 射线生成：为每个像素生成3D射线
# 📊 点采样：沿射线采样3D点用于体积渲染
# 🔄 批量处理：组织数据供神经网络训练

import os
import json 
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# 加载合成数据集的类，继承自PyTorch的Dataset
class LoadSyntheticDataset(Dataset):
    def __init__(self, path_to_images, path_to_labels):

        # 检查图片目录是否存在
        if not os.path.exists(path_to_images):
            raise FileNotFoundError(f"Images directory not found: {path_to_images}")

        # 检查标签文件是否存在
        if not os.path.exists(path_to_labels):
            raise FileNotFoundError(f"Labels file not found: {path_to_labels}")
            
        self.path_to_images = path_to_images
        # 获取目录下所有文件，筛选出PNG图片
        all_files = os.listdir(path_to_images)
        self.images = [im for im in all_files if im.endswith('.png')]

        # 图像转换：PIL图像 → PyTorch张量
        self.transform = transforms.ToTensor()
        
        try:
            # 尝试打开并读取JSON文件
            with open(path_to_labels, 'r') as f:
                # 将JSON内容解析为Python字典，包含相机参数
                self.labels = json.load(f)
            # 获取camera_angle_x的值（相机视角角度，用于计算焦距），如果不存在则返回None
            self.camera_angle_x = self.labels.get('camera_angle_x', None)
        except Exception as e:  # 如果上面任何一步出错
            # 重新抛出错误，让调用者处理
            raise

    # 为图像中的每个像素生成一条3D射线，用于后续的体积渲染
    def get_origins_and_directions(self, frame, width, height):  # width、height分别为图像的宽高
        # frame是是字典，包含单张图像的相机参数，也就是提供相机外参通常包括：
        # frame = {
        #     'file_path': 'path/to/image.png',
        #     'transform_matrix': [
        #         [R11, R12, R13, T1],  # 旋转矩阵的第1行 + 平移向量的第1个分量
        #         [R21, R22, R23, T2],  # 旋转矩阵的第2行 + 平移向量的第2个分量
        #         [R31, R32, R33, T3],  # 旋转矩阵的第3行 + 平移向量的第3个分量
        #         [0, 0, 0, 1]  # 齐次坐标的最后一行
        #     ]
        # }

        # 取出“世界→相机”外参矩阵（4×4），并转成 torch 张量
        origins = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
        # 取矩阵第 4 列前 3 个元素 → 相机光心在世界坐标系中的位置
        origins = origins[:3, 3]  # [3]

        # 把一维向量改成 2-D，方便后面批量复制
        origins = origins.view(1, 3)  # [1, 3]
        # 复制H×W份，使每条像素射线都有一份起点，参数：0维复制width * height份，1维复制1份
        origins = origins.repeat(width * height, 1)  # [H*W, 3]

        # 建立像素坐标网格
        # i 对应列索引（x），j 对应行索引（y）；indexing='xy' 保证顺序正确
        i, j = torch.meshgrid(
            torch.arange(width, dtype=torch.float32),
            torch.arange(height, dtype=torch.float32),
            indexing='xy'  # i/j 形状都是 [H, W]
            )   

        # 假设传感器宽度 = 图像宽度的一半 → 算出焦距
        focal = width / 2
        # 把像素坐标转成相机空间下的归一化坐标（成像平面z = -1）
        x = (i - width * 0.5) / focal 
        y = (j - height * 0.5) / focal
        # 相机朝 -Z 方向看
        z = -torch.ones_like(x) * 1

        # 拼成3D方向向量并压扁成二维数组
        directions = torch.stack((x, y, z), dim=-1)  # [H, W, 3]
        directions = directions.view(-1, 3)  # [H*W, 3]

        return origins, directions

    # 从已经生成的全部射线里，随机抽 N_rays 条
    def sample_random_rays(self, rays_o, rays_d, N_rays):
        total_rays = rays_o.shape[0] # 总射线量 rays_0===>[H*w, 3],total_rays = h*W
        # 随机选择N_rays条射线的索引
        indices = torch.randint(0, total_rays, (N_rays,))   # [N_rays]

        # 根据索引采样原点和方向
        rays_o_sampled = rays_o[indices]  # [N_rays, 3]
        rays_d_sampled = rays_d[indices]  # [N_rays, 3]

        return rays_o_sampled, rays_d_sampled

    # 沿射线采样点（用于体积渲染）
    def get_rays_sampling(self, origins, directions, near, far, samples):
        # 生成 samples 个深度值，范围 [near, far]
        z_vals = torch.linspace(near, far, steps=samples)  # [samples]
        # 加两维占位，方便后面广播
        # [samples] → [1, samples, 1]
        z_vals = z_vals[None, :, None]  # [1, samples, 1]

        # 把起点、方向也升维，变成 [N_rays, 1, 3]
        origins = origins[:, None, :]     # [N_rays, 1, 3]
        directions = directions[:, None, :]  # [N_rays, 1, 3]

        #  广播乘法：每条射线沿自身方向走 z_vals 深度，得到采样点
        # origins + directions * z_vals
        # 广播后： [N_rays, 1, 3] * [1, samples, 1] → [N_rays, samples, 3]
        points = origins + directions * z_vals  # [N_rays, samples, 3]

        # 返回采样点坐标 & 对应的深度值（去掉多余维度）
        # points: [N_rays, samples, 3]
        # z_vals: [samples]
        # squeeze() 把长度为1的维度去掉
        return points.float(), z_vals.squeeze(0)  # z_vals: [samples] → used for rendering

    # 返回数据集大小（图像数量）
    def __len__(self): 
        return len(self.images)

    # 核心方法：获取单个数据项
    def __getitem__ (self, idx): 
        try:
            # 获取当前图像的标签信息（从JSON文件中读取的相机参数）
            label = self.labels['frames'][idx]
            # 构建图像文件路径（从标签中获取文件名并添加.png后缀）
            file_name = os.path.basename(label['file_path']) + '.png'
            img_path = os.path.join(self.path_to_images, file_name)

            # 检查图像文件是否存在
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")

            # 打开图像文件并转换为RGB格式（确保3通道）
            image = Image.open(img_path).convert("RGB")  # (H, W, 3)

            # 如果定义了图像转换函数（将PIL图像转换为PyTorch张量）
                # # 在 __init__ 方法中：
                # self.transform = transforms.ToTensor()  # 这是一个函数对象
                # # 在 __getitem__ 方法中：
                # if self.transform:  # 因为 self.transform 是一个函数对象，所以为 True
                #     image = self.transform(image)  # 执行图像转换
            if self.transform:
                # transforms.ToTensor()完成两个操作：
                # 1、将PIL Image转换为PyTorch张量
                # 2、将维度顺序从HWC改为CHW
                image = self.transform(image)  # (3, H, W,)

            # 设置光线数量
            N_rays = 4096
            # 获取图像的高度和宽度
            H, W = image.shape[1], image.shape[2]

            # 如果有相机视角参数，使用公式计算焦距
            if self.camera_angle_x is not None:
                focal = W / (2 * np.tan(self.camera_angle_x / 2))
            else:
                # 如果没有相机参数，使用默认焦距（宽度的一半）
                focal = W / 2

            # 随机选像素坐标（在图像上随机选择4096个点作为射线）
            i = torch.randint(0, W, (N_rays,))  # 随机X坐标
            j = torch.randint(0, H, (N_rays,))  # 随机Y坐标

            # 获取这些像素点的真实RGB颜色值
            # image[:, j, i]：获取所有通道在(j,i)位置的值 → [3, N_rays]
            # .permute(1, 0)：转置维度 → [N_rays, 3]
            rgb_gt = image[:, j, i].permute(1, 0)  # [N_rays, 3]

            # ========像素→成像平面→相机空间方向=======
            x = (i.float() - W * 0.5) / focal  # X坐标标准化
            y = (j.float() - H * 0.5) / focal  # Y坐标标准化
            z = -torch.ones_like(x)  # Z坐标设为-1（指向相机前方）
            # 组合成方向向量 [N_rays, 3]
            dirs = torch.stack([x, y, z], dim=-1)  # [N_rays, 3]

            # ========相机空间方向→世界坐标=======
            # 取相机到世界坐标的变换矩阵
            c2w = torch.tensor(label['transform_matrix'], dtype=torch.float32)  # [4, 4]
            # 将射线方向从相机坐标系转换到世界坐标系
            # @ 表示矩阵乘法，c2w[:3, :3].T 是旋转矩阵的转置
            # c2w:存储的是世界到相机的旋转矩阵，所以相机到世界要做转置
            rays_d = (dirs @ c2w[:3, :3].T).float()  # Rotate ray directions
            # 取出相机光心坐标（坐标，不是方向向量，所以不用乘以旋转矩阵）
            rays_o = c2w[:3, 3].expand(rays_d.shape)  # [N_rays, 3]

            # 设置采样范围
            near, far = 2.0, 6.0
            # 均匀采样64个点
            t_vals = torch.linspace(0., 1., steps=64)
            # 线性插值
            # 把在 [0, 1] 区间的 t_vals 映射到实际的 [near, far] 深度范围，得到每条射线从近到远的采样深度 z_vals
            z_vals = near * (1. - t_vals) + far * t_vals  # [64]
            # expand()只扩张维度长度为1的维度
            z_vals = z_vals.expand(N_rays, -1)  # [N_rays, 64]

            # 分层采样（hierarchical sampling） - 让采样点更集中在重要区域
            mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1]) # 计算中间点
            # 上边界和下边界，他们对应位置组合就是一个边界区间，他们是相互错开的那种
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand_like(z_vals)  # 随机偏移量
            z_vals = lower + (upper - lower) * t_rand  # 在每个区间内随机采样（提高采样效率）

            # 计算3D空间中的采样点坐标
            # 射线公式：points = ray_origin + ray_direction * depth
            # rays_o[:, None, :]：从 [N_rays, 3] 扩展为 [N_rays, 1, 3]
            # rays_d[:, None, :]：从 [N_rays, 3] 扩展为 [N_rays, 1, 3]
            # z_vals[..., :, None]：从 [N_rays, 64] 扩展为 [N_rays, 64, 1]
            points = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # [N_rays, 64, 3]

            # 返回数据字典（用于训练）
            return {
                'points': points,  # 3D采样点坐标 [N_rays, 64, 3]
                'rays_d': rays_d,  # 射线方向 [N_rays, 3]
                'rgb_gt': rgb_gt,  # 真实颜色值 [N_rays, 3]
                'z_vals': z_vals   # 采样深度值 [N_rays, 64]
            }
        except Exception as e:
            # 如果处理过程中出现错误，打印详细错误信息
            print(f"Error processing item {idx}: {str(e)}")
            import traceback
            traceback.print_exc()  # 打印完整的错误堆栈跟踪
            raise  # 重新抛出异常（让调用者知道出错）

# 自定义数据加载器类
class CustomDataloader:
    # 初始化方法，创建数据加载器实例
    def __init__(self, batch_size, path_to_images=None, path_to_labels=None):
        # 检查参数是否提供，确保路径参数不为空
        if path_to_images is None or path_to_labels is None:
            # 如果任何一个路径参数未提供，抛出值错误异常
            raise ValueError("Both path_to_images and path_to_labels must be provided")

        # 创建数据集实例，加载合成数据集
        # LoadSyntheticDataset是自定义的数据集类，负责读取图像和相机参数
        self.dataset = LoadSyntheticDataset(
                path_to_images=path_to_images,  # 图像文件所在目录路径
                path_to_labels=path_to_labels   # 相机参数JSON文件路径
            )
        # 设置批量大小，即每次训练时处理的图片数
        self.batch_size = batch_size
        # 创建PyTorch的DataLoader实例，用于批量加载数据
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    # 定义迭代器方法，使CustomDataloader可迭代
    # 如for batch in dataloader，Python会自动调用 __iter__()
    def __iter__(self):
        # 返回底层DataLoader的迭代器，用于遍历所有数据批次
        return iter(self.loader)

    # 定义长度方法，返回数据加载器中的批次数量
    def __len__(self):
        # 返回底层DataLoader的长度（总批次数 = 总样本数 / 批量大小）
        return len(self.loader)

# dataset = LoadSyntheticDataset(
#     path_to_images= '/teamspace/studios/this_studio/nerf_synthetic/chair', 
#     path_to_labels= '/teamspace/studios/this_studio/nerf_synthetic/chair/transforms_train.json'
# )


# loader = DataLoader(dataset, batch_size = 4, shuffle= True)



# for points in loader: 
#     print(points.shape)
#     break
    # print(labels)
