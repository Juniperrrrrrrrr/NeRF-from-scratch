### ENTIRELY AI GENERATED CODE ###

import os
import torch
import json
import numpy as np
from PIL import Image
import imageio.v2 as imageio  # 用来写 GIF
from model import NeRFModel
from renderer import save_rendered_image  # 导入图像保存函数
import torch.nn.functional as F

# ========== 1. 读 checkpoint 加载训练好的模型==========
def load_checkpoint(model, checkpoint_path):
    """Load the latest checkpoint."""  """加载训练好的模型检查点"""
    if not os.path.exists(checkpoint_path):  # 检查检查点文件是否存在
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")  # 如果不存在则抛出错误
    
    checkpoint = torch.load(checkpoint_path)  # 加载检查点文件: dict或state_dict
    
    # Handle different checkpoint formats
    # 处理不同的检查点格式
    if isinstance(checkpoint, dict):  # 如果检查点是字典类型
        if 'model_state_dict' in checkpoint:  # 如果检查点是字典类型
            state_dict = checkpoint['model_state_dict']  # 获取模型状态字典: dict
        elif 'state_dict' in checkpoint:   # 检查是否有state_dict键
            state_dict = checkpoint['state_dict']  # 获取状态字典: dict
        else:
            # Assume the dict itself is the state dict
            # 假设字典本身就是状态字典
            state_dict = checkpoint  # 直接使用整个字典: dict
    else:
        raise ValueError("Checkpoint format not recognized")  # 格式不被识别时抛出错误

    # load_state_dict()是内置函数，目的是将参数写进网络
    model.load_state_dict(state_dict)  # 加载模型权重到模型实例
    print("Successfully loaded checkpoint from:", checkpoint_path)
    return model  # 返回加载好的模型

# ========== 2. 给定相机内参 + 外参，生成整张图的射线 =========
def get_rays(H, W, focal, c2w, device='cuda'):
    """Generate rays for the given camera parameters."""
    # 创建像素网格坐标 [W, H]
    #  生成0到W-1的序列: [W]；生成0到H-1的序列: [H]  indexing='xy': 第一个参数对应x坐标（列），第二个对应y坐标（行）
    # i,j 形状：[H, W]
    # torch.arange(W) ===> 0开始 步长为1
    i, j = torch.meshgrid(torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
    i = i.float()
    j = j.float()

    # 像素→相机
    x = (i - W * 0.5) / focal  # [H, W]
    y = (j - H * 0.5) / focal  # [H, W]
    z = -torch.ones_like(x)  # 生成z坐标 [H, W]
    dirs = torch.stack([x, y, z], dim=-1)  # 方向向量，[H, W, 3]

    # 相机→世界
    # Rotate ray directions
    rays_d = (dirs @ c2w[:3, :3].T)  # [H, W, 3]
    # c2w[:3, 3] 形状是 (3,)，expand()会把维度长度为1的维度扩张
    rays_o = c2w[:3, 3].expand(rays_d.shape)  # [H, W, 3]
    
    return rays_o, rays_d

# ========== 3. 在给定相机位姿 c2w 下渲染一整张图（新视角） ==========
def render_novel_view(model, c2w, H=400, W=400, focal=None, device='cuda'):
    """Render a novel view from the given camera pose."""
    """从给定相机位姿渲染新视角"""
    model.eval()  # 设置模型为评估模式 ===> 基本上不是在训练参数阶段都需要设置成这个
    if focal is None:
        focal = W / 2
        
    with torch.no_grad():  # 不需要反向传播  禁用后节省内存
        rays_o, rays_d = get_rays(H, W, focal, c2w, device=device)  # 调用get_rays来生成所有射线

        # 展平射线以便处理 [H*W, 3]
        # Flatten rays for processing
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        # 分块处理射线以节省内存
        # Process rays in chunks to save memory
        chunk_size = 4096  # 每个块处理4096条射线
        rgb_final = torch.zeros((H*W, 3), device=device)  # 存最后颜色  [H*W, 3]
        
        for chunk_start in range(0, rays_o.shape[0], chunk_size):  # (0~H*W),也就是所有射线中，每次取chunk_size个射线
            # 按 chunk 前向，防显存爆炸
            chunk_end = min(chunk_start + chunk_size, rays_o.shape[0])

            # 获取当前块的射线数据
            rays_o_chunk = rays_o[chunk_start:chunk_end]  # [chunk_start:chunk_end]只给了一个维度，[chunk_size, 3]
            rays_d_chunk = rays_d[chunk_start:chunk_end]  # [chunk_size, 3]
            # 标准化方向向量 ===> 将射线方向向量转换为单位向量（长度为1），标准化后：point = O + td，t就可以表示真实步长
            rays_d_chunk = F.normalize(rays_d_chunk, p=2, dim=-1)  # [chunk_size, 3]
            
            # Sampling strategy from training  使用与训练相同的采样策略
            near, far = 2.0, 6.0   # 在 [near, far] 线性采样 64 个深度
            t_vals = torch.linspace(0., 1., steps=64, device=device) # [64] 生成均匀采样参数
            z_vals = near * (1. - t_vals) + far * t_vals  # [64] 利用插值生成实际深度值
            z_vals = z_vals.expand(rays_o_chunk.shape[0], -1)  # [chunk_size, 64] - 扩展到每条射线

            # 添加分层采样 就是添加扰动，使采样点不是均匀取值
            # Add stratified sampling
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])  # [chunk_size, 63] - 中间点
            upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)  # [chunk_size, 64] - 上边界
            lower = torch.cat([z_vals[..., :1], mids], dim=-1)   # [chunk_size, 64] - 下边界
            t_rand = torch.rand(z_vals.shape, device=device)  # [chunk_size, 64] - 随机数===>[0, 1)
            z_vals = lower + (upper - lower) * t_rand  # [chunk_size, 64] - 分层采样后的深度值

            # 获取采样点坐标（射线方程: o + t*d）
            # Get sample points
            points = rays_o_chunk[:, None, :] + rays_d_chunk[:, None, :] * z_vals[..., :, None]  # [chunk_size, 64, 3]
            view_dirs = rays_d_chunk[:, None, :].expand_as(points)  # [chunk_size, 64, 3]

            # 展平点和方向以便批量处理
            # Flatten points and directions
            points_flat = points.reshape(-1, 3)  # [chunk_size*64, 3]
            dirs_flat = view_dirs.reshape(-1, 3)  # [chunk_size*64, 3]

            # 在更小的子块中处理以避免内存溢出  ===> 一次处理多少个点，刚刚的chunck是一次处理多少射线
            # Process in smaller sub-chunks
            sub_chunk_size = 8192  # 每个子块处理8192个点
            outputs_chunks = []  # 存储子块输出
            
            for i in range(0, points_flat.shape[0], sub_chunk_size):
                points_sub = points_flat[i:i+sub_chunk_size]   # [min(8192, 剩余点数), 3]
                dirs_sub = dirs_flat[i:i+sub_chunk_size]   # [min(8192, 剩余点数), 3]
                outputs_sub = model(points_sub, dirs_sub)   # [min(8192, 剩余点数), 3] ===> RGB+sigma
                outputs_chunks.append(outputs_sub)  # 添加到子块输出列表
                torch.cuda.empty_cache()  # 清理GPU内存

            # 合并所有子块的输出
            outputs = torch.cat(outputs_chunks, 0)  # [chunk_size*64, 4]
            outputs = outputs.reshape(points.shape[0], points.shape[1], 4)  # [chunk_size, 64, 4]

            # 分离RGB和sigma输出
            # Split outputs
            rgb = outputs[..., :3]  # [chunk_size, N_samples, 3]
            sigma = outputs[..., 3]  # [chunk_size, N_samples] ===> 无“：”，降维

            # 使用与训练相同的密度缩放
            # Use same density scaling as training
            sigma = sigma * 0.1  # [chunk_size, 64] - 缩放后的密度

            # 计算相邻采样点之间的距离
            # Compute distances
            dists = z_vals[..., 1:] - z_vals[..., :-1]  # [chunk_size, N_samples-1]，[chunk_size, 63]
            # 添加最后一个距离
            dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)  # [chunk_size, N_samples]
            # [chunk_size, 64]

            # 计算alpha合成（透明度）
            # Compute alpha compositing
            # relu保证密度>0
            alpha = 1.0 - torch.exp(-F.relu(sigma) * dists)  # [chunk_size, N_samples]，[chunk_size, 64]

            # 计算透射率 T 与权重 weights
            # Compute weights
            T = torch.ones_like(alpha[:, :1])  # [chunk_size, 1] - 初始透射率
            weights = []  # 存储权重列表
            
            for i in range(alpha.shape[1]):  # 遍历N_samples，64 ===> 遍历每个采样点
                # 当前点的权重 = 透射率 × 不透明度
                weights.append(T * alpha[:, i:i+1])  # alpha[:, i:i+1]表示第i个点的不透明度，而“：”只是为了保持维度
                T = T * (1.0 - alpha[:, i:i+1] + 1e-10)  # 更新透射率: [chunk_size, 1]

            # [chunk_size, 64] - 合并所有权重 前面权重是按照点计算的，所以在dim=1上拼接
            weights = torch.cat(weights, dim=1)  # [chunk_size, N_samples]
            
            # Compute final RGB
            # rgb：[chunk_size, N_samples, 3]
            # weights.unsqueeze(-1)：[chunk_size, N_samples] ===> [chunk_size, N_samples，1]
            # [chunk_size, 3] 每个像素的rgb 而每个射线表示每个像素，所以dim=1上相加
            rgb_chunk = (weights.unsqueeze(-1) * rgb).sum(dim=1)
            rgb_final[chunk_start:chunk_end] = rgb_chunk  # 存入一个chunk的最终结果

            # 清理内存
            # Clear memory
            del points, view_dirs, outputs, rgb, sigma, weights
            torch.cuda.empty_cache()   # 清理GPU内存

        # 重塑和后处理
        # Reshape and post-process
        rgb_final = rgb_final.reshape(H, W, 3)  # [H, W, 3] - 重塑为图像格式
        rgb_final = torch.clamp(rgb_final, 0.0, 1.0)  # [H, W, 3] - 钳制到[0,1]范围
        
        return rgb_final  # 返回渲染图像: [H, W, 3]

# ========== 4. 在物体周围生成一圈（360°）相机位姿，用于环绕拍摄/出 GIF==========
def create_360_degree_poses(num_frames=120, radius=4.0, h=0.5):

    """Create camera poses for a 360-degree rotation around the object."""
    """
    =================创建围绕物体的360度旋转相机位姿===================
    参数:
    num_frames: 要生成的帧数（相机位姿数量），默认120帧
    radius: 相机距离原点的半径，默认4.0个单位
    h: 相机的基准高度，默认0.5个单位（在y轴方向上的偏移）
    返回:
    poses: 包含所有相机位姿的列表，每个位姿是一个字典，包含4x4变换矩阵
    """
    poses = []  # 存储所有位姿的列表，形状：最终为 [num_frames]，就是每个位姿一帧
    # 生成均匀分布的角度：从0°到360°，不包括360°
    for th in np.linspace(0., 360., num_frames, endpoint=False):
        # np.deg2rad() ===> 角度转弧度：θ = th × π/180
        theta = np.deg2rad(th)

        # 螺旋路径参数 - 固定倾斜角度30度（让相机稍微俯视）
        # Spiral path
        phi = np.deg2rad(30.0)  # Tilt angle  φ = 30 × π/180 ≈ 0.5236弧度


        # Camera position
            # ==================== 球坐标转笛卡尔坐标 ====================
            # 数学公式:
            # x = r × cos(θ) × cos(φ)
            # y = r × sin(φ) + h      （加上基准高度h）
            # z = r × sin(θ) × cos(φ)
        x = radius * np.cos(theta) * np.cos(phi)
        y = h + radius * np.sin(phi)  # Slight elevation
        z = radius * np.sin(theta) * np.cos(phi)
        
        # Look-at point (slightly above origin)
        # ==================== 定义相机参数 ====================
        # 观察点（稍微高于原点，让相机看向物体上方一点的位置）
        target = np.array([0, 0.2, 0])  # Look slightly above center [3] - 相机注视点 (0, 0.2, 0)
        eye = np.array([x, y, z])  # [3] - 相机位置 (x, y, z)
        up = np.array([0, 1, 0])  # [3] - 世界上方向参考 (0, 1, 0)
        
        # Create camera-to-world matrix
        # ==================== 创建相机矩阵 ====================
        # 创建相机到世界矩阵（look-at矩阵）
        c2w = look_at(eye, target, up)  # [3, 4] - 3x4变换矩阵
        # 将3x4矩阵转换为4x4齐次坐标矩阵
        # 齐次坐标公式: 添加一行 [0, 0, 0, 1]
        #  np.vstack() ===> 将数组沿垂直方向堆叠起来
        c2w = np.vstack([c2w, np.array([0, 0, 0, 1])])  # [4, 4]

        # 将位姿添加到列表
        poses.append({'transform_matrix': c2w})  # 每个元素是字典，包含4x4矩阵
    return poses  # 返回形状: [num_frames]，每个元素是包含4x4矩阵的字典

# =============根据相机位置 eye、注视点 target、参考上向量 up，构建相机到世界的 3×4 外参（[R|t]=================
def look_at(eye, target, up):
    """Create a look-at matrix."""
    # eye：相机位置  target: 相机注视点  up: 世界上方向参考
    forward = target - eye  # [3] 计算前向向量 f = t - e
    forward = forward / np.linalg.norm(forward)  # [3] 归一化 f := f / ||f||

    # p.cross()是向量叉积（外积）函数
    right = np.cross(forward, up)  # [3] 右向量 r = f × up
    right = right / np.linalg.norm(right)  # [3] 归一化 r
    
    up = np.cross(right, forward)  # [3] 重新计算正交的上向量 u' = r × f
    up = up / np.linalg.norm(up)  # [3] 归一化 u'
    
    rot = np.stack([right, up, -forward], axis=1)  # [3,3] 旋转部分 rot=[r, u', -f] 作为列向量
    trans = eye # [3]   平移部分就是相机位置 e

   # np.column_stack() ===> 用于将多个一维数组或二维数组按列堆叠（横向拼接）成一个二维数组
    return np.column_stack([rot, trans])  # [3,4] 返回 [R | t]，即 c2w

# ========== 5. 把一个文件夹里按文件名排序的 .png 序列合成为 GIF ==========
def create_gif(image_folder, output_path, duration=0.1):
    """Create a GIF from a folder of images."""  """从图像文件夹创建GIF动画"""
    images = []  # 存储图像列表
    # Sort files to ensure correct order
    # 排序文件以确保正确顺序
    files = sorted(os.listdir(image_folder))   # 获取排序后的文件列表
    for filename in files:  # 遍历所有文件
        if filename.endswith('.png'):  # 只处理PNG文件
            file_path = os.path.join(image_folder, filename)  # 完整文件路径
            images.append(imageio.imread(file_path))  # 读取图像并添加到列表

    # 保存为GIF
    # Save as GIF
    imageio.mimsave(output_path, images, duration=duration)   # 保存GIF动画
    print(f"GIF saved to {output_path}")  # 打印保存信息

# ========== 6. 从 transforms_test.json 读取每帧相机位姿与水平视场角，用于推理可视化 ==========
def load_test_poses(transforms_path):
    # transforms_test.json（Blender/NeRF 数据格式）通常包含：
    # 1、camera_angle_x（水平视场角，单位常为弧度），之后用于计算焦距
    # 2、frames：每帧里有 transform_matrix（通常为 4×4 相机到世界矩阵 c2w，外参，包含旋转矩阵和平移矩阵），以及文件路径
    """Load test poses from transforms.json file."""
    """从 transforms.json 文件加载测试位姿"""
    with open(transforms_path, 'r') as f:  # 打开JSON文件
        transforms = json.load(f)  # 解析JSON为 Python 字典
    
    frames = []  # 用于收集每一帧信息
    camera_angle_x = transforms.get('camera_angle_x', None)  # 获取相机视角角度，即水平视场角 θx
    
    for frame in transforms.get('frames', []):  # 遍历所有帧
        frames.append({
            'transform_matrix': np.array(frame['transform_matrix'], dtype=np.float32),  # [4,4] c2w
            'file_path': frame.get('file_path', None)  # 文件路径
        })
    
    return frames, camera_angle_x  # 返回帧列表和相机角度

def main():
    # 设置计算设备：优先使用GPU（CUDA），如果没有则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    # 初始化NeRF模型
    # pos_freqs=10: 位置编码的频率数（3D坐标的编码维度）
    # dir_freqs=4: 方向编码的频率数（观看方向的编码维度）
    model = NeRFModel(pos_freqs=10, dir_freqs=4).to(device)
    
    # Load checkpoint
    # 加载训练好的模型检查点（权重）
    # checkpoint_path: 保存的模型参数文件路径
    # **************************注意：这是 Linux 风格的硬编码路径，Windows 下通常不存在***********************
    checkpoint_path = '/teamspace/studios/this_studio/checkpoint_epoch_200.pth'
    # 从 .pth 加载权重到 model。若文件不存在会抛 FileNotFoundError
    model = load_checkpoint(model, checkpoint_path)
    
    # Create output directory
    # 创建输出目录
    render_dir = 'rendered_views'
    # exist_ok=True: 如果目录已存在则不报错
    os.makedirs(render_dir, exist_ok=True)
    
    # Load test transforms
    # 加载测试集的相机位姿（变换矩阵）
    # 计算工作空间根目录的路径
    # workspace_root: 通过向上三级目录得到“项目根”
    workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # 构建测试集变换文件的完整路径
    # 假定数据布局为 <root>/nerf_synthetic/chair/transforms_test.json
    transforms_path = os.path.join(workspace_root, 'nerf_synthetic', 'chair', 'transforms_test.json')
    # 加载测试帧和相机角度
    # frames: list，长度=N，每项含 'transform_matrix' [4,4]
    # camera_angle_x: float（弧度），水平视场角 θx
    frames, camera_angle_x = load_test_poses(transforms_path)
    
    # Calculate focal length from camera angle
    # 根据相机角度计算焦距
    # 图像高度和宽度，与训练时分辨率保持一致
    H = W = 400  # Match training resolution
    # focal: 焦距（像素单位），用于将3D点投影到2D图像平面
    # f = W / (2 * tan(θx/2))
    focal = W / (2 * np.tan(camera_angle_x / 2))

    # 开始渲染所有测试视角
    print(f"Rendering {len(frames)} test views...")
    for idx, frame in enumerate(frames):
        print(f"Rendering view {idx + 1}/{len(frames)}")

        # 形状 [4,4]，上 3×4 是 [R|t]，下行通常 [0,0,0,1]
        c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32, device=device)
        
        # Render using the correct focal length
        # 使用正确的焦距进行渲染
        # render_novel_view(),返回形状为[H, W, 3]的渲染好的图像
        # rgb_map: torch.float32，[H, W, 3]，范围约 [0,1]，三通道 RGB
        rgb_map = render_novel_view(model, c2w, H=H, W=W, focal=focal, device=device)

        # 将渲染结果从GPU移动到CPU，便于保存
        rgb_map = rgb_map.cpu()
        # 设置输出文件路径
        output_path = os.path.join(render_dir, f'view_{idx:03d}.png')
        # 保存渲染的图像
        # 这里传入 width=rgb_map.shape[1]==W，高度=rgb_map.shape[0]==H，路径=output_path
        save_rendered_image(rgb_map, rgb_map.shape[1], rgb_map.shape[0], output_path)
    
    print("Creating GIF from rendered views...")
    # 把 render_dir 下的 .png 合成 GIF，帧间隔 0.1s
    create_gif(render_dir, 'nerf_test_views.gif', duration=0.1)
    print("Done! Check nerf_test_views.gif for the final animation.")

if __name__ == '__main__':
    main() 