from PIL import Image  # 图像处理库，用于保存图片
import torch
import numpy as np
import torch.nn.functional as F


# ChatGPT Generated Code
# --------------------------------------------------
# 下面这个函数只负责“保存图片”，不是真正的渲染过程
# --------------------------------------------------
def save_rendered_image(rendered_colors, image_width, image_height, output_path):
    """
    Save the rendered image to a file.

    Args:
        rendered_colors (torch.Tensor): Rendered colors (N_rays, 3).
        image_width (int): Width of the output image.
        image_height (int): Height of the output image.
        output_path (str): Path to save the output image.
    """
    """
    将渲染的图像保存到文件。

    Args:
        rendered_colors (torch.Tensor): 渲染得到的颜色。
            - 期望形状为 (N_rays, 3) 或 (H, W, 3)，值域通常在 [0,1]。
              若是 (N_rays,3)，其中 N_rays 应该等于 image_width * image_height。
        image_width (int): 输出图像的宽 W（像素）。
        image_height (int): 输出图像的高 H（像素）。
        output_path (str): 输出文件路径（例如 "output.png"）。

    处理流程：
        1) reshape 到 (H,W,3)
        2) 从 torch 张量转 numpy 数组
        3) 映射到 [0,255] 并转为 uint8
        4) 用 Pillow 落盘
    """
    # Reshape the rendered colors to match the image dimensions
    # 将渲染的颜色重塑为图像尺寸
    # rendered_colors: [H*W, 3] → [H, W, 3]
    image = rendered_colors.reshape(image_height, image_width, 3).cpu().numpy()
    # 将浮点数颜色值（0-1范围）转换为8位整数（0-255范围）
    image = (image * 255).astype(np.uint8)  # Convert to 8-bit color

    # Save the image using Pillow
    # 用 Pillow 保存
    img = Image.fromarray(image)  # 创建 PIL 图像
    img.save(output_path)  # 保存
    print(f"Rendered image saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Dummy NeRF model
    # 虚拟NeRF模型（用于演示）测试渲染流程是否正常工作
    # 作用：一个简单的替代模型，用于演示目的
    #
    # 输出：对输入的前3个维度应用sigmoid作为颜色，密度固定为1
    #
    # 为什么需要：在没有训练好的真实NeRF模型时，可以用这个测试渲染流程
    class DummyNeRF(torch.nn.Module):
        def forward(self, x):
            # 返回虚拟的颜色和密度值
            return torch.cat([torch.sigmoid(x[:, :3]), torch.ones(x.shape[0], 1, device=x.device)], dim=-1)

    # Initialize model and inputs
    model = DummyNeRF().to("cuda")   # 初始化模型和输入
    image_width, image_height = 32, 32  # 图像尺寸
    ray_origins = torch.rand(image_width * image_height, 3, device="cuda")  # 随机射线起点
    ray_directions = torch.rand(image_width * image_height, 3, device="cuda").normalize(dim=-1)  # 随机方向（标准化）
    near, far = 0.1, 4.0
    num_samples = 64  # 每条射线采样点数

    # Render rays
    # 渲染射线
    rendered_image = render_rays(ray_origins, ray_directions, model, near, far, num_samples, device="cuda")

    # Save the rendered image
    # 保存渲染的图像
    save_rendered_image(rendered_image, image_width, image_height, "output.png")
