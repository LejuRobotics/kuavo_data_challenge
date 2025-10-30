from diffusers import FlowMatchEulerDiscreteScheduler
import torch

class CustomFlowMatchEulerDiscreteSchedulerWrapper(FlowMatchEulerDiscreteScheduler):
    """
    在 FlowMatchEulerDiscreteScheduler 基础上，添加 add_noise 方法
    用于训练阶段生成带噪 x_t
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        """
        模拟原 DDPM 的 add_noise 接口
        x0: 原始样本, shape [B, C, H, W]
        noise: 标准高斯噪声, shape [B, C, H, W]
        t: 时间步, shape [B], 范围 [0, 1]
        """
        # 扩展 t 维度匹配 x0
        # t = t[:, None, None, None]
        # 线性混合
        x_t = (1 - t) * x0 + t * noise
        return x_t
