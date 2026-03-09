file_path = "outputs/train/dev_task1/rgb_depth_act/run_20251119_094752/policy_preprocessor_step_3_normalizer_processor.safetensors"

from safetensors import safe_open
import torch

# 打开safetensor文件

with safe_open(file_path, framework="pt", device="cpu") as f:
    # 获取所有张量名称
    tensor_names = f.keys()
    print("所有张量名称：")
    for name in tensor_names:
        print(f"- {name}")
    print("\n张量详细信息：")
    # 查看每个张量的详细信息
    for name in tensor_names:
        if name in ["action.max","action.min","action.mean","action.std"]:
            tensor = f.get_tensor(name)
            print(f"名称: {name}")
            print(f"  形状: {tensor.shape}")
            print(f"  数据类型: {tensor.dtype}")
            print(f"  值范围: [{tensor.min():.6f}, {tensor.max():.6f}]")
            print(f"  均值: {tensor.mean():.6f}")
            
            # 对于小张量，可以直接打印值
            if tensor.numel() <= 50:  # 元素数量小于等于10
                print(f"  值: {tensor}")
            print("-" * 50)