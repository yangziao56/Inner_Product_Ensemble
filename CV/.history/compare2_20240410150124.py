import torch
import numpy as np
# 加载tensor
tensor1 = torch.load('IP_save/IP0-0.1.pth')
tensor2 = torch.load('IP_save/IP0-0.01.pth')

# 确保两个tensor形状相同
if tensor1.shape != tensor2.shape:
    print("Tensors have different shapes.")
else:
    # 计算每个tensor中最小5%元素的indices
    num_elements = tensor1.numel()
    k = int(num_elements * 0.05)  # 计算5%的元素数量

    # 获取最小5%的indices
    _, indices1 = torch.topk(tensor1.view(-1), k, largest=False)
    _, indices2 = torch.topk(tensor2.view(-1), k, largest=False)

    # # 计算两组indices的交集
    # intersection = torch.intersect1d(indices1, indices2)
        # 将PyTorch张量转换为numpy数组
    indices1_np = np.array(indices1)
    indices2_np = np.array(indices2)

    # 使用numpy的intersect1d函数来找出交集
    intersection_np = np.intersect1d(indices1_np, indices2_np)

    # 将结果转换回PyTorch张量，如果需要的话
    intersection = torch.tensor(intersection_np)

    # 输出交集结果
    print(f"Common indices: {intersection}")
    
    # 输出结果
    print(f"Index of the smallest 5% in tensor1: {indices1}")
    print(f"Index of the smallest 5% in tensor2: {indices2}")
    print(f"Common indices: {intersection}")
    print(f"Overlap percentage: {100.0 * len(intersection) / k:.2f}%")
