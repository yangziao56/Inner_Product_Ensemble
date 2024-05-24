import torch

# 加载tensor
tensor1 = torch.load('IP_save/grad_1.pth')
tensor2 = torch.load('IP_save/grad1.pth')

# 确保两个tensor的形状相同
if tensor1.shape != tensor2.shape:
    print("Tensors have different shapes.")
else:
    # 计算两个tensor的差异
    difference = tensor1 - tensor2
    
    # 检查差异
    if torch.all(difference == 0):
        print("两个tensor完全相同。")
    else:
        print("两个tensor有差异。")
        print("差异：", difference)
