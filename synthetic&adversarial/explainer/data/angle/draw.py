import numpy as np
import matplotlib.pyplot as plt

def normalize(vector):
    """Normalize a vector to unit length; return the original if zero length."""
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector

def plot_vectors(util_grad_hvp, util_loss_total_grad, train_indiv_grad, od_index):
    """Plot normalized vectors from origin with adaptive axis scaling."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 归一化并绘制基本向量
    util_grad_hvp_norm = normalize(util_grad_hvp)
    util_loss_total_grad_norm = normalize(util_loss_total_grad)
    ax.quiver(0, 0, util_grad_hvp_norm[0], util_grad_hvp_norm[1], angles='xy', scale_units='xy', scale=1, color='r', label='Utility Grad HVP (normalized)')
    ax.quiver(0, 0, util_loss_total_grad_norm[0], util_loss_total_grad_norm[1], angles='xy', scale_units='xy', scale=1, color='g', label='Utility Loss Total Grad (normalized)')

    vectors = [util_grad_hvp_norm, util_loss_total_grad_norm]

    # 绘制选定索引的向量
    for idx in od_index:
        indiv_grad_norm = normalize(train_indiv_grad[idx])
        ax.quiver(0, 0, indiv_grad_norm[0], indiv_grad_norm[1], angles='xy', scale_units='xy', scale=1, label=f'Train Indiv Grad {idx} (normalized)')
        vectors.append(indiv_grad_norm)

    # 调整坐标轴范围以包括所有向量
    max_extent = np.max(np.abs(vectors))
    ax.set_xlim(-max_extent-0.1, max_extent+0.1)
    ax.set_ylim(-max_extent-0.1, max_extent+0.1)

    # 设置图例和坐标轴比例
    ax.legend()
    ax.set_aspect('equal')

    # 设置标题和轴标签
    plt.title('Adaptively Resized Normalized Vector Visualization of Gradients')
    plt.xlabel('Gradient in x (normalized)')
    plt.ylabel('Gradient in y (normalized)')

    # 显示图形
    plt.show()




# 示例数据加载
util_grad_hvp = np.load('util_grad_hvp.npy')
util_loss_total_grad = np.load('util_loss_total_grad.npy')
train_indiv_grad = np.load('train_indiv_grad.npy')
influence_values_util = np.load('util_infl_lr.npy')
od_index = np.argsort(influence_values_util)[:10]

util_grad_hvp_nn = np.load('util_grad_hvp_nn.npy')
util_loss_total_grad_nn = np.load('util_loss_total_grad_nn.npy')
train_indiv_grad_nn = np.load('train_indiv_grad_nn.npy')

od_index_nn = np.load('od_idxs.npy').tolist()
print(util_grad_hvp_nn.shape)

# 调用函数绘图
plot_vectors(util_grad_hvp, util_loss_total_grad, train_indiv_grad, od_index)
#plot_vectors(util_loss_total_grad_nn, util_loss_total_grad, train_indiv_grad_nn, od_index_nn)
