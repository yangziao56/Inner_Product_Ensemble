import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as PathEffects
import torch
import torch.nn as nn
import sklearn.linear_model
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches



n_bins = 5
size = 19
front_size = 22
alpha1 = 1
alpha2 = 0.3
alpha_minority = 1
w1 = np.load('explainer/data/binaries/w1.npy')
w2 = np.load('explainer/data/binaries/w2.npy')
b1 = np.load('explainer/data/binaries/b.npy')
log_reg = sklearn.linear_model.LogisticRegression()
log_reg.coef_ = np.array([[w1, w2]])
log_reg.intercept_ = np.array([b1])
log_reg.classes_ = np.array([0, 1])

w1_utility = np.load('explainer/data/binaries/w1_utility.npy')
w2_utility = np.load('explainer/data/binaries/w2_utility.npy')
b_utility = np.load('explainer/data/binaries/b_utility.npy')
log_reg_utility = sklearn.linear_model.LogisticRegression()
log_reg_utility.coef_ = np.array([[w1_utility, w2_utility]])
log_reg_utility.intercept_ = np.array([b_utility])
log_reg_utility.classes_ = np.array([0, 1])

w1_fairness = np.load('explainer/data/binaries/w1_fairness.npy')
w2_fairness = np.load('explainer/data/binaries/w2_fairness.npy')
b_fairness = np.load('explainer/data/binaries/b_fairness.npy')
log_reg_fairness = sklearn.linear_model.LogisticRegression()
log_reg_fairness.coef_ = np.array([[w1_fairness, w2_fairness]])
log_reg_fairness.intercept_ = np.array([b_fairness])
log_reg_fairness.classes_ = np.array([0, 1])

w1_robustness = np.load('explainer/data/binaries/w1_robustness.npy')
w2_robustness = np.load('explainer/data/binaries/w2_robustness.npy')
b_robustness = np.load('explainer/data/binaries/b_robustness.npy')
log_reg_robustness = sklearn.linear_model.LogisticRegression()
log_reg_robustness.coef_ = np.array([[w1_robustness, w2_robustness]])
log_reg_robustness.intercept_ = np.array([b_robustness])
log_reg_robustness.classes_ = np.array([0, 1])



class MLPClassifier(nn.Module):
    def __init__(self, input_dim=2, factor=1):
        super(MLPClassifier, self).__init__()
        self.layer = nn.Sequential(
            #nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),
            nn.Linear(32, 32, bias=True),
            nn.ReLU(),

            # nn.Linear(input_dim // factor, input_dim // factor, bias=True),
            # nn.ReLU(),
            # nn.Linear(input_dim // factor, input_dim // factor, bias=True),
            # nn.ReLU(),
            # nn.Linear(input_dim // factor, input_dim // factor, bias=True),
            # nn.ReLU(),
            # nn.Linear(input_dim // factor, input_dim // factor, bias=True),
            # nn.ReLU(),
            # nn.Linear(input_dim // factor, input_dim // factor, bias=True),
            # nn.ReLU(),
            # nn.Linear(input_dim // factor, input_dim // factor, bias=True),
            # nn.ReLU(),
        )

        self.last_fc = nn.Linear(32, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer(x)
        x = self.sigmoid(self.last_fc(x))
        return x.squeeze()

    def emb(self, x):
        x = self.layer(x)
        return x
    
    def reset_parameters(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    @property
    def num_parameters(self):
        return self.last_fc.weight.nelement()
mlp = MLPClassifier(input_dim=2)
mlp_new = MLPClassifier(input_dim=2)
# 加载模型状态
model_state = torch.load('explainer/data/binaries/mlp.pth')
model_state_new = torch.load('explainer/data/binaries/mlp_new.pth')

# 应用模型状态
mlp.load_state_dict(model_state)
mlp_new.load_state_dict(model_state_new)


# 设置为评估模式
mlp.eval()
def standardize_data(X, mean=None, std=None):
    """
    对输入数据进行标准化处理：中心化（减去均值）和缩放（除以标准差）。

    Parameters:
    X (np.ndarray): 输入数据，形状为 (n_samples, n_features)。
    mean (np.ndarray, optional): 特征的平均值。如果为 None，则计算 X 的平均值。
    std (np.ndarray, optional): 特征的标准差。如果为 None，则计算 X 的标准差。

    Returns:
    np.ndarray: 标准化后的数据。
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
    
    # 防止除以0
    std_replaced = np.where(std == 0, 1, std)
    
    X_standardized = (X - mean) / std_replaced
    return X_standardized



# 假设这是您的绘图函数，每个都接受一个matplotlib的axes对象作为参数
def plot_training_toy_Utility (ax):
    # Load the dataset
    df = pd.read_csv('data/toy/train.csv', header=None)
    f1, f2, y = df[0].to_list(), df[1].to_list(), df[3].to_list()
    majority_minority = df[2] # 0: majority, 1: minority
    influence_values = np.load('explainer/data/binaries/util_infl_lr.npy')
    # Define the color map
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}
    
    # Initialize the plot
    # fig, ax = plt.subplots(figsize=(6.8, 6))

    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-5, 9)
    ax.set_ylim(-7, 9)

    # 创建网格，用于绘制决策边界
    x_min, x_max = df[0].min() - 5, df[0].max() + 5
    y_min, y_max = df[1].min() - 5, df[1].max() + 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    test_mean = np.mean(np.array([f1, f2]), axis=1)
    test_std = np.std(np.array([f1, f2]), axis=1)
    grid_standardized = standardize_data(grid, mean=test_mean, std=test_std)
    #grid_tensor = torch.FloatTensor(grid_standardized)
    
    # 用log_reg预测并绘制决策边界
    Z = log_reg.predict_proba(grid_standardized)[:, 1].reshape(xx.shape)
    Z = Z > 0.5  # 假设使用0.5作为分类阈值
    ax.contourf(xx, yy, Z, alpha=alpha2, levels=[-1, 0, 1], cmap='coolwarm')
    
    drawn_as_outlier = set()

    # Highlight outliers(influence_value<0) with a different marker
    for i, (a, b) in enumerate(zip(f1, f2)):
        #最小的10个 influence_values的点是outlier
        sorted_infl = np.sort(influence_values)
        if influence_values[i] < sorted_infl[10]:
            #ax.scatter(a, b, marker='X', color=y_colormap[y[i]], s=500, edgecolor='black')
            drawn_as_outlier.add(i)
        else:
            ax.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200, edgecolor='black', alpha=alpha1)

    for i in drawn_as_outlier:
        ax.scatter(f1[i], f2[i], marker='X', color=y_colormap[y[i]], s=500, edgecolor='black', alpha=alpha1)
    # Set labels
    ax2 = ax.twinx()
    ax2.yaxis.set_label_position("left")
    ax2.spines["top"].set_position(("axes", 1))
    ax2.xaxis.set_label_position("top")  # 确认标签位置在顶部
    ax2.tick_params(axis='both',          # 应用到y轴
               which='both',      # 应用到主要和次要刻度
               left=False,        # 隐藏左侧刻度
               right=False,       # 隐藏右侧刻度
               labelleft=False,   # 隐藏左侧刻度标签
               labelright=False)   # 显示右侧刻度标签
    ax2.set_title('Training', fontsize=front_size, fontweight='bold')
    ax2.set_ylabel('Linear-Utility', fontsize=front_size,labelpad=40, fontweight='bold')
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Feature 1', fontsize=front_size)
    ax.set_ylabel('Feature 2', fontsize=front_size, rotation=270, labelpad=25)

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
    # plt.axis('equal')
    # Adjust layout to make room for the full feature 2 label

    ax.xaxis.set_major_locator(ticker.FixedLocator([-4,-2,0,2, 4,6,8]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-6,-4,-2,0,2, 4,6,8]))
    fig.tight_layout()
    
    # Save plot
    # ax.savefig('figures/training_toy_utility.pdf', format='pdf')
    #plt.close()  # Close the plot to free memory

def plot_test_toy_utility(ax):
    # Load the test dataset
    df = pd.read_csv('data/toy/test.csv', header=None)
    majority_minority = df[2]
    f1, f2, y_te = df[0].to_list(), df[1].to_list(), df[3].to_list()
    
    # # Initialize the plot
    # plt.figure(figsize=(6.8, 6))
    
    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-5, 9)
    ax.set_ylim(-7, 9)

    # 创建网格，用于绘制决策边界
    x_min, x_max = df[0].min() - 5, df[0].max() + 5
    y_min, y_max = df[1].min() - 5, df[1].max() + 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    test_mean = np.mean(np.array([f1, f2]), axis=1)
    test_std = np.std(np.array([f1, f2]), axis=1)
    grid_standardized = standardize_data(grid, mean=test_mean, std=test_std)
    #grid_tensor = torch.FloatTensor(grid_standardized)
    
    # 用log_reg预测并绘制决策边界
    Z = log_reg_utility.predict_proba(grid_standardized)[:, 1].reshape(xx.shape)
    Z = Z > 0.5  # 假设使用0.5作为分类阈值
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap='coolwarm', alpha=alpha2)

    # Define the color map for the test set classes
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}
    
    # Create scatter plot for each point in the test dataset
    for i, (a, b) in enumerate(zip(f1, f2)):
        if majority_minority[i] == 'Female':  # majority实框
            ax.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200, edgecolor='black', alpha=alpha1)
        else:  # minority虚框
            ax.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200, edgecolor='black', alpha=alpha_minority)
            #ax.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200, edgecolors='black', linestyle='--', alpha=alpha1)
        #plt.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200)

    # Set labels for the axes
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Feature 1', fontsize=front_size)
    ax.set_ylabel('Feature 2', fontsize=front_size, rotation=270, labelpad=25)

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
    #plt.axis('equal')
    # Adjust layout to make room for the full feature 2 label
    ax.xaxis.set_major_locator(ticker.FixedLocator([-4,-2,0,2, 4,6,8]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-6,-4,-2,0,2, 4,6,8]))
    plt.tight_layout()

    # Save plot
    # plt.savefig('figures/test_toy_utility.pdf', format='pdf')
    # plt.close()  # Close the plot to free memory

def plot_influence_inner_product_LR_Utility(ax):
    # Load influence values and inner products
    influence_values = np.load('explainer/data/binaries/util_infl_lr.npy')
    inner_products = np.load('explainer/data/binaries/util_infl_wo_hess_lr.npy')
    
    # Load class labels
    df = pd.read_csv('data/toy/train.csv', header=None)
    labels = df[3].to_list()  # Assuming the labels are in the fourth column
    
    # Define color map based on the classes
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}
    
    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)

    # Load outliers indices
    outliers_indices = np.load('data/half_moons/od_idxs.npy').tolist()
    
    # # Initialize the plot
    # plt.figure(figsize=(7, 6))
    
    # Keep track of which points are outliers to skip drawing them as 'o'
    drawn_as_outlier = set()
    # 将网格设置为底层
    ax.set_axisbelow(True)

    # # Highlight outliers with a different marker
    # for outlier in outliers_indices:
    #     if influence_values[outlier] > -6000:
    #         plt.scatter(influence_values[outlier], inner_products[outlier], marker='X', color=y_colormap[labels[outlier]], s=500, edgecolor='black')
    #     drawn_as_outlier.add(outlier)  # Mark this point as drawn as an outlier

    #if influence_values<0, that point will be outlier
    #最小的10个 influence_values的点是outlier
    for i, (infl, inner_prod) in enumerate(zip(influence_values, inner_products)):
        #最小的10个 influence_values的点是outlier
        sorted_infl = np.sort(influence_values)
        if infl < sorted_infl[10]:
            #plt.scatter(infl, inner_prod, marker='X', color=y_colormap[labels[i]], s=500, edgecolor='black')
            drawn_as_outlier.add(i)  # Mark this point as drawn as an outlier
        else:
            ax.scatter(infl, inner_prod, marker='o', color=y_colormap[labels[i]], s=200, alpha=alpha1, edgecolor='black')
    
    for i in drawn_as_outlier:
        ax.scatter(influence_values[i], inner_products[i], marker='X', color=y_colormap[labels[i]], s=500, edgecolor='black', alpha=alpha1)
    
    # # Scatter plot for each point, skip the ones already drawn as outliers
    # for i, (infl, inner_prod) in enumerate(zip(influence_values, inner_products)):
    #     if i not in drawn_as_outlier:  # Only draw if not already drawn as an outlier
    #         plt.scatter(infl, inner_prod, marker='o', color=y_colormap[labels[i]], s=200)
    
    # Set labels and title
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Influence Value', fontsize=front_size)
    ax.set_ylabel('Inner Product', fontsize=front_size, rotation=270, labelpad=25)
    
    

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=n_bins))
    ax.xaxis.set_major_locator(ticker.FixedLocator([-2,-1,0, 1,2]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-2,-1,0, 1,2]))
    ax.grid(True, linestyle='--')  # Enable grid
    #plt.axis('equal')
    # Adjust layout and show the plot
    plt.tight_layout()

    # # Save plot
    # plt.savefig('figures/influence_inner_product_LR_Utility.pdf', format='pdf')
    # #plt.show()
    # plt.close()  # Close the plot to free memory

def plot_training_toy_Fariness (ax):
    # Load the dataset
    df = pd.read_csv('data/toy/train.csv', header=None)
    f1, f2, y = df[0].to_list(), df[1].to_list(), df[3].to_list()
    majority_minority = df[2] # 0: majority, 1: minority
    influence_values = np.load('explainer/data/binaries/fair_infl_lr.npy')
    # Define the color map
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}
    
    # # Initialize the plot
    # plt.figure(figsize=(6.8, 6))

    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-5, 9)
    ax.set_ylim(-7, 9)

    # 创建网格，用于绘制决策边界
    x_min, x_max = df[0].min() - 5, df[0].max() + 5
    y_min, y_max = df[1].min() - 5, df[1].max() + 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    test_mean = np.mean(np.array([f1, f2]), axis=1)
    test_std = np.std(np.array([f1, f2]), axis=1)
    grid_standardized = standardize_data(grid, mean=test_mean, std=test_std)
    #grid_tensor = torch.FloatTensor(grid_standardized)
    
    # 用log_reg预测并绘制决策边界
    Z = log_reg.predict_proba(grid_standardized)[:, 1].reshape(xx.shape)
    Z = Z > 0.5  # 假设使用0.5作为分类阈值
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap='coolwarm', alpha=alpha2)
    
    drawn_as_outlier = set()
    # Scatter plot for each point in the dataset
    # for i, (a, b) in enumerate(zip(f1, f2)):
    #     plt.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200)
    
    # # Highlight outliers with a different marker
    # indices = np.load('data/half_moons/od_idxs.npy').tolist()
    # for i, (a, b) in enumerate(zip(f1, f2)):
    #     if i in indices:
    #         plt.scatter(a, b, marker='X', color=y_colormap[y[i]], s=500, edgecolor='black')
    
    # Highlight outliers(influence_value<0) with a different marker
    for i, (a, b) in enumerate(zip(f1, f2)):
        #最小的10个 influence_values的点是outlier
        sorted_infl = np.sort(influence_values)
        if influence_values[i] < sorted_infl[10]:
            #plt.scatter(a, b, marker='X', color=y_colormap[y[i]], s=500, edgecolor='black')
            drawn_as_outlier.add(i)
        else:
            if majority_minority[i] == 'Female':  # majority实框
                ax.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200, edgecolor='black', alpha=alpha1)
            else:  # minority虚框
                ax.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200, edgecolors='black', linestyle='--', alpha=alpha1)
            #plt.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200)
    
    for i in drawn_as_outlier:
        if majority_minority[i] == 'Female':  # majority实框
            ax.scatter(f1[i], f2[i], marker='X', color=y_colormap[y[i]], s=500, edgecolor='black', alpha=alpha1)
        else:  # minority虚框
            ax.scatter(f1[i], f2[i], marker='X', color=y_colormap[y[i]], s=500, edgecolors='black', linestyle='--', alpha=alpha1)


    # Set labels
    ax2 = ax.twinx()
    ax2.yaxis.set_label_position("left")
    ax2.tick_params(axis='y',          # 应用到y轴
               which='both',      # 应用到主要和次要刻度
               left=False,        # 隐藏左侧刻度
               right=False,       # 隐藏右侧刻度
               labelleft=False,   # 隐藏左侧刻度标签
               labelright=False)   # 显示右侧刻度标签
    ax2.set_ylabel('Linear-Fairness', fontsize=front_size,labelpad=40, fontweight='bold')
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Feature 1', fontsize=front_size)
    ax.set_ylabel('Feature 2', fontsize=front_size, rotation=270, labelpad=25)

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
    # plt.axis('equal')
    # Adjust layout to make room for the full feature 2 label
    ax.xaxis.set_major_locator(ticker.FixedLocator([-4,-2,0,2, 4,6,8]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-6,-4,-2,0,2, 4,6,8]))
    plt.tight_layout()
    
    # # Save plot
    # plt.savefig('figures/training_toy_fairness.pdf', format='pdf')
    # plt.close()  # Close the plot to free memory

def plot_test_toy_fairness(ax):
    # Load the test dataset
    df = pd.read_csv('data/toy/test.csv', header=None)
    majority_minority = df[2]
    f1, f2, y_te = df[0].to_list(), df[1].to_list(), df[3].to_list()
    
    # # Initialize the plot
    # plt.figure(figsize=(6.8, 6))
    
    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-5, 9)
    ax.set_ylim(-7, 9)

    # 创建网格，用于绘制决策边界
    x_min, x_max = df[0].min() - 5, df[0].max() + 5
    y_min, y_max = df[1].min() - 5, df[1].max() + 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    test_mean = np.mean(np.array([f1, f2]), axis=1)
    test_std = np.std(np.array([f1, f2]), axis=1)
    grid_standardized = standardize_data(grid, mean=test_mean, std=test_std)
    #grid_tensor = torch.FloatTensor(grid_standardized)
    
    # 用log_reg预测并绘制决策边界
    Z = log_reg_fairness.predict_proba(grid_standardized)[:, 1].reshape(xx.shape)
    Z = Z > 0.5  # 假设使用0.5作为分类阈值
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap='coolwarm', alpha=alpha2)

    # Define the color map for the test set classes
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}
    
    # Create scatter plot for each point in the test dataset
    for i, (a, b) in enumerate(zip(f1, f2)):
        if majority_minority[i] == 'Female':  # majority实框
            ax.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200, edgecolor='black', alpha=alpha1)
        else:  # minority虚框
            ax.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200, edgecolors='black', linestyle='--', alpha=alpha1)
        #plt.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200)

    # Set labels for the axes
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Feature 1', fontsize=front_size)
    ax.set_ylabel('Feature 2', fontsize=front_size, rotation=270, labelpad=25)

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
    #plt.axis('equal')
    # Adjust layout to make room for the full feature 2 label
    ax.xaxis.set_major_locator(ticker.FixedLocator([-4,-2,0,2, 4,6,8]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-6,-4,-2,0,2, 4,6,8]))
    plt.tight_layout()

    # # Save plot
    # plt.savefig('figures/test_toy_fairness.pdf', format='pdf')
    # plt.close()  # Close the plot to free memory

def plot_influence_inner_product_LR_Fairness(ax):
    # Load influence values and inner products
    influence_values = np.load('explainer/data/binaries/fair_infl_lr.npy')
    inner_products = np.load('explainer/data/binaries/fair_infl_wo_hess_lr.npy')
    
    # Load class labels
    df = pd.read_csv('data/toy/train.csv', header=None)
    labels = df[3].to_list()  # Assuming the labels are in the fourth column
    
    # Define color map based on the classes
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}

    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-0.03, 0.03)
    #ax.set_ylim(-0.04, 0.05)
    ax.set_ylim(-0.03, 0.03)
    
    # Load outliers indices
    outliers_indices = np.load('data/half_moons/od_idxs.npy').tolist()
    
    # # Initialize the plot
    # plt.figure(figsize=(7, 6))
    
    # 将网格设置为底层
    ax.set_axisbelow(True)
    
    # Keep track of which points are outliers to skip drawing them as 'o'
    drawn_as_outlier = set()

    # # Highlight outliers with a different marker
    # for outlier in outliers_indices:
    #     if influence_values[outlier] > -6000:
    #         plt.scatter(influence_values[outlier], inner_products[outlier], marker='X', color=y_colormap[labels[outlier]], s=500, edgecolor='black')
    #     drawn_as_outlier.add(outlier)  # Mark this point as drawn as an outlier
    
    # Scatter plot for each point, skip the ones already drawn as outliers
    # for i, (infl, inner_prod) in enumerate(zip(influence_values, inner_products)):
    #     if i not in drawn_as_outlier:  # Only draw if not already drawn as an outlier
    #         plt.scatter(infl, inner_prod, marker='o', color=y_colormap[labels[i]], s=200)
    
    for i, (infl, inner_prod) in enumerate(zip(influence_values, inner_products)):
        sorted_infl = np.sort(influence_values)
        if infl < sorted_infl[10]:
            #plt.scatter(infl, inner_prod, marker='X', color=y_colormap[labels[i]], s=500, edgecolor='black')
            drawn_as_outlier.add(i)  # Mark this point as drawn as an outlier
        else:
            ax.scatter(infl, inner_prod, marker='o', color=y_colormap[labels[i]], s=200, alpha=alpha1)
    
    for i in drawn_as_outlier:
        ax.scatter(influence_values[i], inner_products[i], marker='X', color=y_colormap[labels[i]], s=500, edgecolors='black', alpha=alpha1)

    # Set labels and title
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Influence Value', fontsize=front_size)
    ax.set_ylabel('Inner Product', fontsize=front_size, rotation=270, labelpad=25)
    
    # 设置x轴和y轴使用科学计数法
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # 可选：强制所有刻度都使用科学计数法（如果matplotlib没有自动选择）
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    # 调整科学计数法中 10 的指数部分的字体大小
    ax.xaxis.get_offset_text().set_fontsize(15) # 设置 x 轴偏移文本的字体大小
    ax.yaxis.get_offset_text().set_fontsize(15) # 设置 y 轴偏移文本的字体大小

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=n_bins))
    ax.xaxis.set_major_locator(ticker.FixedLocator([-0.02,-0.01,0, 0.01,0.02]))
    #ax.yaxis.set_major_locator(ticker.FixedLocator([-0.01, 0.02]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-0.02,-0.01,0, 0.01,0.02]))
    ax.grid(True, linestyle='--')  # Enable grid
    #plt.axis('equal')
    # Adjust layout and show the plot
    plt.tight_layout()

def plot_training_toy_Robutness (ax):
    # Load the dataset
    df = pd.read_csv('data/toy/train.csv', header=None)
    f1, f2, y = df[0].to_list(), df[1].to_list(), df[3].to_list()
    majority_minority = df[2] # 0: majority, 1: minority
    influence_values = np.load('explainer/data/binaries/robust_infl_lr.npy')
    # Define the color map
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}
    
    # # Initialize the plot
    # plt.figure(figsize=(6.8, 6))

    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-5, 9)
    ax.set_ylim(-7, 9)

    # 创建网格，用于绘制决策边界
    x_min, x_max = df[0].min() - 5, df[0].max() + 5
    y_min, y_max = df[1].min() - 5, df[1].max() + 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    test_mean = np.mean(np.array([f1, f2]), axis=1)
    test_std = np.std(np.array([f1, f2]), axis=1)
    grid_standardized = standardize_data(grid, mean=test_mean, std=test_std)
    #grid_tensor = torch.FloatTensor(grid_standardized)
    
    # 用log_reg预测并绘制决策边界
    Z = log_reg.predict_proba(grid_standardized)[:, 1].reshape(xx.shape)
    Z = Z > 0.5  # 假设使用0.5作为分类阈值
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap='coolwarm', alpha=alpha2)

    
    drawn_as_outlier = set()
    # Scatter plot for each point in the dataset
    # for i, (a, b) in enumerate(zip(f1, f2)):
    #     plt.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200)
    
    # # Highlight outliers with a different marker
    # indices = np.load('data/half_moons/od_idxs.npy').tolist()
    # for i, (a, b) in enumerate(zip(f1, f2)):
    #     if i in indices:
    #         plt.scatter(a, b, marker='X', color=y_colormap[y[i]], s=500, edgecolor='black')
    
    # Highlight outliers(influence_value<0) with a different marker
    for i, (a, b) in enumerate(zip(f1, f2)):
        #最小的10个 influence_values的点是outlier
        sorted_infl = np.sort(influence_values)
        if influence_values[i] < sorted_infl[10]:
            #plt.scatter(a, b, marker='X', color=y_colormap[y[i]], s=500, edgecolor='black')
            drawn_as_outlier.add(i)
        else:
            if majority_minority[i] == 'Female':  # majority实框
                ax.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200, edgecolor='black', alpha=alpha1)
            else:  # minority虚框
                ax.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200, edgecolors='black', linestyle='--', alpha=alpha1)
            #plt.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200)
    
    for i in drawn_as_outlier:
        if majority_minority[i] == 'Female':  # majority实框
            ax.scatter(f1[i], f2[i], marker='X', color=y_colormap[y[i]], s=500, edgecolor='black', alpha=alpha1)
        else:
            ax.scatter(f1[i], f2[i], marker='X', color=y_colormap[y[i]], s=500, edgecolors='black', linestyle='--', alpha=alpha1)
    
    # Set labels
    ax2 = ax.twinx()
    ax2.yaxis.set_label_position("left")
    ax2.tick_params(axis='y',          # 应用到y轴
               which='both',      # 应用到主要和次要刻度
               left=False,        # 隐藏左侧刻度
               right=False,       # 隐藏右侧刻度
               labelleft=False,   # 隐藏左侧刻度标签
               labelright=False)   # 显示右侧刻度标签
    ax2.set_ylabel('Linear-Robutness', fontsize=front_size,labelpad=40, fontweight='bold')
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Feature 1', fontsize=front_size)
    ax.set_ylabel('Feature 2', fontsize=front_size, rotation=270, labelpad=25)

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
    # plt.axis('equal')
    # Adjust layout to make room for the full feature 2 label
    ax.xaxis.set_major_locator(ticker.FixedLocator([-4,-2,0,2, 4,6,8]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-6,-4,-2,0,2, 4,6,8]))
    plt.tight_layout()
    
    # # Save plot
    # plt.savefig('figures/training_toy_robutness.pdf', format='pdf')
    # plt.close()  # Close the plot to free memory

def plot_advtest_toy(ax):
    # Load the test dataset
    df = pd.read_csv('data/toy/test.csv', header=None)
    adv_val = np.load('xadv_val.npy')
    adv_test = np.load('xadv_test.npy')
    #f1,f2 = adv_val[:,0],adv_val[:,1]
    #f1, f2, y_te = df[0].to_list(), df[1].to_list(), df[3].to_list()
    f1, f2, y_te = adv_val[:,0], adv_val[:,1], df[3].to_list()
    f1, f2 = adv_test[:,0], adv_test[:,1]
    majority_minority = df[2]

    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-5, 9)
    ax.set_ylim(-7, 9)

    # 创建网格，用于绘制决策边界
    x_min, x_max = df[0].min() - 5, df[0].max() + 5
    y_min, y_max = df[1].min() - 5, df[1].max() + 5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    test_mean = np.mean(np.array([f1, f2]), axis=1)
    test_std = np.std(np.array([f1, f2]), axis=1)
    grid_standardized = standardize_data(grid, mean=test_mean, std=test_std)
    #grid_tensor = torch.FloatTensor(grid_standardized)
    
    # 用log_reg预测并绘制决策边界
    Z = log_reg_robustness.predict_proba(grid_standardized)[:, 1].reshape(xx.shape)
    Z = Z > 0.5  # 假设使用0.5作为分类阈值
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap='coolwarm', alpha=alpha2)
    
    # Define the color map for the test set classes
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}
    
    # Create scatter plot for each point in the test dataset
    for i, (a, b) in enumerate(zip(f1, f2)):
        if majority_minority[i] == 'Female':  # majority实框
            ax.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200, edgecolor='black', alpha=alpha1)
        else:  # minority虚框
            ax.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200, edgecolors='black', linestyle='--', alpha=alpha1)
        #plt.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200)
    

    # Set labels for the axes
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Feature 1', fontsize=front_size)
    ax.set_ylabel('Feature 2', fontsize=front_size, rotation=270, labelpad=25)

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=3))
    #plt.axis('equal')
    # Adjust layout to make room for the full feature 2 label
    ax.xaxis.set_major_locator(ticker.FixedLocator([-4,-2,0,2, 4,6,8]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-6,-4,-2,0,2, 4,6,8]))
    plt.tight_layout()


def plot_influence_inner_product_LR_Robutness(ax):
    # Load influence values and inner products
    influence_values = np.load('explainer/data/binaries/robust_infl_lr.npy')
    inner_products = np.load('explainer/data/binaries/robust_infl_wo_hess_lr.npy')
    
    # Load class labels
    df = pd.read_csv('data/toy/train.csv', header=None)
    labels = df[3].to_list()  # Assuming the labels are in the fourth column
    
    # Define color map based on the classes
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}

    # Set the same scale for x and y axis as in the toy plot
    #ax.set_xlim(-23, 22)
    ax.set_xlim(-24, 24)
    #ax.set_ylim(-25, 20)
    ax.set_ylim(-24, 24)
    # 将网格设置为底层
    ax.set_axisbelow(True)
    
    # Load outliers indices
    outliers_indices = np.load('data/half_moons/od_idxs.npy').tolist()
    
    # Keep track of which points are outliers to skip drawing them as 'o'
    drawn_as_outlier = set()

    # # Highlight outliers with a different marker
    # for outlier in outliers_indices:
    #     if influence_values[outlier] > -6000:
    #         plt.scatter(influence_values[outlier], inner_products[outlier], marker='X', color=y_colormap[labels[outlier]], s=500, edgecolor='black')
    #     drawn_as_outlier.add(outlier)  # Mark this point as drawn as an outlier
    
    # Scatter plot for each point, skip the ones already drawn as outliers
    # for i, (infl, inner_prod) in enumerate(zip(influence_values, inner_products)):
    #     if i not in drawn_as_outlier:  # Only draw if not already drawn as an outlier
    #         plt.scatter(infl, inner_prod, marker='o', color=y_colormap[labels[i]], s=200)

    for i, (infl, inner_prod) in enumerate(zip(influence_values, inner_products)):
        sorted_infl = np.sort(influence_values)
        if infl < sorted_infl[10]:
            #plt.scatter(infl, inner_prod, marker='X', color=y_colormap[labels[i]], s=500, edgecolor='black')
            drawn_as_outlier.add(i)  # Mark this point as drawn as an outlier
        else:
            ax.scatter(infl, inner_prod, marker='o', color=y_colormap[labels[i]], s=200, alpha=alpha1)
    
    for i in drawn_as_outlier:
        ax.scatter(influence_values[i], inner_products[i], marker='X', color=y_colormap[labels[i]], s=500, edgecolor='black', alpha=alpha1)
    
    # Set labels and title
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Influence Value', fontsize=front_size)
    ax.set_ylabel('Inner Product', fontsize=front_size, rotation=270, labelpad=25)
    
    

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小

    #ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=n_bins))
    ax.xaxis.set_major_locator(ticker.FixedLocator([-16,-8,0, 8,16]))
    #ax.yaxis.set_major_locator(ticker.FixedLocator([-10, 5]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-16,-8,0, 8,16]))
    ax.grid(True, linestyle='--')  # Enable grid
    #plt.axis('equal')
    # Adjust layout and show the plot
    plt.tight_layout()

def plot_training_half_moon (ax):
    # Load the dataset
    df = pd.read_csv('data/half_moons/train.csv', header=None)
    f1, f2, y = df[0].to_list(), df[1].to_list(), df[3].to_list()
    majority_minority = df[2] # 0: majority, 1: minority
    #print(majority_minority)
    #mlp = np.load('explainer/data/binaries/mlp.pth')

    
    # Define the color map
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}

    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1, 1.5)
    # 设置 y 轴的刻度仅显示整数
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 创建网格，用于绘制决策边界
    x_min, x_max = df[0].min() -1, df[0].max() + 1
    y_min, y_max = df[1].min() -1, df[1].max() + 1
    #print(x_min, x_max, y_min, y_max)

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    train_mean = np.mean(np.array([f1, f2]), axis=1)
    train_std = np.std(np.array([f1, f2]), axis=1)
    grid_standardized = standardize_data(grid, mean=train_mean, std=train_std)

    grid_tensor = torch.FloatTensor(grid_standardized)

    
    # 预测并绘制决策边界
    with torch.no_grad():
        Z = mlp(grid_tensor).view(xx.shape)
    Z = Z > 0.5  # 假设使用0.5作为分类阈值
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap='coolwarm', alpha=alpha2)
    

    for i, (a, b) in enumerate(zip(f1, f2)):
        if majority_minority[i] == 'Female':  # majority实框
            ax.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200, edgecolor='black', alpha=alpha1)
        else:  # minority虚框
            ax.scatter(a, b, marker='o', color=y_colormap[y[i]], s=200, edgecolors='black', alpha=alpha1)  # 先画散点
            # circle = plt.Circle((a, b), 0.03, color='black', fill=False, linestyle='--', linewidth=2)  # 添加虚线圆形边框
            # plt.gca().add_artist(circle)

            # plt.plot(a, b, marker='o', linestyle='None', markersize=10,
            #      markeredgecolor='black', markeredgewidth=1, 
            #      color=y_colormap[y[i]], markevery=[1], 
            #      dash_capstyle='butt', solid_capstyle='butt', 
            #      dash_joinstyle='bevel', path_effects=[PathEffects.withStroke(linewidth=1, foreground="black", linestyle="--")])
    
    # Highlight outliers with a different marker
    indices = np.load('data/half_moons/od_idxs.npy').tolist()
    for i, (a, b) in enumerate(zip(f1, f2)):
        if i in indices:
            if majority_minority[i] == 'Female':
                ax.scatter(a, b, marker='X', color=y_colormap[y[i]], s=500, edgecolor='black', alpha=alpha1)
            else:
                ax.scatter(a, b, marker='X', color=y_colormap[y[i]], s=500, edgecolors='black', alpha=alpha1)
            #plt.scatter(a, b, marker='X', color=y_colormap[y[i]], s=500, edgecolor='black')

    



    # Set labels
    ax2 = ax.twinx()
    ax2.yaxis.set_label_position("left")
    ax2.tick_params(axis='y',          # 应用到y轴
               which='both',      # 应用到主要和次要刻度
               left=False,        # 隐藏左侧刻度
               right=False,       # 隐藏右侧刻度
               labelleft=False,   # 隐藏左侧刻度标签
               labelright=False)   # 显示右侧刻度标签
    ax2.set_ylabel('Non-Linear-Utility', fontsize=front_size,labelpad=40, fontweight='bold')
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Feature 1', fontsize=front_size)
    ax.set_ylabel('Feature 2', fontsize=front_size, rotation=270, labelpad=25)

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小

    #plt.axis('equal')
    # Adjust layout to make room for the full feature 2 label
    #ax.xaxis.set_major_locator(ticker.FixedLocator([-4,-2,0,2, 4,6,8]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-0.5,0,0.5,1]))
    plt.tight_layout()


def plot_test_half_moon(ax):
    # Load the test dataset
    df = pd.read_csv('data/half_moons/test.csv', header=None)
    f1, f2, y_te = df[0].to_list(), df[1].to_list(), df[3].to_list()
    majority_minority = df[2]
    
    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1, 1.5)
    # 设置 y 轴的刻度仅显示整数
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # 创建网格，用于绘制决策边界
    x_min, x_max = df[0].min() - 1, df[0].max() + 1
    y_min, y_max = df[1].min() - 1, df[1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    grid = np.c_[xx.ravel(), yy.ravel()]
    test_mean = np.mean(np.array([f1, f2]), axis=1)
    test_std = np.std(np.array([f1, f2]), axis=1)
    grid_standardized = standardize_data(grid, mean=test_mean, std=test_std)

    grid_tensor = torch.FloatTensor(grid_standardized)
    # 预测并绘制决策边界
    with torch.no_grad():
        Z = mlp_new(grid_tensor).view(xx.shape)
    Z = Z > 0.5  # 假设使用0.5作为分类阈值
    ax.contourf(xx, yy, Z, levels=[-1, 0, 1], cmap='coolwarm', alpha=alpha2)


    # Define the color map for the test set classes
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}
    
    # Create scatter plot for each point in the test dataset
    for i, (a, b) in enumerate(zip(f1, f2)):
        if majority_minority[i] == 'Female':  # majority实框
            ax.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200, edgecolor='black', alpha=alpha1)
        else:  # minority虚框
            ax.scatter(a, b, marker='o', color=y_colormap[y_te[i]], s=200, edgecolors='black', alpha=alpha1)


    
    # Set labels for the axes
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Feature 1', fontsize=front_size)
    ax.set_ylabel('Feature 2', fontsize=front_size, rotation=270, labelpad=25)

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小
    
    #plt.axis('equal')
    # Adjust layout to make room for the full feature 2 label
    ax.yaxis.set_major_locator(ticker.FixedLocator([-0.5,0,0.5,1]))
    plt.tight_layout()

def plot_influence_inner_product_MLP(ax):
    # Load influence values and inner products
    influence_values = np.load('explainer/data/binaries/util_infl_nn.npy')
    inner_products = np.load('explainer/data/binaries/util_infl_wo_hess_nn.npy')
    
    # Load class labels
    df = pd.read_csv('data/half_moons/train.csv', header=None)
    labels = df[3].to_list()  # Assuming the labels are in the fourth column
    
    # Define color map based on the classes
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}

    # Set the same scale for x and y axis as in the toy plot
    ax.set_xlim(-6000, 6000)
    ax.set_ylim(-6000, 6000)

    # 将网格设置为底层
    ax.set_axisbelow(True)
    
    # Load outliers indices
    outliers_indices = np.load('data/half_moons/od_idxs.npy').tolist()
    
    # Keep track of which points are outliers to skip drawing them as 'o'
    drawn_as_outlier = set()

    # Highlight outliers with a different marker
    for outlier in outliers_indices:
        if influence_values[outlier] > -6000:
            ax.scatter(influence_values[outlier], inner_products[outlier], marker='X', color=y_colormap[labels[outlier]], s=500, edgecolor='black', alpha=alpha1)
        drawn_as_outlier.add(outlier)  # Mark this point as drawn as an outlier
    
    # Scatter plot for each point, skip the ones already drawn as outliers
    for i, (infl, inner_prod) in enumerate(zip(influence_values, inner_products)):
        if i not in drawn_as_outlier:  # Only draw if not already drawn as an outlier
            ax.scatter(infl, inner_prod, marker='o', color=y_colormap[labels[i]], s=200, alpha=alpha1, edgecolor='black')
    
    # Set labels and title
    ax.yaxis.set_label_position("right")
    ax.set_xlabel('Influence Value', fontsize=front_size)
    ax.set_ylabel('Inner Product', fontsize=front_size, rotation=270, labelpad=25)
    

    ax.tick_params(axis='both', which='major', labelsize=size)  # 调整主要刻度的字体大小
    ax.tick_params(axis='both', which='minor', labelsize=size)  # 如有需要，还可以调整次要刻度的字体大小

    # 设置x轴和y轴使用科学计数法
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    # 可选：强制所有刻度都使用科学计数法（如果matplotlib没有自动选择）
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))

    # 调整科学计数法中 10 的指数部分的字体大小
    ax.xaxis.get_offset_text().set_fontsize(15) # 设置 x 轴偏移文本的字体大小
    ax.yaxis.get_offset_text().set_fontsize(15) # 设置 y 轴偏移文本的字体大小
    #ax.xaxis.set_major_locator(ticker.MaxNLocator( nbins=n_bins))
    ax.xaxis.set_major_locator(ticker.FixedLocator([-4000,-2000,0,2000,4000]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([-4000,-2000,0,2000,4000]))
    ax.grid(True, linestyle='--')  # Enable grid
    #plt.axis('equal')
    # Adjust layout and show the plot
    plt.tight_layout()


# 创建一个4x3的子图网格
#fig, axs = plt.subplots(2, 3, figsize=(15.8, 20))  # figsize可根据需要调整
fig, axs = plt.subplots(2, 3, figsize=(20, 12))  # figsize可根据需要调整

# 将每个绘图函数映射到对应的子图上
plot_functions = [plot_training_toy_Utility, plot_test_toy_utility, plot_influence_inner_product_LR_Utility, 
                  plot_training_half_moon, plot_test_half_moon, plot_influence_inner_product_MLP]
for ax, plot_function in zip(axs.flatten(), plot_functions):
    plot_function(ax)

# 字母编号
letters = [chr(ord('A') + i) for i in range(6)]

for ax, plot_function, letter in zip(axs.flatten(), plot_functions, letters):
    plot_function(ax)
    ax.text(0.15, 0.9, letter, transform=ax.transAxes, fontsize=front_size, fontweight='bold', va='top', ha='right')

# 创建图例
legend_elements = [
    mpatches.Patch(color='tab:red', label='Positive Class'),
    mpatches.Patch(color='tab:blue', label='Negative Class'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=15, label='Normal Point'),
    plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='gray', markersize=25, label='Outlier')
]

# 添加图例
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=4,
           fontsize=front_size, title_fontsize='large', handlelength=2, handletextpad=1, labelspacing=1.2)

# # 调整子图和画布布局
# fig.subplots_adjust(bottom=0.1)  # 适当调整顶部空间以适应图例
# 调整子图间距
#plt.tight_layout(w_pad=-0.5, h_pad=-0.3)
fig.subplots_adjust(bottom=0.15, wspace=0.3, hspace=0.3, top=0.9)
plt.savefig('figures/figure1.pdf', format='pdf')
#plt.show()
