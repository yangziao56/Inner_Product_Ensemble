# -*- coding:utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data.datasets import input_dataset
from models import *
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import random
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from typing import Optional, Sized, Iterator

from tqdm import tqdm

from torch import autograd

#from pyod.models.iforest import IForest
from sklearn.ensemble import IsolationForest

from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection

parser = argparse.ArgumentParser()
parser.add_argument('--contamination', type = float, default = 0.05) #0.05
parser.add_argument('--threshold', type = float, default = 1) #0.05
parser.add_argument('--batch_size', type = int, default = 128) #128
parser.add_argument('--lr', type = float, default = 0.1) #0.1
parser.add_argument('--noise_type', type = str, help='clean, aggre, worst, rand1, rand2, rand3, clean100, noisy100', default='aggre')
parser.add_argument('--noise_path', type = str, help='path of CIFAR-10_human.pt', default=None)
parser.add_argument('--dataset', type = str, help = ' cifar10 or cifar100', default = 'cifar10')
parser.add_argument('--n_epoch', type=int, default=80)#100
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--print_freq', type=int, default=50)
parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')
parser.add_argument('--is_human', action='store_true', default=False)
parser.add_argument('--device', type=str, default='cuda:1')
parser.add_argument('--repeat', type=int, default=5)
parser.add_argument('--ensemble_size', type=int, default=5)
parser.add_argument('--dropout_IP', type = float, default = 0.01) #0.05
parser.add_argument('--last_layer', type = bool, default = False) #0.05
parser.add_argument('--acc', type = float, default = 91.62) #0.05
args = parser.parse_args()
for key, value in vars(args).items():
    print(f'{key}: {value}')
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
# Adjust learning rate and for SGD Optimizer
def adjust_learning_rate(optimizer, epoch,alpha_plan):
    for param_group in optimizer.param_groups:
        param_group['lr']=alpha_plan[epoch]
        

def accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

# Train the Model
def train(epoch, train_loader, model, optimizer):
    train_total=0
    train_correct=0

    for i, (images, labels, indexes) in enumerate(train_loader):
        ind=indexes.cpu().numpy().transpose()
        batch_size = len(ind)
       
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)
       
        # Forward + Backward + Optimize
        logits = model(images)

        prec, _ = accuracy(logits, labels, topk=(1, 5))
        # prec = 0.0
        train_total+=1
        train_correct+=prec
        loss = F.cross_entropy(logits, labels, reduce = True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % args.print_freq == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy: %.4F, Loss: %.4f, LR: %.8f'
                  %(epoch+1, args.n_epoch, i+1, len(train_dataset)//batch_size, prec, loss.data, optimizer.param_groups[0]['lr']))


    train_acc=float(train_correct)/float(train_total)
    return train_acc

# Evaluate the Model
def evaluate(test_loader, model):
    model.eval()    # Change model to 'eval' mode.
    correct = 0
    total = 0
    for images, labels, _ in test_loader:
        images = Variable(images).to(device)
        logits = model(images)
        outputs = F.softmax(logits, dim=1)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred.cpu() == labels).sum()
    acc = 100*float(correct)/float(total)

    return acc


class RemovalSampler(RandomSampler):
    r"""Sample elements randomly. 
    Not everything from RandomSampler is implemented.

    Args:
        data_source (Dataset): dataset to sample from
        forbidden  (Optional[list]): list of forbidden numbers
    """
    data_source: Sized
    forbidden: Optional[list]

    def __init__(self, data_source: Sized, forbidden: Optional[list] = []) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.forbidden = forbidden
        self.refill()

    def remove(self, new_forbidden):
        # Remove numbers from the available indices
        for num in new_forbidden:
            if not (num in self.forbidden):
                self.forbidden.append(num)
        self._remove(new_forbidden)

    def _remove(self, to_remove):
        # Remove numbers just for this epoch
        for num in to_remove:
            if num in self.idx:
                self.idx.remove(num)

        self._num_samples = len(self.idx)

    def refill(self):
        # Refill the indices after iterating through the entire DataLoader
        self.idx = list(range(len(self.data_source)))
        self._remove(self.forbidden)

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples // 32):
            batch = random.sample(self.idx, 32)
            self._remove(batch)
            yield from batch
        yield from random.sample(self.idx, self.num_samples % 32)
        self.refill()


def sample_remove_dataloader(ds, idx_list, bs=128):
    sampler = RemovalSampler(ds, forbidden=idx_list)

    dl = DataLoader(dataset = train_dataset,
                                batch_size = bs,
                                num_workers=args.num_workers,
                                shuffle=False, #True originally
                                drop_last = False,
                                sampler=sampler)
    return dl


def plot_grad(grads, od_idxs, figname):
    plt.clf()

    for i,g in enumerate(grads):
        if i in od_idxs:
            plt.scatter(g[0], g[1], color='tab:purple', marker='X', s=75, edgecolor='black')
            continue

        plt.scatter(g[0], g[1], color='tab:purple', marker='o')

    plt.xlabel('Gradient Value (Projection 1)')
    plt.ylabel('Gradient Value (Projection 2)')
    o_point = Line2D([0], [0], label='Predicted Non-outlier', marker='o', markersize=10, markerfacecolor='tab:purple', markeredgecolor='tab:purple', linestyle='')
    x_point = Line2D([0], [0], label='Predicted Outlier', marker='X', markersize=10, markerfacecolor='tab:purple', markeredgecolor='black', linestyle='')

    plt.legend(bbox_to_anchor=(1.5, 0.5), handles=[o_point, x_point])

    plt.savefig(figname, dpi=300, bbox_inches='tight')


#####################################main code ################################################
# Seed
# 设置Python随机种子
random.seed(args.seed)
# 设置NumPy随机种子
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyper Parameters
batch_size = args.batch_size #128
learning_rate = args.lr
noise_type_map = {'clean':'clean_label', 'worst': 'worse_label', 'aggre': 'aggre_label', 'rand1': 'random_label1', 'rand2': 'random_label2', 'rand3': 'random_label3', 'clean100': 'clean_label', 'noisy100': 'noisy_label'}
args.noise_type = noise_type_map[args.noise_type]
# load dataset
if args.noise_path is None:
    if args.dataset == 'cifar10':
        args.noise_path = './data/CIFAR-10_human.pt'
    elif args.dataset == 'cifar100':
        args.noise_path = './data/CIFAR-100_human.pt'
    else: 
        raise NameError(f'Undefined dataset {args.dataset}')


train_dataset,test_dataset,num_classes,num_training_samples = input_dataset(args.dataset,args.noise_type, args.noise_path, args.is_human)
noise_prior = train_dataset.noise_prior
noise_or_not = train_dataset.noise_or_not
print('train_labels:', len(train_dataset.train_labels), train_dataset.train_labels[:10])

org_acc_ = []
new_acc_ = []
for _ in range(args.repeat):
    beginning = time.time()

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                    batch_size = batch_size, #128
                                    num_workers=args.num_workers,
                                    shuffle=False, #True originally
                                    drop_last = False)


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size = batch_size, # 64
                                    num_workers=args.num_workers,
                                    shuffle=False)

    alpha_plan = [0.1] * 60 + [0.01] * 40

    # original model on full data has been trained

    # find first (and second) order gradients
    temp_dl_train = torch.utils.data.DataLoader(dataset = train_dataset,
                                    batch_size = 1,
                                    num_workers=args.num_workers,
                                    shuffle=False, #True originally
                                    drop_last = False)
    temp_dl_val = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size = 256,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    drop_last = False)
    # 使用 f-string 构造文件名
    dataset = args.dataset  # 例如: 'cifar10'
    noise_type = args.noise_type
    suffix = 1
    #best_acc = 91.62
    best_acc = args.acc

    filename = f"/home/exxact-yza/桌面/Research/Influence_Function/IP_data_sel/cifar-10-100n/ckpt/model-{dataset}-{noise_type}-{suffix}-{best_acc}.pth"
    #初始化IP_sum为长度与training set相同的torch0，用于存储IP的和
    IP_sum = torch.zeros(len(train_dataset)).to(device)
    for __ in range(args.ensemble_size):
        # load the best model
        #model = ResNet34(num_classes).to(device)
        #dropout_rate是一个(0到args.droupout_IP)随机的参数，用于控制dropout的比例
        if __ == 0:
            dropout_rate = 0
        else:
            dropout_rate = np.random.uniform(0, args.dropout_IP)
        model = ResNet34_dropout(num_classes,dropout_rate=dropout_rate).to(device)
        # state_dict = torch.load(filename)
        # model.load_state_dict(state_dict)
        # 加载旧的状态字典，忽略不匹配的层
        old_state_dict = torch.load(filename)
        new_state_dict = model.state_dict()

        # 将旧状态字典中的参数复制到新模型中，只复制匹配的参数
        for name, param in old_state_dict.items():
            if name in new_state_dict:
                new_state_dict[name].copy_(param)

        model.load_state_dict(new_state_dict, strict=False)
        print("load the best model with acc:", best_acc)
        if args.last_layer:
            total_params = model.linear.weight.numel() + model.linear.bias.numel()
        else:
            total_params = sum(p.numel() for p in model.parameters())
        accumulated_grads_val = torch.zeros(total_params).to(device)  # 假设total_params是正确的
        #torch.save(model.state_dict(), 'ckpt/model.pth')
        for images, labels, _ in tqdm(temp_dl_val):
            images = images.to(device)
            labels = labels.to(device)
            model.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
            # 累加本批次的梯度
            if args.last_layer:
                grads = torch.cat([model.linear.weight.grad.flatten(), model.linear.bias.grad.flatten()])
            else:
                grads = torch.cat([param.grad.flatten() for param in model.parameters()])
            accumulated_grads_val += grads

        # 计算平均梯度
        average_grads_val = accumulated_grads_val / len(temp_dl_val.dataset)
        print("accumulated_grads_val.shape", average_grads_val.shape)

        # 计算IP
        IP = []
        for images, labels, _ in tqdm(temp_dl_train):
            # if _ != 0:
            #     #break the loop
            #     break

            images = images.to(device)
            labels = labels.to(device)
            model.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            
            # 累加本批次的梯度
            if args.last_layer:
                grads = torch.cat([model.linear.weight.grad.flatten(), model.linear.bias.grad.flatten()])
            else:
                grads = torch.cat([param.grad.flatten() for param in model.parameters()])
            
            if _ == 0:
                #save the grads
                torch.save(grads, f'IP_save/training_1st_grad{__}-{args.dropout_IP}.pth')


            # print("grads shape", grads.shape)
            inner_product = torch.dot(grads, average_grads_val)
            IP.append(inner_product)
            # print("IP ", inner_product)
            # print("IP shape", inner_product.shape)

        IP = torch.stack(IP)
        # save the IP
        torch.save(IP, f'IP_save/IP{__}-{args.dropout_IP}.pth')

        IP_sum = IP_sum + IP

    #save the IP_sum
    torch.save(IP_sum, f'IP_save/IP_sum{__}-{args.dropout_IP}.pth')

    # print("IP shape", IP_sum.shape)
    IP_sorted, _ = torch.sort(IP_sum)
    
    # 确保保存目录存在
    os.makedirs("IP_save", exist_ok=True)
    filename_tensor = f"IP_save/IP_sum-{dataset}-{suffix}-{best_acc}-{args.ensemble_size}-{args.dropout_IP}.pth"
    torch.save(IP_sum, filename_tensor)
    # print("IP shape", IP_sum.shape)
    IP_sorted, _ = torch.sort(IP_sum)

    # 使用matplotlib进行可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(IP_sorted)), IP_sorted.cpu().numpy(), label='IP Value', s=1)  # 使用更小的点大小
    plt.xlabel('Index')
    plt.ylabel('IP Value')
    plt.title('Visualization of IP Values')
    plt.legend()
    #plt.show()
    plt.savefig(f'IP_save/IP_sum-{dataset}-{suffix}-{best_acc}-{args.ensemble_size}-{args.dropout_IP}.png', dpi=300)


    # add delete index to list, delete samples with loweat args.contamination IP
    if args.threshold == 0:
        threshold = 0
    else:
        threshold = IP_sorted[int(len(IP_sorted) * args.contamination)]
    del_idxs = []

    np.save(f'IP_save/del_idxs-{dataset}-{suffix}-{best_acc}-{args.ensemble_size}-{args.dropout_IP}.npy', del_idxs)
    
    for i, ip in enumerate(IP):
        if ip <= threshold:
            del_idxs.append(i)

    '''
    start_time = time.time()
    #clf = IForest(contamination=args.contamination) #contaimination=0.05 works best so far
    clf = IsolationForest(contamination=args.contamination)
    clf.fit(grads)
    scores = clf.predict(grads)
    end_time = time.time()
    print("\nIForest completed in {} seconds.\n".format(end_time-start_time))

    del_idxs = []
    for i,score in enumerate(scores):
        #if score == 1:
        if score == -1:
            del_idxs.append(i)

    print("\nIForest identified {} outliers.\n".format(len(del_idxs)))
    '''
    # do projection and plot (and save) gradient space with outliers
    #start_time = time.time()
    #grads_embed = SparseRandomProjection(n_components=2).fit_transform(grads) #grads_embed = TSNE(n_components=2).fit_transform(grads)
    #end_time = time.time()
    #np.save('without-hess/data/grads_resnet.npy', grads)
    #plot_grad(grads_embed, del_idxs, 'without-hess/figs/grads_resnet.png')
    #print("\nProjection step completed in {} seconds.\n".format(end_time-start_time))

    # remove samples from original dataset
    new_train_dl = sample_remove_dataloader(train_dataset, del_idxs, bs=batch_size)

    # retrain model on new dataset
    print('building model...')
    #model = ResNet18(num_classes)
    model = ResNet34(num_classes)
    print('building model done')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
    model.to(device)

    noise_prior_cur = noise_prior
    new_acc = []
    for epoch in range(args.n_epoch):
        print(f'epoch {epoch}')
        adjust_learning_rate(optimizer, epoch, alpha_plan)
        model.train()
        train_acc = train(epoch, new_train_dl, model, optimizer)
        test_acc = evaluate(test_loader=test_loader, model=model)
        print('train acc on train images is ', train_acc)
        print('test acc on test images is ', test_acc)
        new_acc.append(test_acc)

    # new model on trimmed data has been trained
    new_acc = np.max(new_acc)

    ending = time.time()
    #print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
    print("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(best_acc, new_acc))
    new_acc_.append(new_acc)

new_acc_ = np.array(new_acc_)
#report the acc and std
print("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(best_acc, np.mean(new_acc_)))
print("\nOriginal Accuracy Std: {} || New Accuracy Std: {}\n".format(0, np.std(new_acc_)))

# print("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
# print("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
# if args.noise_type == 'aggre':
#     with open('output_aggre.txt', 'a') as f:
#         for key, value in vars(args).items():
#             f.write(f'{key}: {value}')
#         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
#         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
#         f.write('\n')
# elif args.noise_type == 'worst':
#     with open('output_worse.txt', 'a') as f:
#         for key, value in vars(args).items():
#             f.write(f'{key}: {value}')
#         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
#         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
#         f.write('\n')
# elif args.noise_type == 'rand1':
#     with open('output_rand1.txt', 'a') as f:
#         for key, value in vars(args).items():
#             f.write(f'{key}: {value}')
#         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
#         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
#         f.write('\n')
# elif args.noise_type == 'rand2':
#     with open('output_rand2.txt', 'a') as f:
#         for key, value in vars(args).items():
#             f.write(f'{key}: {value}')
#         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
#         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
#         f.write('\n')
# elif args.noise_type == 'rand3':
#     with open('output_rand3.txt', 'a') as f:
#         for key, value in vars(args).items():
#             f.write(f'{key}: {value}')
#         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
#         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
#         f.write('\n')
# elif
# with open('output.txt', 'a') as f:
#     for key, value in vars(args).items():
#         f.write(f'{key}: {value}')
#     f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
#     f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
#     f.write('\n')
