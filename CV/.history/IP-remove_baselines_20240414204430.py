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
from typing import Optional, Sized, Iterator, Tuple

from tqdm import tqdm
from torch import autograd
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.arguments import FactorArguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.arguments import ScoreArguments
#from pyod.models.iforest import IForest
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
from sklearn.random_projection import SparseRandomProjection
from datainf_influence_functions import IFEngine

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
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--acc', type = float, default = 91.62) #0.05

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

def find_idxs(scores, contamination=0.05):
    del_idxs = np.argpartition(scores, int(contamination*len(scores)))
    del_idxs = del_idxs[:int(contamination*len(scores))]
    return del_idxs



class ClassificationTask(Task):
    def compute_train_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, labels, _ = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels.detach(), reduction="sum")

    def compute_measurement(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
        inputs, labels, _ = batch
        logits = model(inputs)

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()






#####################################main code ################################################
# Seed
def set_random_seeds(seed):
    """
    设置随机种子以保证实验的可重复性。
    参数:
    - seed (int): 要设置的随机种子。
    """
    # 设置Python的随机种子
    random.seed(seed)

    # 设置NumPy的随机种子
    np.random.seed(seed)

    # 设置PyTorch的随机种子
    torch.manual_seed(seed)

    # 为所有GPU设置相同的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 确保PyTorch的行为具有确定性，这可能会降低一些运算速度
    torch.backends.cudnn.deterministic = True

    # 禁用benchmarking以避免因随机性引发的非确定性
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parser.parse_args()
    set_random_seeds(args.seed)  # 使用任何您选择的整数作为种子
    for key, value in vars(args).items():
        print(f'{key}: {value}')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
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

    
    #for _ in range(args.repeat):
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
                                    batch_size = 32,
                                    num_workers=args.num_workers,
                                    shuffle=False, #True originally
                                    drop_last = False)
    temp_dl_val = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size = 32,
                                    num_workers=args.num_workers,
                                    shuffle=False,
                                    drop_last = False)
    # 使用 f-string 构造文件名
    dataset = args.dataset  # 例如: 'cifar10'
    noise_type = args.noise_type
    suffix = 1
    best_acc = args.acc#

    filename = f"ckpt/model-{dataset}-{noise_type}-{suffix}-{best_acc}.pth"

    # load the best model
    model = ResNet34_vmap(num_classes).to(device)
    state_dict = torch.load(filename, map_location=device)
    model.load_state_dict(state_dict)
    # 加载旧的状态字典，忽略不匹配的层
    old_state_dict = torch.load(filename)
    new_state_dict = model.state_dict()
    # 将旧状态字典中的参数复制到新模型中，只复制匹配的参数
    for name, param in old_state_dict.items():
        if name in new_state_dict:
            new_state_dict[name].copy_(param)
    print("load the best model with acc:", best_acc)





    # ============================= baseline methods for del_idxs =============================================
    methods, del_lists, time_costs = [], [], []



    grad_layers = [model.linear.weight] # Last layer only. For all layers, use model.parameters()
    total_params = sum(p.numel() for p in grad_layers)

    # 计算val grads
    grads_val = []
    for images, labels, _ in tqdm(temp_dl_val):
        set_random_seeds(0)
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        cur_grads = torch.cat([param.grad.flatten() for param in grad_layers])
        grads_val.append(cur_grads.cpu().detach().numpy())
    grads_val = np.stack(grads_val)
    print(grads_val.shape)

    # 计算avg val grad
    average_grads_val = np.mean(grads_val, axis=0)
    print("accumulated_grads_val.shape", average_grads_val.shape)

    # 计算train grads
    '''
    grads = []
    for images, labels, _ in tqdm(temp_dl_train):
        images = images.to(device)
        labels = labels.to(device)
        model.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        cur_grads = torch.cat([param.grad.flatten() for param in grad_layers])
        #grads.append(cur_grads.cpu().detach().numpy())
        grads.append(cur_grads)
    #grads = np.stack(grads)
    grads = torch.stack(grads).cpu().detach().numpy()
    print(grads.shape)
    '''

    from torch.func import functional_call, vmap, grad

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    
    # 定义一个函数计算梯度和内积
    def compute_loss(params, buffers, image, label):
        # 确保 image 是四维的
        if image.dim() == 3:
            image = image.unsqueeze(0)  # 添加批次维度
        logits = functional_call(model, (params, buffers), (image,))
        loss = F.cross_entropy(logits, label.unsqueeze(0))
        return loss
    
    

    # 迭代数据加载器
    grads = []
    for images, labels,_ in tqdm(temp_dl_train):
        set_random_seeds(0)
        images = images.to(device)
        #print("images shape", images.shape)
        labels = labels.to(device)
        model.zero_grad()
        ft_compute_grad = grad(compute_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad,randomness="same", in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, images, labels)
        # 首先获取批次大小（batch size），假设任一参数的第一维都是批次大小
        batch_size_temp = next(iter(ft_per_sample_grads.values())).shape[0]
        # 为每个样本初始化一个梯度列表
        grads_per_sample = [torch.cat([g[i].flatten() for g in ft_per_sample_grads.values()]) for i in range(batch_size_temp)]
        # 将列表转换为一个Tensor
        total_grads_per_sample = torch.stack(grads_per_sample)# [batch_size, total_number_of_grads]

        # # 计算内积
        # inner_product = torch.matmul(total_grads_per_sample, average_grads_val)
        #print("inner_product shape", inner_product.shape)
        grads.append(total_grads_per_sample)
    grads = torch.cat(grads)
    print("grads shape", grads.shape)
    

    # ============================= Inner Product ============================= 
    start_time = time.time()
    IP = np.matmul(grads, average_grads_val[:, np.newaxis]).reshape(-1)
    cur_time_cost = time.time() - start_time
    del_idxs = find_idxs(IP, contamination=args.contamination)

    methods.append('Ours')
    del_lists.append(del_idxs)
    time_costs.append(cur_time_cost)
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))


    # ============================= GradientTracing, DataInf, LiSSA =============================
    grads_dict_val = {cur_i: {'main': None} for cur_i in range(len(grads_val))}
    for cur_i in range(len(grads_val)):
        grads_dict_val[cur_i]['main'] = torch.from_numpy(grads_val[cur_i].reshape(-1, 512))

    grads_dict = {cur_i: {'main': None} for cur_i in range(len(grads))}
    for cur_i in range(len(grads)):
        grads_dict[cur_i]['main'] = torch.from_numpy(grads[cur_i].reshape(-1, 512))

    inf_eng = IFEngine()
    inf_eng.preprocess_gradients(grads_dict, grads_dict_val)
    inf_eng.compute_hvps(compute_accurate=False)
    inf_eng.compute_IF()

    cur_time_cost = inf_eng.time_dict['identity']
    del_idxs = find_idxs(inf_eng.IF_dict['identity'], contamination=args.contamination)
    methods.append('GradientTracing')
    del_lists.append(del_idxs)
    time_costs.append(cur_time_cost)

    new_acc_ = []
    for _ in range(args.repeat):
        # remove samples from original dataset
        new_train_dl = sample_remove_dataloader(train_dataset, del_idxs, bs=batch_size)

        # retrain model on new dataset
        print('building model...')
        #model = ResNet18(num_classes)
        model = ResNet34(num_classes)
        print('building model done')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
        model.to(device)

        #noise_prior_cur = noise_prior
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
        #print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
        print("\nOriginal Accuracy: {} || New Accuracy of GradientTracing: {}\n".format(best_acc, new_acc))
        new_acc_.append(new_acc)

    new_acc_ = np.array(new_acc_)
    #report the acc and std
    print("\nOriginal Accuracy Avg: {} || New Accuracy Avg of GradientTracing: {}\n".format(best_acc, np.mean(new_acc_)))
    print("\nOriginal Accuracy Std: {} || New Accuracy Std of GradientTracing: {}\n".format(0, np.std(new_acc_)))
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))



    cur_time_cost = inf_eng.time_dict['proposed']
    del_idxs = find_idxs(inf_eng.IF_dict['proposed'], contamination=args.contamination)
    methods.append('DataInf')
    del_lists.append(del_idxs)
    time_costs.append(cur_time_cost)

    new_acc_ = []
    for _ in range(args.repeat):
        # remove samples from original dataset
        new_train_dl = sample_remove_dataloader(train_dataset, del_idxs, bs=batch_size)

        # retrain model on new dataset
        print('building model...')
        #model = ResNet18(num_classes)
        model = ResNet34(num_classes)
        print('building model done')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
        model.to(device)

        #noise_prior_cur = noise_prior
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
        #print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
        print("\nOriginal Accuracy: {} || New Accuracy of DataInf: {}\n".format(best_acc, new_acc))
        new_acc_.append(new_acc)
    new_acc_ = np.array(new_acc_)
    #report the acc and std
    print("\nOriginal Accuracy Avg: {} || New Accuracy Avg of DataInf: {}\n".format(best_acc, np.mean(new_acc_)))
    print("\nOriginal Accuracy Std: {} || New Accuracy Std of DataInf: {}\n".format(0, np.std(new_acc_)))
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))



    cur_time_cost = inf_eng.time_dict['LiSSA']
    del_idxs = find_idxs(inf_eng.IF_dict['LiSSA'], contamination=args.contamination)
    methods.append('LiSSA')
    del_lists.append(del_idxs)
    time_costs.append(cur_time_cost)
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))
    new_acc_ = []
    for _ in range(args.repeat):
        # remove samples from original dataset
        new_train_dl = sample_remove_dataloader(train_dataset, del_idxs, bs=batch_size)

        # retrain model on new dataset
        print('building model...')
        #model = ResNet18(num_classes)
        model = ResNet34(num_classes)
        print('building model done')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
        model.to(device)

        #noise_prior_cur = noise_prior
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
        #print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
        print("\nOriginal Accuracy: {} || New Accuracy of LiSSA: {}\n".format(best_acc, new_acc))
        new_acc_.append(new_acc)
    new_acc_ = np.array(new_acc_)
    #report the acc and std
    print("\nOriginal Accuracy Avg: {} || New Accuracy Avg of LiSSA: {}\n".format(best_acc, np.mean(new_acc_)))
    print("\nOriginal Accuracy Std: {} || New Accuracy Std of LiSSA: {}\n".format(0, np.std(new_acc_)))
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))


    # ============================= Self LiSSA =============================
    self_influences_lissa = []
    cur_time_cost = 0.0
    for i in range(len(grads)):
        temp_gr_d = {0: {'main': None}}
        temp_gr_d[0]['main'] = grads_dict[i]['main']

        influence_engine = IFEngine()
        influence_engine.preprocess_gradients(temp_gr_d, temp_gr_d) #Note no validation set gradients will be used
        influence_engine.compute_hvps(compute_accurate=False)
        influence_engine.compute_IF()

        self_influences_lissa.append(influence_engine.IF_dict['LiSSA'][0])
        cur_time_cost += influence_engine.time_dict['LiSSA']

    self_influences_lissa = np.array(self_influences_lissa)
    self_influences_lissa = -1*self_influences_lissa
    del_idxs = find_idxs(self_influences_lissa, contamination=args.contamination)

    methods.append('SelfLiSSA')
    del_lists.append(del_idxs)
    time_costs.append(cur_time_cost)
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))
    new_acc_ = []
    for _ in range(args.repeat):
        # remove samples from original dataset
        new_train_dl = sample_remove_dataloader(train_dataset, del_idxs, bs=batch_size)

        # retrain model on new dataset
        print('building model...')
        #model = ResNet18(num_classes)
        model = ResNet34(num_classes)
        print('building model done')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
        model.to(device)

        #noise_prior_cur = noise_prior
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
        #print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
        print("\nOriginal Accuracy: {} || New Accuracy of Self Self-LiSSA: {}\n".format(best_acc, new_acc))
        new_acc_.append(new_acc)

    new_acc_ = np.array(new_acc_)
    #report the acc and std
    print("\nOriginal Accuracy Avg: {} || New Accuracy Avg of Self-LiSSA: {}\n".format(best_acc, np.mean(new_acc_)))
    print("\nOriginal Accuracy Std: {} || New Accuracy Std of Self-LiSSA: {}\n".format(0, np.std(new_acc_)))
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))



    # ============================= Self GradientTracing =============================
    start_time = time.time()
    norm2 = np.sum(grads**2, axis=1)
    idx_norm2 = np.argsort(norm2)[::-1]
    del_idxs = list(idx_norm2[:int(args.contamination*len(norm2))])
    cur_time_cost = time.time() - start_time

    methods.append('SelfGradientTracing')
    del_lists.append(del_idxs)
    time_costs.append(cur_time_cost)
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))
    new_acc_ = []
    for _ in range(args.repeat):
        # remove samples from original dataset
        new_train_dl = sample_remove_dataloader(train_dataset, del_idxs, bs=batch_size)

        # retrain model on new dataset
        print('building model...')
        #model = ResNet18(num_classes)
        model = ResNet34(num_classes)
        print('building model done')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
        model.to(device)

        #noise_prior_cur = noise_prior
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
        #print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
        print("\nOriginal Accuracy: {} || New Accuracy of SelfGradientTracing: {}\n".format(best_acc, new_acc))
        new_acc_.append(new_acc)
    new_acc_ = np.array(new_acc_)
    #report the acc and std
    print("\nOriginal Accuracy Avg: {} || New Accuracy Avg of SelfGradientTracing: {}\n".format(best_acc, np.mean(new_acc_)))
    print("\nOriginal Accuracy Std: {} || New Accuracy Std of SelfGradientTracing: {}\n".format(0, np.std(new_acc_)))
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))




    # ============================= Outlier Detection ============================= 
    start_time = time.time()
    clf = IsolationForest(contamination=args.contamination)
    clf.fit(grads)
    scores = clf.predict(grads)
    cur_time_cost = time.time() - start_time

    del_idxs = []
    for i,score in enumerate(scores):
        if score == -1:
            del_idxs.append(i)

    methods.append('Outlier')
    del_lists.append(np.array(del_idxs))
    time_costs.append(cur_time_cost)
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))
    new_acc_ = []
    for _ in range(args.repeat):
        # remove samples from original dataset
        new_train_dl = sample_remove_dataloader(train_dataset, del_idxs, bs=batch_size)

        # retrain model on new dataset
        print('building model...')
        #model = ResNet18(num_classes)
        model = ResNet34(num_classes)
        print('building model done')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
        model.to(device)

        #noise_prior_cur = noise_prior
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
        #print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
        print("\nOriginal Accuracy: {} || New Accuracy of Detection: {}\n".format(best_acc, new_acc))
        new_acc_.append(new_acc)
    new_acc_ = np.array(new_acc_)
    #report the acc and std
    print("\nOriginal Accuracy Avg: {} || New Accuracy Avg of Outlier Detection: {}\n".format(best_acc, np.mean(new_acc_)))
    print("\nOriginal Accuracy Std: {} || New Accuracy Std of Outlier Detection: {}\n".format(0, np.std(new_acc_)))
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))


    # ============================= EKFAC =============================            
    # Prepare the model for influence computation.
    task = ClassificationTask()
    model = prepare_model(model=model, task=task)
    analyzer = Analyzer(analysis_name=args.dataset, model=model, task=task)

    dataloader_kwargs = DataLoaderKwargs(num_workers=4)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    start_time = time.time()
    factor_args = FactorArguments(strategy="ekfac")
    analyzer.fit_all_factors(
        factors_name="ekfac",
        dataset=train_dataset,
        per_device_batch_size=batch_size,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )
    score_args = ScoreArguments(
        damping=None,
        immediate_gradient_removal=False,
        data_partition_size=1,
        module_partition_size=1,
        per_module_score=False,
        query_gradient_rank=None,
        query_gradient_svd_dtype=torch.float64,
        cached_activation_cpu_offload=False,
        score_dtype=torch.float32,
        per_sample_gradient_dtype=torch.float32,
        precondition_dtype=torch.float32,
    )
    # Computing self influence scores.
    analyzer.compute_self_scores(scores_name="self", factors_name="ekfac", score_args=score_args, train_dataset=train_dataset)
    cur_time_cost = time.time() - start_time
    # Loading self influence scores.
    scores = analyzer.load_self_scores(scores_name="self")['all_modules']
    del_idxs = find_idxs(-1*scores, contamination=args.contamination)

    methods.append('EKFAC')
    del_lists.append(del_idxs)
    time_costs.append(cur_time_cost)
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))
    new_acc_ = []
    for _ in range(args.repeat):
        # remove samples from original dataset
        new_train_dl = sample_remove_dataloader(train_dataset, del_idxs, bs=batch_size)

        # retrain model on new dataset
        print('building model...')
        #model = ResNet18(num_classes)
        model = ResNet34(num_classes)
        print('building model done')
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
        model.to(device)

        #noise_prior_cur = noise_prior
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
        #print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
        print("\nOriginal Accuracy: {} || New Accuracy of EKFAC: {}\n".format(best_acc, new_acc))
        new_acc_.append(new_acc)
    new_acc_ = np.array(new_acc_)
    #report the acc and std
    print("\nOriginal Accuracy Avg: {} || New Accuracy Avg of EKFAC: {}\n".format(best_acc, np.mean(new_acc_)))
    print("\nOriginal Accuracy Std: {} || New Accuracy Std of EKFAC: {}\n".format(0, np.std(new_acc_)))
    print('Method:', methods[-1], 'Method Time Cost:', time_costs[-1], 'Del Length:', len(del_lists[-1]))


    print('Total Time Cost ===================', time.time() - beginning)





        #     # do projection and plot (and save) gradient space with outliers
        #     #start_time = time.time()
        #     #grads_embed = SparseRandomProjection(n_components=2).fit_transform(grads) #grads_embed = TSNE(n_components=2).fit_transform(grads)
        #     #end_time = time.time()
        #     #np.save('without-hess/data/grads_resnet.npy', grads)
        #     #plot_grad(grads_embed, del_idxs, 'without-hess/figs/grads_resnet.png')
        #     #print("\nProjection step completed in {} seconds.\n".format(end_time-start_time))

        #     # remove samples from original dataset
        #     new_train_dl = sample_remove_dataloader(train_dataset, del_idxs, bs=batch_size)

        #     # retrain model on new dataset
        #     print('building model...')
        #     #model = ResNet18(num_classes)
        #     model = ResNet34(num_classes)
        #     print('building model done')
        #     optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.0005, momentum=0.9)
        #     model.to(device)

        #     noise_prior_cur = noise_prior
        #     new_acc = []
        #     for epoch in range(args.n_epoch):
        #         print(f'epoch {epoch}')
        #         adjust_learning_rate(optimizer, epoch, alpha_plan)
        #         model.train()
        #         train_acc = train(epoch, new_train_dl, model, optimizer)
        #         test_acc = evaluate(test_loader=test_loader, model=model)
        #         print('train acc on train images is ', train_acc)
        #         print('test acc on test images is ', test_acc)
        #         new_acc.append(test_acc)

        #     # new model on trimmed data has been trained
        #     new_acc = np.max(new_acc)

        #     ending = time.time()
        #     #print("\nEntire experiment concluded in {} seconds.\n".format(ending-beginning))
        #     print("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(best_acc, new_acc))
        #     new_acc_.append(new_acc)

        # org_acc_ = np.array(org_acc_)
        # # print("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
        # # print("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
        # # if args.noise_type == 'aggre':
        # #     with open('output_aggre.txt', 'a') as f:
        # #         for key, value in vars(args).items():
        # #             f.write(f'{key}: {value}')
        # #         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
        # #         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
        # #         f.write('\n')
        # # elif args.noise_type == 'worst':
        # #     with open('output_worse.txt', 'a') as f:
        # #         for key, value in vars(args).items():
        # #             f.write(f'{key}: {value}')
        # #         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
        # #         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
        # #         f.write('\n')
        # # elif args.noise_type == 'rand1':
        # #     with open('output_rand1.txt', 'a') as f:
        # #         for key, value in vars(args).items():
        # #             f.write(f'{key}: {value}')
        # #         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
        # #         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
        # #         f.write('\n')
        # # elif args.noise_type == 'rand2':
        # #     with open('output_rand2.txt', 'a') as f:
        # #         for key, value in vars(args).items():
        # #             f.write(f'{key}: {value}')
        # #         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
        # #         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
        # #         f.write('\n')
        # # elif args.noise_type == 'rand3':
        # #     with open('output_rand3.txt', 'a') as f:
        # #         for key, value in vars(args).items():
        # #             f.write(f'{key}: {value}')
        # #         f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
        # #         f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
        # #         f.write('\n')
        # # elif
        # # with open('output.txt', 'a') as f:
        # #     for key, value in vars(args).items():
        # #         f.write(f'{key}: {value}')
        # #     f.write("\nOriginal Accuracy: {} || New Accuracy: {}\n".format(org_acc_, new_acc_))
        # #     f.write("\nOriginal Accuracy Avg: {} || New Accuracy Avg: {}\n".format(np.mean(org_acc_), np.mean(new_acc_)))
        # #     f.write('\n')
