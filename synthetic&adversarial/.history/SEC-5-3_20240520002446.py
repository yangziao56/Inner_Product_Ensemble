import os
import time
import argparse
import numpy as np
from typing import Sequence

from dataset import fetch_data, DataTemplate
from dataset2 import fetch_data2, DataTemplate2
from eval import Evaluator
from model import LogisticRegression, NNLastLayerIF, MLPClassifier
from fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp
from utils import fix_seed, save2csv

import json

from robust_fn import grad_robust, calc_robust_acc
from robust_fn_nn import grad_robust_nn, calc_robust_acc_nn

import pickle
import random

import copy

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import seaborn as sns
import pandas as pd
from datainf_influence_functions import IFEngine
import torch
import torch.nn as nn
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.arguments import FactorArguments
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.arguments import ScoreArguments

def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--metric', type=str, default="eop", help="eop or dp")
    parser.add_argument('--seed', type=float, help="random seed")
    parser.add_argument('--save_model', type=str, default="n", help="y/n")
    parser.add_argument('--type', type=str, default="util", help="util/fair/robust")
    parser.add_argument('--strategy', type=str, default="dec", help="inc/dec/random")
    parser.add_argument('--points_to_delete', type=int, default=500, help="points to delete")
    parser.add_argument('--random_seed', type=int, default=42, help="seed for random strategy")
    parser.add_argument('--only_pre', type=str, default="n", help="y/n")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn")
    parser.add_argument('--task', type=str, default="remove", help="remove/reweight/relabel")

    args = parser.parse_args()

    return args




def get_full_dataset(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train.csv'
        f.seek(0)        
        json.dump(json_data, f, indent=4)
        f.truncate()

    data: DataTemplate = fetch_data(args.dataset)
    return data


def train_model(args, data):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)

    """ vanilla training """

    model.fit(data.x_train, data.y_train)

    return model


def compute_influence(args, data, model):
    if args.metric == "eop":
        ori_fair_loss_val = loss_ferm(model.log_loss, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        pred_val, _ = model.pred(data.x_val)
        ori_fair_loss_val = loss_dp(data.x_val, data.s_val, pred_val)
    else:
        raise ValueError
    ori_util_loss_val = model.log_loss(data.x_val, data.y_val)

    """ compute the influence and save data """

    pred_train, _ = model.pred(data.x_train)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)
    if args.metric == "eop":
        fair_loss_total_grad = grad_ferm(model.grad, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        fair_loss_total_grad = grad_dp(model.grad_pred, data.x_val, data.s_val)
    else:
        raise ValueError

    if args.model_type != 'nn':
        robust_loss_total_grad = grad_robust(model, data.x_val, data.y_val)
    else:
        robust_loss_total_grad = grad_robust_nn(model, data.x_val, data.y_val)

    hess = model.hess(data.x_train)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)
    
    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)
    
    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)


    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    return util_pred_infl

# IP
def compute_IP(args, data, model):

    """ compute the influence and save data """

    pred_train, _ = model.pred(data.x_train)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)
    if args.metric == "eop":
        fair_loss_total_grad = grad_ferm(model.grad, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        fair_loss_total_grad = grad_dp(model.grad_pred, data.x_val, data.s_val)
    else:
        raise ValueError
    
    util_pred_infl = train_indiv_grad.dot(util_loss_total_grad)
    
    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)


    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    return util_pred_infl

# IP ensemble
def compute_IP_ensemble(args, data, model):
    """ compute the influence and save data """

    pred_train, _ = model.pred(data.x_train)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)

    util_pred_infl = train_indiv_grad.dot(util_loss_total_grad)

    # ensemble
    for i in range(5):
        # data2 = get_full_dataset(args)
        # model2 = train_model(args, data2)
        model2 = copy.deepcopy(model)
        #将model的参数复制到model2中
        # model2.model.coef_ = model.model.coef_
        # model2.model.intercept_ = model.model.intercept_
        #往model2的参数中添加噪声
        #print("model.model.coef_ before", model.model.coef_)
        model2.model.coef_ += np.random.normal(0, 0.01, model2.model.coef_.shape)
        model2.model.intercept_ += np.random.normal(0, 0.01, model2.model.intercept_.shape)
        #print("model.model.coef_ after", model.model.coef_)
        train_total_grad2, train_indiv_grad2 = model2.grad(data.x_train, data.y_train)
        util_loss_total_grad2, acc_loss_indiv_grad2 = model2.grad(data.x_val, data.y_val)
        util_pred_infl += train_indiv_grad2.dot(util_loss_total_grad2)


    
    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)


    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    return util_pred_infl


# Gradient Tracing, Datainf
def compute_Gradient_Tracing(args, data, model):

    """ compute the influence and save data """

    pred_train, _ = model.pred(data.x_train)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)
    grads_val = acc_loss_indiv_grad
    grads_val = util_loss_total_grad[np.newaxis,:]
    grads = train_indiv_grad
    grads_dict_val = {cur_i: {'main': None} for cur_i in range(len(grads_val))}
    for cur_i in range(len(grads_val)):
        #print(grads_val[cur_i].shape)
        grads_dict_val[cur_i]['main'] = torch.from_numpy(grads_val[cur_i].reshape(-1, 1))
    #util_pred_infl = train_indiv_grad.dot(util_loss_total_grad)
    grads_dict = {cur_i: {'main': None} for cur_i in range(len(grads))}
    for cur_i in range(len(grads)):
        grads_dict[cur_i]['main'] = torch.from_numpy(grads[cur_i].reshape(-1, 1))
    
    inf_eng = IFEngine()
    inf_eng.preprocess_gradients(grads_dict, grads_dict_val)
    inf_eng.compute_hvps(compute_accurate=True)
    inf_eng.compute_IF()

    #cur_time_cost = inf_eng.time_dict['identity']
    #del_idxs = find_idxs(inf_eng.IF_dict['identity'], contamination=args.contamination)
    util_pred_infl = inf_eng.IF_dict['identity']
    util_pred_infl_datainf = inf_eng.IF_dict['proposed']
    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)


    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    return -util_pred_infl, -util_pred_infl_datainf


# Self-gradient Tracing, Datainf
def compute_Self(args, data, model):

    """ compute the influence and save data """

    pred_train, _ = model.pred(data.x_train)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)
    grads_val = acc_loss_indiv_grad
    grads_val = util_loss_total_grad[np.newaxis,:]
    grads = train_indiv_grad
    grads_dict_val = {cur_i: {'main': None} for cur_i in range(len(grads))}
    for cur_i in range(len(grads)):
        #print(grads_val[cur_i].shape)
        grads_dict_val[cur_i]['main'] = torch.from_numpy(grads[cur_i].reshape(-1, 1))
    #util_pred_infl = train_indiv_grad.dot(util_loss_total_grad)
    grads_dict = {cur_i: {'main': None} for cur_i in range(len(grads))}
    for cur_i in range(len(grads)):
        grads_dict[cur_i]['main'] = torch.from_numpy(grads[cur_i].reshape(-1, 1))
    
    inf_eng = IFEngine()
    inf_eng.preprocess_gradients(grads_dict, grads_dict_val)
    inf_eng.compute_hvps(compute_accurate=True)
    inf_eng.compute_IF()

    #cur_time_cost = inf_eng.time_dict['identity']
    #del_idxs = find_idxs(inf_eng.IF_dict['identity'], contamination=args.contamination)
    util_pred_infl = inf_eng.IF_dict['identity']
    util_pred_infl_datainf = inf_eng.IF_dict['proposed']
    util_pred_infl_selflissa = inf_eng.IF_dict['LiSSA']
    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)


    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    return -util_pred_infl, -util_pred_infl_datainf, -util_pred_infl_selflissa


#EKFAC
def compute_EKFAC(args, data, model):
    # # 保存逻辑回归模型的权重
    # model.save_model('logreg_weights.npy')
    
    # # 定义一个新的PyTorch逻辑回归模型
    # class PyTorchLogisticRegression(nn.Module):
    #     def __init__(self, input_dim):
    #         super(PyTorchLogisticRegression, self).__init__()
    #         self.linear = nn.Linear(input_dim, 1)
        
    #     def forward(self, x):
    #         return torch.sigmoid(self.linear(x))
    
    # # 加载保存的权重到新的PyTorch模型中
    # input_dim = data.x_train.shape[1]
    # pytorch_model = PyTorchLogisticRegression(input_dim)
    # weights = np.load('logreg_weights.npy')
    # with torch.no_grad():
    #     pytorch_model.linear.weight = nn.Parameter(torch.tensor(weights[:-1]).float().unsqueeze(0))
    #     pytorch_model.linear.bias = nn.Parameter(torch.tensor(weights[-1]).float())

    # # 使用EKFAC进行计算
    # task = Task(name=args.dataset)
    # model_prepared = prepare_model(model=pytorch_model, task=task)
    # analyzer = Analyzer(analysis_name="logreg_ekfac", model=model_prepared, task=task)
    
    # # Fit all EKFAC factors for the given model
    # analyzer.fit_all_factors(factors_name="logreg_factors", dataset=data.x_train)

    # # Compute all pairwise influence scores with the computed factors
    # analyzer.compute_pairwise_scores(
    #     scores_name="logreg_scores",
    #     factors_name="logreg_factors",
    #     query_dataset=data.x_val,
    #     train_dataset=data.x_train,
    #     per_device_query_batch_size=1024,
    # )

    # # Load the scores with dimension `len(data.x_val) x len(data.x_train)`
    # scores = analyzer.load_pairwise_scores(scores_name="logreg_scores")
    
    # return scores
    pred_train, _ = model.pred(data.x_train)

    train_total_grad, train_indiv_grad = model.grad(data.x_train, data.y_train)
    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)

    #util_pred_infl = train_indiv_grad.dot(util_loss_total_grad)

    # ensemble

    # data2 = get_full_dataset(args)
    # model2 = train_model(args, data2)
    model2 = copy.deepcopy(model)
    #将model的参数复制到model2中
    # model2.model.coef_ = model.model.coef_
    # model2.model.intercept_ = model.model.intercept_
    #往model2的参数中添加噪声
    model2.model.coef_ += np.random.normal(0, 0.01, model2.model.coef_.shape)
    model2.model.intercept_ += np.random.normal(0, 0.01, model2.model.intercept_.shape)
    train_total_grad2, train_indiv_grad2 = model2.grad(data.x_train, data.y_train)
    util_loss_total_grad2, acc_loss_indiv_grad2 = model2.grad(data.x_val, data.y_val)
    util_pred_infl = train_indiv_grad2.dot(util_loss_total_grad2)
    return util_pred_infl


def find_trimming_points(args, I2):

    indices_to_delete = I2.argsort()[::-1][-args.points_to_delete:][::-1].tolist()

    print("# Indices to Delete ==> ", len(indices_to_delete))

    return indices_to_delete

#use the influence to reweight the points
def reweight_points(args, data, I2):
    trimmed_data = copy.deepcopy(data)

    X, y, s = [], [], []

    for idx in range(trimmed_data.x_train.shape[0]):
        X.append(trimmed_data.x_train[idx])
        y.append(trimmed_data.y_train[idx])
        s.append(trimmed_data.s_train[idx])

    trimmed_data.x_train = np.array(X)
    trimmed_data.y_train = np.array(y)
    trimmed_data.s_train = np.array(s)
    # 计算影响分数的均值和标准差
    mu_I2 = np.mean(I2)
    sigma_I2 = np.std(I2)
    epsilon = 1e-9  # 小的常数用于防止除零错误

    # 归一化影响分数
    normalized_I2 = (I2 - mu_I2) / np.sqrt(sigma_I2**2 + epsilon)
    # w_train = normalized_I2
    # w_train = I2
    # w_train = np.exp(I2)
    #w_train = np.clip(I2, -1, 1)
    w_train = np.exp(normalized_I2*2)
    
    w_train = w_train[:, None]
    trimmed_data.x_train = trimmed_data.x_train * w_train

    return trimmed_data
    

def delete_points(args, data, points_idx):
    trimmed_data = copy.deepcopy(data)

    X, y, s = [], [], []

    for idx in range(trimmed_data.x_train.shape[0]):
        if idx in points_idx:
            continue
        X.append(trimmed_data.x_train[idx])
        y.append(trimmed_data.y_train[idx])
        s.append(trimmed_data.s_train[idx])

    trimmed_data.x_train = np.array(X)
    trimmed_data.y_train = np.array(y)
    trimmed_data.s_train = np.array(s)
    return trimmed_data

def relabel_points(args, data, points_idx):
    trimmed_data = copy.deepcopy(data)

    X, y, s = [], [], []

    for idx in range(trimmed_data.x_train.shape[0]):
        if idx in points_idx:
            y.append(1 - trimmed_data.y_train[idx])
        else:
            y.append(trimmed_data.y_train[idx])
        X.append(trimmed_data.x_train[idx])
        s.append(trimmed_data.s_train[idx])

    trimmed_data.x_train = np.array(X)
    trimmed_data.y_train = np.array(y)
    trimmed_data.s_train = np.array(s)
    return trimmed_data


def attack_lr(args, data, model):
    clf = model.model

    num2attack = np.random.randint(int(0.05*data.x_test.shape[0]), int(0.25*data.x_test.shape[0]))
    idx2attack = np.random.choice(data.x_test.shape[0], num2attack, replace=False)

    w = clf.coef_[0]
    b = clf.intercept_
    x_adv = []
    for i,x0 in enumerate(data.x_test):
        if i not in idx2attack:
            x_adv.append(x0)
            continue
        perturbation = 1.3 * (np.dot(w, x0) + b) / np.dot(w, w) * w
        x1 = x0 - perturbation
        x_adv.append(x1)
    x_adv = np.array(x_adv)

    return x_adv




if __name__ == "__main__":
    args = parse_args()

    pre, post, defense = [], [], []
    defense_IP = []
    defense_IP_ensemble = []
    defense_Gradient_Tracing = []
    defense_datainf = []
    defense_Self_Gradient_Tracing = []
    defense_Self_datainf = []
    defense_EKFAC = []
    defense_self_LiSSA = []

    n_iters = 10
    args.seed = None

    data = get_full_dataset(args)

    args.points_to_delete = int(0.05*len(data.x_train))
    model = train_model(args, data)

    for _ in range(n_iters):
        print("ITERATION ==> ", _)
        x_adv_test = attack_lr(args, data, model)

        data.x_val, data.y_val, data.s_val = x_adv_test, data.y_test, data.s_test


        influences = compute_influence(args, data, model)
        IP = compute_IP(args, data, model)
        IP_ensemble = compute_IP_ensemble(args, data, model)
        Gradient_Tracing, Datainf = compute_Gradient_Tracing(args, data, model)
        Self_Gradient_Tracing, Self_datainf, Self_Lissa = compute_Self(args, data, model)
        EKFAC = compute_EKFAC(args, data, model)

        if args.task == 'reweight':
            #REWEIGHT
            trimmed_data = reweight_points(args, data, influences)
            trimmed_data_IP = reweight_points(args, data, IP)
            trimmed_data_IP_ensemble = reweight_points(args, data, IP_ensemble)
            trimmed_data_Gradient_Tracing = reweight_points(args, data, Gradient_Tracing)
            trimmed_data_datainf = reweight_points(args, data, Datainf)
            trimmed_data_Self_Gradient_Tracing = reweight_points(args, data, Self_Gradient_Tracing)
            trimmed_data_Self_datainf = reweight_points(args, data, Self_datainf)
            trimmed_data_self_LiSSA = reweight_points(args, data, Self_Lissa)
            trimmed_data_EKFAC = reweight_points(args, data, EKFAC)
        else:
            points_idx = find_trimming_points(args, influences)
            points_idx_IP = find_trimming_points(args, IP)
            points_idx_IP_ensemble = find_trimming_points(args, IP_ensemble)
            points_idx_Gradient_Tracing = find_trimming_points(args, Gradient_Tracing)
            points_idx_datainf = find_trimming_points(args, Datainf)
            points_idx_Self_Gradient_Tracing = find_trimming_points(args, Self_Gradient_Tracing)
            points_idx_Self_datainf = find_trimming_points(args, Self_datainf)
            points_idx_self_LiSSA = find_trimming_points(args, Self_Lissa)
            points_idx_EKFAC = find_trimming_points(args, EKFAC)




            if args.task == 'remove':
                trimmed_data = delete_points(args, data, points_idx)
                trimmed_data_IP = delete_points(args, data, points_idx_IP)
                trimmed_data_IP_ensemble = delete_points(args, data, points_idx_IP_ensemble)
                trimmed_data_Gradient_Tracing = delete_points(args, data, points_idx_Gradient_Tracing)
                trimmed_data_datainf = delete_points(args, data, points_idx_datainf)
                trimmed_data_Self_Gradient_Tracing = delete_points(args, data, points_idx_Self_Gradient_Tracing)
                trimmed_data_Self_datainf = delete_points(args, data, points_idx_Self_datainf)
                trimmed_data_self_LiSSA = delete_points(args, data, points_idx_self_LiSSA)
                trimmed_data_EKFAC = delete_points(args, data, points_idx_EKFAC)

            elif args.task == 'relabel':
                trimmed_data = relabel_points(args, data, points_idx)
                trimmed_data_IP = relabel_points(args, data, points_idx_IP)
                trimmed_data_IP_ensemble = relabel_points(args, data, points_idx_IP_ensemble)
                trimmed_data_Gradient_Tracing = relabel_points(args, data, points_idx_Gradient_Tracing)
                trimmed_data_datainf = relabel_points(args, data, points_idx_datainf)
                trimmed_data_Self_Gradient_Tracing = relabel_points(args, data, points_idx_Self_Gradient_Tracing)
                trimmed_data_Self_datainf = relabel_points(args, data, points_idx_Self_datainf)
                trimmed_data_self_LiSSA = relabel_points(args, data, points_idx_self_LiSSA)
                trimmed_data_EKFAC = relabel_points(args, data, points_idx_EKFAC)
                #relabel
        defense_model = train_model(args, trimmed_data)
        defense_model_IP = train_model(args, trimmed_data_IP)
        defense_model_IP_ensemble = train_model(args, trimmed_data_IP_ensemble)
        defense_model_Gradient_Tracing = train_model(args, trimmed_data_Gradient_Tracing)
        defense_model_datainf = train_model(args, trimmed_data_datainf)
        defense_model_Self_Gradient_Tracing = train_model(args, trimmed_data_Self_Gradient_Tracing)
        defense_model_Self_datainf = train_model(args, trimmed_data_Self_datainf)
        defense_model_self_LiSSA = train_model(args, trimmed_data_self_LiSSA)
        defense_model_EKFAC = train_model(args, trimmed_data_EKFAC)

        #relabel

        #打印model权重
        #print("model.model.coef_:", model.model.coef_)
        #print("model.model.intercept_", model.model.intercept_)
        #打印data.test[1]和data.y_test
        pre.append(accuracy_score(model.pred(data.x_test)[1], data.y_test))
        post.append(accuracy_score(model.pred(x_adv_test)[1], data.y_test))
        defense.append(accuracy_score(defense_model.pred(x_adv_test)[1], data.y_test))
        defense_IP.append(accuracy_score(defense_model_IP.pred(x_adv_test)[1], data.y_test))
        defense_IP_ensemble.append(accuracy_score(defense_model_IP_ensemble.pred(x_adv_test)[1], data.y_test))
        defense_Gradient_Tracing.append(accuracy_score(defense_model_Gradient_Tracing.pred(x_adv_test)[1], data.y_test))
        defense_datainf.append(accuracy_score(defense_model_datainf.pred(x_adv_test)[1], data.y_test))
        defense_Self_Gradient_Tracing.append(accuracy_score(defense_model_Self_Gradient_Tracing.pred(x_adv_test)[1], data.y_test))
        defense_Self_datainf.append(accuracy_score(defense_model_Self_datainf.pred(x_adv_test)[1], data.y_test))
        defense_self_LiSSA.append(accuracy_score(defense_model_self_LiSSA.pred(x_adv_test)[1], data.y_test))
        defense_EKFAC.append(accuracy_score(defense_model_EKFAC.pred(x_adv_test)[1], data.y_test))


    #print(pre, post, defense)
    print("PRE ==> ", np.mean(pre), np.std(pre), pre)
    print("POST ==> ", np.mean(post), np.std(post), post)
    print("DEFENSE ==> ", np.mean(defense), np.std(defense), defense)
    print("DEFENSE_IP ==> ", np.mean(defense_IP), np.std(defense_IP), defense_IP)
    print("DEFENSE_IP_ENSEMBLE ==> ", np.mean(defense_IP_ensemble), np.std(defense_IP_ensemble), defense_IP_ensemble)
    print("DEFENSE_Gradient_Tracing ==> ", np.mean(defense_Gradient_Tracing), np.std(defense_Gradient_Tracing), defense_Gradient_Tracing)
    print("DEFENSE_DATAINF ==> ", np.mean(defense_datainf), np.std(defense_datainf), defense_datainf)
    print("DEFENSE_Self_Gradient_Tracing ==> ", np.mean(defense_Self_Gradient_Tracing), np.std(defense_Self_Gradient_Tracing), defense_Self_Gradient_Tracing)
    print("DEFENSE_Self_DATAINF ==> ", np.mean(defense_Self_datainf), np.std(defense_Self_datainf), defense_Self_datainf)
    print("DEFENSE_self_LiSSA ==> ", np.mean(defense_self_LiSSA), np.std(defense_self_LiSSA), defense_self_LiSSA)
    print("DEFENSE_EKFAC ==> ", np.mean(defense_EKFAC), np.std(defense_EKFAC), defense_EKFAC)
    results = {
    'Method': ['pre', 'post', 'defense', 'defense_IP', 'defense_IP_ensemble', 'defense_Gradient_Tracing', 'defense_datainf', 'defense_Self_Gradient_Tracing', 'defense_Self_datainf', 'defense_EKFAC'],
    'Results': [pre, post, defense, defense_IP, defense_IP_ensemble, defense_Gradient_Tracing, defense_datainf, defense_Self_Gradient_Tracing, defense_Self_datainf, defense_EKFAC]
    }
    # 创建一个DataFrame
    df = pd.DataFrame(results)

    # 将每个方法的结果展开成多列
    df_expanded = df.explode('Results').reset_index(drop=True)

    # 保存到Excel文件,文件明包含 数据集和task
    file_name = 'results/' + args.dataset + '_' + args.task + '_results.xlsx'
    df_expanded.to_excel(file_name, index=False)