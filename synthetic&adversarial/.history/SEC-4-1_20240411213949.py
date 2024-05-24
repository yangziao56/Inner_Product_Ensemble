import os
import time
import argparse
import numpy as np
from typing import Sequence

from dataset import fetch_data, DataTemplate
from dataset_toy import fetch_data as fetch_data_toy
from dataset_toy import DataTemplate as DataTemplate_Toy
#from dataset2 import fetch_data2, DataTemplate2
from eval import Evaluator
from model import LogisticRegression, NNLastLayerIF, MLPClassifier, NN
from fair_fn import grad_ferm, grad_dp, loss_ferm, loss_dp
from utils import fix_seed, save2csv

import json

from robust_fn import grad_robust, calc_robust_acc
from robust_fn_nn import grad_robust_nn, calc_robust_acc_nn

import pickle
import random

import copy

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='Influence Fairness')
    parser.add_argument('--dataset', type=str, default="adult", help="name of the dataset")
    parser.add_argument('--metric', type=str, default="eop", help="eop or dp")
    parser.add_argument('--seed', type=float, default=42, help="random seed")
    parser.add_argument('--save_model', type=str, default="n", help="y/n")
    parser.add_argument('--type', type=str, default="util", help="util/fair/robust")
    parser.add_argument('--strategy', type=str, default="dec", help="inc/dec/random")
    parser.add_argument('--points_to_delete', type=int, default=500, help="points to delete")
    parser.add_argument('--random_seed', type=int, default=42, help="seed for random strategy")
    parser.add_argument('--only_pre', type=str, default="n", help="y/n")
    parser.add_argument('--model_type', type=str, default="logreg", help="logreg/nn")

    args = parser.parse_args()

    return args



def pre_main(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train.csv'
        f.seek(0)        
        json.dump(json_data, f, indent=4)
        f.truncate()


    """ initialization"""
    if args.model_type == 'nn':
        data: DataTemplate = fetch_data(args.dataset)
        print("data.x_train min:", data.x_train.min(), "max:", data.x_train.max())
    else:
        data: DataTemplate_Toy = fetch_data_toy(args.dataset)

    adv_val = np.load('xadv_val.npy')
    adv_test = np.load('xadv_test.npy')
    print(adv_val[0])
    print(adv_test[0])
    print("adv_val shape -> ", adv_val.shape)
    print("adv_test shape -> ", adv_test.shape)
    print("data.x_val shape -> ", data.x_val.shape)
    print("data.y_val", data.y_val)
    print("data.s_val", data.s_val)
    

    model = LogisticRegression(l2_reg=data.l2_reg)
    print("data.l2_reg -> ", data.l2_reg)

    if args.model_type == 'nn':
        #model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)
        model = NN(input_dim=data.dim, seed = args.seed)

    # val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """
    if args.model_type == 'nn':
        model.fit(data.x_train, data.y_train, data.x_val, data.y_val, save_path = 'explainer/data/binaries/mlp.pth')
    else:
        model.fit(data.x_train, data.y_train)
        w1,w2 = model.model.coef_[0]
        b = model.model.intercept_[0]
        print("w1, w2, b -> ", w1, w2, b)
        np.save('explainer/data/binaries/w1.npy', w1)
        np.save('explainer/data/binaries/w2.npy', w2)
        np.save('explainer/data/binaries/b.npy', b)


    # if args.dataset == "toy" and args.save_model == "y":
    #    pickle.dump(model.model, open("toy/model_pre.pkl", "wb"))

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
    #train_total_grad, train_indiv_grad = model.grad_last_layer(data.x_train, data.y_train)
    

    util_loss_total_grad, acc_loss_indiv_grad = model.grad(data.x_val, data.y_val)
    #util_loss_total_grad, acc_loss_indiv_grad = model.grad_last_layer(data.x_val, data.y_val)
    # save
    if args.model_type != 'nn':
        np.save('explainer/data/angle/train_indiv_grad.npy', train_indiv_grad)
        np.save('explainer/data/angle/util_loss_total_grad.npy', util_loss_total_grad)
    else:
        np.save('explainer/data/angle/train_indiv_grad_nn.npy', train_indiv_grad)
        np.save('explainer/data/angle/util_loss_total_grad_nn.npy', util_loss_total_grad)

    
    print("BCEloss", model.log_loss(data.x_val, data.y_val))
    if args.metric == "eop":
        fair_loss_total_grad = grad_ferm(model.grad, data.x_val, data.y_val, data.s_val)
    elif args.metric == "dp":
        fair_loss_total_grad = grad_dp(model.grad_pred, data.x_val, data.s_val)
    else:
        raise ValueError
    # save
    if args.model_type != 'nn':
        np.save('explainer/data/angle/fair_loss_total_grad.npy', fair_loss_total_grad)
        

    if args.model_type != 'nn':
        robust_loss_total_grad = grad_robust(model, adv_val, data.y_val)
    else:
        #robust_loss_total_grad = grad_robust_nn(model, data.x_val, data.y_val)
        pass
    # save
    if args.model_type != 'nn':
        np.save('explainer/data/angle/robust_loss_total_grad.npy', robust_loss_total_grad)

    #hess = model.hess(data.x_train)
    hess = model.hess(data.x_train, data.y_train)
    # 初始化一个hess的shape的单位矩阵
    if args.model_type == 'nn':
        hess_lambda = np.eye(hess.shape[0]) * 1e-3
        hess += hess_lambda
    #hess = model.hess_last_layer(data.x_train, data.y_train)
    # print("Hessian shape -> ", hess.shape)
    # print("Hessian -> ", hess)
    print("util_loss_total_grad shape -> ", util_loss_total_grad.shape)
    # print("Util Loss Total Grad -> ", util_loss_total_grad)
    # print("train_indiv_grad shape -> ", train_indiv_grad.shape)
    # print("Train Indiv Grad -> ", train_indiv_grad)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad, cho = True)
    # print("Util Grad HVP shape -> ", util_grad_hvp.shape)
    # print("Util Grad HVP -> ", util_grad_hvp)
    if args.model_type != 'nn':
        fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)
        robust_grad_hvp = model.get_inv_hvp(hess, robust_loss_total_grad)
    #save
    if args.model_type != 'nn':
        np.save('explainer/data/angle/util_grad_hvp.npy', util_grad_hvp)
        np.save('explainer/data/angle/fair_grad_hvp.npy', fair_grad_hvp)
        np.save('explainer/data/angle/robust_grad_hvp.npy', robust_grad_hvp)
    else:
        np.save('explainer/data/angle/util_grad_hvp_nn.npy', util_grad_hvp)

    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    if args.model_type != 'nn':
        fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)
        robust_pred_infl = train_indiv_grad.dot(robust_grad_hvp)

    if args.model_type != 'nn':
        np.save('explainer/data/binaries/util_infl_lr.npy', util_pred_infl)
        np.save('explainer/data/binaries/fair_infl_lr.npy', fair_pred_infl)
        np.save('explainer/data/binaries/robust_infl_lr.npy', robust_pred_infl)
    else:
        np.save('explainer/data/binaries/util_infl_nn.npy', util_pred_infl)

    # inner prodoct without hessian
    util_pred_infl_wo_hess = train_indiv_grad.dot(util_loss_total_grad)
    if args.model_type != 'nn':
        fair_pred_infl_wo_hess = train_indiv_grad.dot(fair_loss_total_grad)
        robust_pred_infl_wo_hess = train_indiv_grad.dot(robust_loss_total_grad)

    if args.model_type != 'nn':
        np.save('explainer/data/binaries/util_infl_wo_hess_lr.npy', util_pred_infl_wo_hess)
        np.save('explainer/data/binaries/fair_infl_wo_hess_lr.npy', fair_pred_infl_wo_hess)
        np.save('explainer/data/binaries/robust_infl_wo_hess_lr.npy', robust_pred_infl_wo_hess)
    else:
        np.save('explainer/data/binaries/util_infl_wo_hess_nn.npy', util_pred_infl_wo_hess)

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    # val_res = val_evaluator(data.y_val, pred_label_val)
    # test_res = test_evaluator(data.y_test, pred_label_test)

    # s


    '''
    if args.model_type != 'nn':
        val_rob_acc = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'pre')
        test_rob_acc = calc_robust_acc(model, data.x_test, data.y_test, 'test', 'pre')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})
    else:
        val_rob_acc = calc_robust_acc_nn(model, data.x_val, data.y_val, 'val', 'pre')
        test_rob_acc = calc_robust_acc_nn(model, data.x_test, data.y_test, 'test', 'pre')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})
    '''


    np.save('trn.npy', np.append(data.x_train, data.y_train.reshape((-1,1)), 1))
    # return val_res, test_res



def post_main(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train.csv'
        f.seek(0)
        json.dump(json_data, f, indent=4)
        f.truncate()


    """ initialization"""

    #data: DataTemplate2 = fetch_data2(args.dataset)
    data: DataTemplate = fetch_data(args.dataset)
    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)

    #val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """

    model.fit(data.x_train, data.y_train)
    #if args.dataset == "toy" and args.save_model == "y":
    #    pickle.dump(model.model, open("toy/model_post_"+args.type+".pkl", "wb"))


    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    #val_res = val_evaluator(data.y_val, pred_label_val)
    #test_res = test_evaluator(data.y_test, pred_label_test)

    
    if args.model_type != 'nn':
        val_rob_acc = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'post')
        test_rob_acc = calc_robust_acc(model, data.x_test, data.y_test, 'test', 'post')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})
    else:
        val_rob_acc = calc_robust_acc_nn(model, data.x_val, data.y_val, 'val', 'post')
        test_rob_acc = calc_robust_acc_nn(model, data.x_test, data.y_test, 'test', 'post')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})


    return val_res, test_res


def fair_deletion_process(args):
    X_org = np.load('trn.npy')

    #I2 = np.load('explainer/data/binaries/'+args.type+'_infl.npy')
    #I2 = np.load('explainer/data/binaries/'+args.type+'_infl_wo_hess.npy')
    #I2 = np.load('explainer/data/binaries/util_infl_wo_hess_lr.npy')
    #I2 = np.load('explainer/data/binaries/fair_infl_wo_hess_lr.npy')
    I2 = np.load('explainer/data/binaries/robust_infl_wo_hess_lr.npy')

    indices_to_delete = I2.argsort()[::-1][-args.points_to_delete:][::-1].tolist()
    #indices_to_delete = np.where(I2 < 0)[0]
    #indices_to_delete = np.where(I2 < -1500 )[0]

    sort_idx = np.argsort(I2)
    #print("sort_idx", sort_idx)

    def plot_outlier_analysis(infls, od_scores, od_idxs):
        # 创建画布和轴对象
        fig, ax = plt.subplots(figsize=(10, 6))  # 可以调整图的大小

        colormap = {1: 'tab:orange', 0: 'tab:green'}
        for i, infl in enumerate(infls):
            ax.scatter(i, infl, color=colormap[od_scores[i]], marker='o', s=65)

        for i, infl in enumerate(infls):
            if i in od_idxs:
                ax.scatter(i, infl, color=colormap[od_scores[i]], marker='X', s=175, edgecolor='black')

        ax.set_xlabel('Sample Index', fontsize=16)
        ax.set_ylabel('Influence Value', fontsize=16)
        plt.show()  # 显示图表

    def plot_sorted_Influence_and_index(I2, od_idxs):
        # 对I2进行从小到大排序，并保留排序后的索引
        sorted_idxs = np.argsort(I2)

        # 使用排序后的索引获取排序后的I2值
        sorted_I2 = I2[sorted_idxs]

        # 创建一个标记数组，初始化为'o'
        markers = np.array(['o'] * len(I2))

        # 将wrong label点的标记设置为'X'
        markers[sorted_idxs[np.in1d(sorted_idxs, od_idxs)]] = 'X'

        # 绘图
        plt.figure(figsize=(10, 6))

        for i, idx in enumerate(sorted_idxs):
            # 根据标记类型选择颜色
            if markers[idx] == 'X':
                plt.scatter(i, sorted_I2[i], color='green', marker='X')  # 将X标记的颜色改为黄色
            else:
                plt.scatter(i, sorted_I2[i], color='red' if sorted_I2[i] < 0 else 'blue', marker='o')

        plt.xlabel('Sorted Index')
        plt.ylabel('Inner Product Value')
        plt.title('Inner Product Values Sorted with Special Marking for Wrong Labels')
        plt.show()

    # print("# Indices to Delete ==> ", len(indices_to_delete))
    # print("# Indices to Delete ==> ", indices_to_delete)
    od_idxs = np.load('data/half_moons/od_idxs.npy')
    # print("# OD Indices ==> ", od_idxs)

    plot_sorted_Influence_and_index(I2, od_idxs)

    
    import pandas as pd
    X_new = []
    for i,x in enumerate(X_org):
        if i in indices_to_delete:
            continue
        X_new.append(X_org[i])
    X_new = np.array(X_new)

    
    #print(X_new.shape)

    X = X_new
    np.save('2trn.npy', X)

    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train.csv'
        f.seek(0)        
        json.dump(json_data, f, indent=4)
        f.truncate()


    """ initialization"""

    data: DataTemplate = fetch_data(args.dataset)
    #model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        #model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)
        model = NN(input_dim=data.dim, seed = args.seed)
    else:
        model = LogisticRegression(l2_reg=data.l2_reg)

    #val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ re-training """
    if args.model_type == 'nn':
        model.fit(X[:, :-1], X[:, -1], data.x_val, data.y_val)
        model.save_model('explainer/data/binaries/mlp_new.pth')
    else:
        model.fit(X[:, :-1], X[:, -1])
        w1,w2 = model.model.coef_[0]
        b = model.model.intercept_[0]
        print("w1, w2, b -> ", w1, w2, b)
        # np.save('explainer/data/binaries/w1_utility.npy', w1)
        # np.save('explainer/data/binaries/w2_utility.npy', w2)
        # np.save('explainer/data/binaries/b_utility.npy', b)

        # np.save('explainer/data/binaries/w1_fairness.npy', w1)
        # np.save('explainer/data/binaries/w2_fairness.npy', w2)
        # np.save('explainer/data/binaries/b_fairness.npy', b)

        np.save('explainer/data/binaries/w1_robustness.npy', w1)
        np.save('explainer/data/binaries/w2_robustness.npy', w2)
        np.save('explainer/data/binaries/b_robustness.npy', b)


        
    #if args.dataset == "toy" and args.save_model == "y":
    #    pickle.dump(model.model, open("toy/model_post_"+args.type+".pkl", "wb"))


    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    # val_res = val_evaluator(data.y_val, pred_label_val)
    # test_res = test_evaluator(data.y_test, pred_label_test)

    
    if args.model_type != 'nn':
        val_rob_acc = calc_robust_acc(model, data.x_val, data.y_val, 'val', 'post')
        test_rob_acc = calc_robust_acc(model, data.x_test, data.y_test, 'test', 'post')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})
    else:
        val_rob_acc = calc_robust_acc_nn(model, data.x_val, data.y_val, 'val', 'post')
        test_rob_acc = calc_robust_acc_nn(model, data.x_test, data.y_test, 'test', 'post')
        #######################################################
        print("Validation set robustness accuracy -> ", val_rob_acc)
        print("Test set robustness accuracy -> ", test_rob_acc)
        #######################################################
        val_res.update({'robust_acc': val_rob_acc})
        test_res.update({'robust_acc': test_rob_acc})


def plot_with_and_without_hessian():
    util_pred_infl = np.load('explainer/data/binaries/util_infl.npy')
    fair_pred_infl = np.load('explainer/data/binaries/fair_infl.npy')
    robust_pred_infl = np.load('explainer/data/binaries/robust_infl.npy')

    util_pred_infl_wo_hess = np.load('explainer/data/binaries/util_infl_wo_hess.npy')
    fair_pred_infl_wo_hess = np.load('explainer/data/binaries/fair_infl_wo_hess.npy')
    robust_pred_infl_wo_hess = np.load('explainer/data/binaries/robust_infl_wo_hess.npy')

    plt.figure(figsize=(10, 10))
    plt.subplot(3, 1, 1)
    plt.plot(util_pred_infl, label='With Hessian')
    plt.plot(util_pred_infl_wo_hess, label='Without Hessian')
    plt.title('Utility Influence')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(fair_pred_infl, label='With Hessian')
    plt.plot(fair_pred_infl_wo_hess, label='Without Hessian')
    plt.title('Fairness Influence')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(robust_pred_infl, label='With Hessian')
    plt.plot(robust_pred_infl_wo_hess, label='Without Hessian')
    plt.title('Robustness Influence')
    plt.legend()

    plt.show()

def plot_hessian_comparison(args):
    # 加载数据
    if args.model_type == 'nn':
        util_pred_infl = np.load('explainer/data/binaries/util_infl_nn.npy')
        # fair_pred_infl = np.load('explainer/data/binaries/fair_infl.npy')
        # robust_pred_infl = np.load('explainer/data/binaries/robust_infl.npy')

        util_pred_infl_wo_hess = np.load('explainer/data/binaries/util_infl_wo_hess_nn.npy')
        # fair_pred_infl_wo_hess = np.load('explainer/data/binaries/fair_infl_wo_hess.npy')
        # robust_pred_infl_wo_hess = np.load('explainer/data/binaries/robust_infl_wo_hess.npy')
        
        val_res_all = np.load('explainer/data/binaries/val_res_all.npy')
        # import pandas as pd
        # corr_pearson = df['util_pred_infl'].corr(df['util_pred_infl_wo_hess'], method='pearson')
        # print(f"皮尔逊相关系数为: {corr_pearson}")

        from scipy.stats import spearmanr, kendalltau
        # 假设x和y是两个变量的数值列表
        #corr_spearman, _ = spearmanr(util_pred_infl, util_pred_infl_wo_hess)
        corr_kendall, _ = kendalltau(util_pred_infl, util_pred_infl_wo_hess)
        #print(f"Utility Influence的Spearman相关系数为: {corr_spearman}")
        print(f"Utility Influence的Kendall相关系数为: {corr_kendall}")
        corr_kendall, _ = kendalltau(val_res_all, util_pred_infl_wo_hess)
        print(f"LOO和Utility Influence的Kendall相关系数为: {corr_kendall}")
        corr_kendall, _ = kendalltau(val_res_all, util_pred_infl)
        print("LOO和hessian Utility Influence的Kendall相关系数为: ", corr_kendall)



        # 设置图形大小
        plt.figure(figsize=(10, 10))

        # 绘制Utility Influence的比较散点图
        plt.subplot(3, 1, 1)
        plt.scatter(util_pred_infl, util_pred_infl_wo_hess)
        plt.xlabel('With Hessian')
        plt.ylabel('Without Hessian')
        plt.title('Utility Influence Comparison')

        # 绘制Fairness Influence的比较散点图
        plt.subplot(3, 1, 2)
        plt.scatter(val_res_all, util_pred_infl_wo_hess)
        plt.xlabel('loo')
        plt.ylabel('Without Hessian')
        plt.title('Utility Influence Comparison')

        # 绘制Robustness Influence的比较散点图
        plt.subplot(3, 1, 3)
        plt.scatter(util_pred_infl, val_res_all)
        plt.xlabel('With Hessian')
        plt.ylabel('loo')
        plt.title('Utility Influence Comparison')

        # 显示图形
        plt.tight_layout()
        plt.show()
    else:
        util_pred_infl = np.load('explainer/data/binaries/util_infl_lr.npy')
        fair_pred_infl = np.load('explainer/data/binaries/fair_infl_lr.npy')
        robust_pred_infl = np.load('explainer/data/binaries/robust_infl_lr.npy')

        util_pred_infl_wo_hess = np.load('explainer/data/binaries/util_infl_wo_hess_lr.npy')
        fair_pred_infl_wo_hess = np.load('explainer/data/binaries/fair_infl_wo_hess_lr.npy')
        robust_pred_infl_wo_hess = np.load('explainer/data/binaries/robust_infl_wo_hess_lr.npy')

        #plot utility influence vs utility influence without hessian，fair influence vs fair influence without hessian，robust influence vs robust influence without hessian
        plt.figure(figsize=(10, 10))
        plt.subplot(3, 1, 1)
        plt.scatter(util_pred_infl, util_pred_infl_wo_hess)
        plt.xlabel('With Hessian')
        plt.ylabel('Without Hessian')
        plt.title('Utility Influence Comparison')

        plt.subplot(3, 1, 2)
        plt.scatter(fair_pred_infl, fair_pred_infl_wo_hess)
        plt.xlabel('With Hessian')
        plt.ylabel('Without Hessian')
        plt.title('Fairness Influence Comparison')

        plt.subplot(3, 1, 3)
        plt.scatter(robust_pred_infl, robust_pred_infl_wo_hess)
        plt.xlabel('With Hessian')
        plt.ylabel('Without Hessian')
        plt.title('Robustness Influence Comparison')

        plt.tight_layout()
        plt.show()





# plot half moons training set
def plot_first_figures():
    df = pd.read_csv('data/half_moons/train.csv', header=None)
    f1, f2, y = df[0].to_list(), df[1].to_list(), df[3].to_list()
    y_colormap = {'N': 'tab:blue', 'Y': 'tab:red'}
    for i, (a, b) in enumerate(zip(f1, f2)):
        axs[0].scatter(a, b, marker='o', color=y_colormap[y[i]], s=65)
    for i, (a, b) in enumerate(zip(f1, f2)):
        if i in indices:
            axs[0].scatter(a, b, marker='X', color=y_colormap[y[i]], s=175, edgecolor='black')
    axs[0].set_xlabel('Feature 1', fontsize=16)
    axs[0].set_ylabel('Feature 2', fontsize=16)




if __name__ == "__main__":
    args = parse_args()

    #######################
    args.dataset = 'toy'
    #args.dataset = 'half_moons'
    args.save_model = 'y'
    args.metric = 'dp'
    args.seed = 39
    #######################

    print("\nUtility Trimming\n")
    args.type = 'util'

    pre_main(args)
    #pre_val_res, pre_test_res = pre_main(args) #Run pre code

    #plot_with_and_without_hessian()
    plot_hessian_comparison(args)

    args.points_to_delete = 10

    fair_deletion_process(args)

    #plot_first_figures()
    #post_val_res, post_test_res = post_main(args)

    '''
    print("\nFairness Trimming\n")
    args.type = 'fair'

    pre_val_res, pre_test_res = pre_main(args) #Run pre code

    args.points_to_delete = 10

    fair_deletion_process(args)
    post_val_res, post_test_res = post_main(args)


    print("\nRobustness Trimming\n")
    args.type = 'robust'

    pre_val_res, pre_test_res = pre_main(args) #Run pre code

    args.points_to_delete = 10

    fair_deletion_process(args)
    post_val_res, post_test_res = post_main(args)
    '''
