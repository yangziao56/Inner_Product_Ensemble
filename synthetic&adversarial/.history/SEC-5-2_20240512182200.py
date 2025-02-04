import os
import time
import argparse
import numpy as np
from typing import Sequence

from dataset import fetch_data, DataTemplate
from dataset2 import fetch_data2, DataTemplate2
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
    parser.add_argument('--model_type', type=str, default="nn", help="logreg/nn")

    args = parser.parse_args()

    return args



def pre_main_clean(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train_clean.csv'
        f.seek(0)        
        json.dump(json_data, f, indent=4)
        f.truncate()


    """ initialization"""

    data: DataTemplate = fetch_data(args.dataset)
    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)
        model = NN(input_dim=data.dim, seed = args.seed)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """

    model.fit(data.x_train, data.y_train)
    if args.dataset == "toy" and args.save_model == "y":
        pickle.dump(model.model, open("toy/model.pkl", "wb"))

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
        pass
        #robust_loss_total_grad = grad_robust_nn(model, data.x_val, data.y_val)

    hess = model.hess(data.x_train)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)
    #robust_grad_hvp = model.get_inv_hvp(hess, robust_loss_total_grad)

    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)
    #robust_pred_infl = train_indiv_grad.dot(robust_grad_hvp)

    np.save('explainer/data/binaries/util_infl.npy', util_pred_infl)
    np.save('explainer/data/binaries/fair_infl.npy', fair_pred_infl)
    #np.save('explainer/data/binaries/robust_infl.npy', robust_pred_infl)

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)


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


    np.save('trn.npy', np.append(data.x_train, data.y_train.reshape((-1,1)), 1))
    return val_res, test_res


def pre_main_poisoned(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train_poisoned.csv'
        f.seek(0)        
        json.dump(json_data, f, indent=4)
        f.truncate()


    """ initialization"""

    data: DataTemplate = fetch_data(args.dataset)
    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """

    model.fit(data.x_train, data.y_train)
    if args.dataset == "toy" and args.save_model == "y":
        pickle.dump(model.model, open("toy/model.pkl", "wb"))

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
        pass
        #robust_loss_total_grad = grad_robust_nn(model, data.x_val, data.y_val)

    hess = model.hess(data.x_train)
    util_grad_hvp = model.get_inv_hvp(hess, util_loss_total_grad)
    fair_grad_hvp = model.get_inv_hvp(hess, fair_loss_total_grad)
    #robust_grad_hvp = model.get_inv_hvp(hess, robust_loss_total_grad)

    util_pred_infl = train_indiv_grad.dot(util_grad_hvp)
    fair_pred_infl = train_indiv_grad.dot(fair_grad_hvp)
    #robust_pred_infl = train_indiv_grad.dot(robust_grad_hvp)

    np.save('explainer/data/binaries/util_infl.npy', util_pred_infl)
    np.save('explainer/data/binaries/fair_infl.npy', fair_pred_infl)
    #np.save('explainer/data/binaries/robust_infl.npy', robust_pred_infl)

    #inner product
    util_pred_infl_wo_hess = train_indiv_grad.dot(util_loss_total_grad)
    fair_pred_infl_wo_hess = train_indiv_grad.dot(fair_loss_total_grad)
    #robust_pred_infl_wo_hess = train_indiv_grad.dot(robust_loss_total_grad)

    np.save('explainer/data/binaries/util_infl_wo_hess.npy', util_pred_infl_wo_hess)
    np.save('explainer/data/binaries/fair_infl_wo_hess.npy', fair_pred_infl_wo_hess)
    #np.save('explainer/data/binaries/robust_infl_wo_hess.npy', robust_pred_infl_wo_hess)

    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)


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


    np.save('trn.npy', np.append(data.x_train, data.y_train.reshape((-1,1)), 1))
    return val_res, test_res



def post_main_defense(args):
    tik = time.time()

    if args.seed is not None:
        fix_seed(args.seed)

    with open('data/' + args.dataset  + '/meta.json', 'r+') as f:
        json_data = json.load(f)
        json_data['train_path'] = './data/' + args.dataset + '/train_clean.csv'
        f.seek(0)
        json.dump(json_data, f, indent=4)
        f.truncate()


    """ initialization"""

    data: DataTemplate2 = fetch_data2(args.dataset)
    model = LogisticRegression(l2_reg=data.l2_reg)

    if args.model_type == 'nn':
        model = NNLastLayerIF(input_dim=data.dim, base_model_cls=MLPClassifier, l2_reg=1e-4)

    val_evaluator, test_evaluator = Evaluator(data.s_val, "val"), Evaluator(data.s_test, "test")

    """ vanilla training """

    model.fit(data.x_train, data.y_train)
    if args.dataset == "toy" and args.save_model == "y":
        pickle.dump(model.model, open("toy/model.pkl", "wb"))


    _, pred_label_val = model.pred(data.x_val)
    _, pred_label_test = model.pred(data.x_test)

    val_res = val_evaluator(data.y_val, pred_label_val)
    test_res = test_evaluator(data.y_test, pred_label_test)

    
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

    I1 = np.load('explainer/data/binaries/util_infl.npy')
    I2 = np.load('explainer/data/binaries/fair_infl.npy')

    # inner product
    # I1 = np.load('explainer/data/binaries/util_infl_wo_hess.npy')
    # I2 = np.load('explainer/data/binaries/fair_infl_wo_hess.npy')

    indices_to_delete = I2.argsort()[::-1][-args.points_to_delete:][::-1].tolist()

    print("# Indices to Delete ==> ", len(indices_to_delete))

    X_new = []
    for i,x in enumerate(X_org):
        if i in indices_to_delete:
            continue
        X_new.append(X_org[i])
    X_new = np.array(X_new)


    print(X_new.shape)

    X = X_new
    np.save('2trn.npy', X)


if __name__ == "__main__":
    args = parse_args()

    args.seed = 42
    args.metric = 'dp'

    print("\nOriginal Model Performance:\n")
    pre_val_res, pre_test_res = pre_main_clean(args) #Run pre code
    print("\nPoisoned Model Performance:\n")
    pre_val_res, pre_test_res = pre_main_poisoned(args) #Run pre code

    if args.only_pre == 'y': #If only pre needed, exit
        exit(0)

    if args.dataset == 'nraa_attack_drug' or args.dataset == 'nraa_attack_compas':
        args.points_to_delete = 150
    if args.dataset == 'nraa_attack_german':
        args.points_to_delete = 40
    if args.dataset == 'raa_attack_german':
        args.points_to_delete = 2
    if args.dataset == 'raa_attack_compas':
        args.points_to_delete = 1
    if args.dataset == 'raa_attack_drug':
        args.points_to_delete = 125

    fair_deletion_process(args)
    print("\nTrimmed Model Performance:\n")
    post_val_res, post_test_res = post_main_defense(args)

