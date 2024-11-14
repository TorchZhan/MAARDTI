#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhan
Date: 2023-08-23 15:13:50
'''
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score,roc_curve)
from utils.plot_curve import *
import matplotlib.pyplot as plt

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6

def test_process(MODEL, pbar, LOSS, FOLD_NUM):
    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, compounds_mask, proteins_mask, labels = data
            compounds = compounds.cuda()
            proteins = proteins.cuda()
            compounds_mask = compounds_mask.cuda()
            proteins_mask  = proteins_mask.cuda()
            labels = labels.cuda()


            if isinstance(MODEL, list):
                predicted_scores = torch.zeros(2).cuda()
                for i in range(len(MODEL)):
                    predicted_scores = predicted_scores + \
                        MODEL[i](compounds, proteins)
                predicted_scores = predicted_scores / FOLD_NUM
            else:
                predicted_scores = MODEL(compounds, proteins, compounds_mask, proteins_mask)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.to('cpu').data.numpy()
            predicted_scores = F.softmax(
                predicted_scores, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]
    
            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())

    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    fpr_rc, tpr_rc, _ = roc_curve(Y, S)
    AUC = roc_auc_score(Y, S)
    
    tpr_pr, fpr_pr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr_pr, tpr_pr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return Y, P, S, test_loss, Accuracy, Precision, Recall, AUC, PRC, tpr_rc, fpr_rc, tpr_pr, fpr_pr

tpr_rcs, fpr_rcs, tpr_prs, fpr_prs = [], [], [], []

def test_model(MODEL, dataset_loader, save_path, DATASET, LOSS, path, dataset_class="Train", save=True, FOLD_NUM=1, PLOT=1):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader))
    T, P, S, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test, tpr_rc, fpr_rc, tpr_pr, fpr_pr = test_process(
        MODEL, test_pbar, LOSS, FOLD_NUM)
    
    if PLOT == 1:
        
        tpr_rcs.append(tpr_rc)
        fpr_rcs.append(fpr_rc)
        tpr_prs.append(tpr_pr)
        fpr_prs.append(fpr_pr)

        roc_curve_self(save_path, dataset_class, tpr_rc, fpr_rc)
        prauc_curve_self(save_path, dataset_class, tpr_pr, fpr_pr)

        if len(tpr_rcs) != 0 :
            roc_curve_all(path, dataset_class, tpr_rcs, fpr_rcs)
            prauc_curve_all(path, dataset_class, tpr_prs, fpr_prs)
    if save:
        if FOLD_NUM == 1:
            filepath = save_path + \
                "/{}_{}_prediction.txt".format(DATASET, dataset_class)
            
        with open(filepath, 'a') as f:
            for i in range(len(T)):
                f.write(str(T[i]) + " " + str(P[i]) + " " + str(S[i]) +'\n')
    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test
