import os
import numpy as np
from utils.plot_curve import *
import matplotlib.pyplot as plt

def show_result(DATASET, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List, PATH):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)

    print("The model's results:")
    filepath = PATH + '/result.txt'

    with open(filepath, 'w') as f:
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(
            Accuracy_mean, Accuracy_var) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(
            Precision_mean, Precision_var) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(
            Recall_mean, Recall_var) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var) + '\n')
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_var))
    print('Precision(std):{:.4f}({:.4f})'.format(Precision_mean, Precision_var))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_var))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_var))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_var))

