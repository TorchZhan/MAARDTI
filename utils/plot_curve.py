#!/usr/bin/env python
# coding=utf-8
'''
Author: Zhan
Date: 2023-08-15 00:20:18
'''
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def roc_curve_self(save_path, dataset_class, tpr_rc, fpr_rc):

	np.save(save_path+'/{}_fpr_rc.npy'.format(dataset_class),np.array(fpr_rc))
	np.save(save_path+'/{}_tpr_rc.npy'.format(dataset_class),np.array(tpr_rc))

	#sigle
	roc_auc = auc(fpr_rc, tpr_rc)
	plt.figure()
	plt.plot(fpr_rc, tpr_rc,
		lw=1, label= 'test' + ' (AUC = %0.4f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	fontsize = 14
	plt.xlabel('False Positive Rate', fontsize = fontsize)
	plt.ylabel('True Positive Rate', fontsize = fontsize)
	plt.title('Receiver Operating Characteristic Curve (ROC)')
	plt.legend(loc="lower right")
	plt.savefig(save_path +'/{}_roc.jpg'.format(dataset_class),dpi=300)
	plt.savefig(save_path +'/{}_roc.pdf'.format(dataset_class),dpi=300)

	return 

def roc_curve_all(save_path, dataset_class, tpr_rcs, fpr_rcs):
	
	colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'black']
	roc_auc_all = []
	max_len = []
	for j in range(len(tpr_rcs)):
		max_len.append(len(tpr_rcs[j]))
		max_len.append(len(fpr_rcs[j]))
		max_out = np.max(max_len)
	
	mean_fpr = [0] * max_out
	mean_tpr = [0] * max_out
	for j in range(len(tpr_rcs)):	
		diff_tpr = max_out - len(tpr_rcs[j])
		diff_fpr = max_out - len(fpr_rcs[j])

		tpr_rcs[j] = np.append(tpr_rcs[j], np.array([1]*diff_tpr))
		fpr_rcs[j] = np.append(fpr_rcs[j], np.array([1]*diff_fpr))

		mean_tpr = [x+y for x, y in zip(mean_tpr, tpr_rcs[j])]
		mean_fpr = [x+y for x, y in zip(mean_fpr, fpr_rcs[j])]

	mean_tpr_all = [i/len(tpr_rcs) for i in mean_tpr]
	mean_fpr_all = [i/len(tpr_rcs) for i in mean_fpr]

	np.save(save_path+'/{}_Mean_fpr_rc.npy'.format(dataset_class),np.array(mean_fpr_all))
	np.save(save_path+'/{}_Mean_tpr_rc.npy'.format(dataset_class),np.array(mean_tpr_all))

	plt.figure()
	for i in range(len(fpr_rcs)):
		roc_auc = auc(fpr_rcs[i], tpr_rcs[i])
		
		roc_auc_all.append(roc_auc)
		plt.plot(fpr_rcs[i], tpr_rcs[i], color=colorlist[i],
			lw=1, label= 'Fold-%d (AUC = %0.4f)' % (i+1, roc_auc))
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

	auc_mean = np.mean(roc_auc_all)	
	plt.plot(mean_fpr_all, mean_tpr_all, color=colorlist[5],
			lw=2, label= 'Mean' + ' (AUC = %0.4f)' % (auc_mean))
	
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	fontsize = 14
	plt.xlabel('False Positive Rate', fontsize = fontsize)
	plt.ylabel('True Positive Rate', fontsize = fontsize)
	plt.title('Receiver Operating Characteristic Curve (ROC)')
	plt.legend(loc="lower right")

	plt.savefig(save_path +'/{}_all_roc.jpg'.format(dataset_class),dpi=300)
	plt.savefig(save_path +'/{}_all_roc.pdf'.format(dataset_class),dpi=300)

	return 


def prauc_curve_self(save_path, dataset_class, tpr_pr, fpr_pr):

	np.save(save_path+'/{}_fpr_pr.npy'.format(dataset_class),np.array(fpr_pr))
	np.save(save_path+'/{}_tpr_pr.npy'.format(dataset_class),np.array(tpr_pr))

	#sigle
	roc_auc = auc(fpr_pr, tpr_pr)
	plt.figure()
	plt.plot(fpr_pr, tpr_pr,
		lw=2, label= 'test' + ' (AUPR = %0.4f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	fontsize = 14
	plt.xlabel('Recall', fontsize = fontsize)
	plt.ylabel('Precision', fontsize = fontsize)
	plt.title('Precision Recall Curve (PR)')
	plt.legend(loc="lower right")
	plt.savefig(save_path +'/{}_pr.jpg'.format(dataset_class),dpi=300)
	plt.savefig(save_path +'/{}_pr.pdf'.format(dataset_class),dpi=300)

	return 

def prauc_curve_all(save_path, dataset_class, tpr_prs, fpr_prs):
	
	colorlist = ['red', 'gold', 'purple', 'green', 'blue', 'black']
	roc_auc_all = []
	max_len = []
	for j in range(len(tpr_prs)):
		max_len.append(len(tpr_prs[j]))
		max_len.append(len(fpr_prs[j]))
		max_out = np.max(max_len)
	
	mean_fpr = [0] * max_out
	mean_tpr = [0] * max_out
	for j in range(len(tpr_prs)):	
		diff_tpr = max_out - len(tpr_prs[j])
		diff_fpr = max_out - len(fpr_prs[j])

		tpr_prs[j] = np.append(np.array([1]*diff_tpr), tpr_prs[j])
		fpr_prs[j] = np.append(np.array([1]*diff_fpr), fpr_prs[j])

		mean_tpr = [x+y for x, y in zip(mean_tpr, tpr_prs[j])]
		mean_fpr = [x+y for x, y in zip(mean_fpr, fpr_prs[j])]

	plt.figure()
	for i in range(len(fpr_prs)):
		roc_auc = auc(fpr_prs[i], tpr_prs[i])
		
		roc_auc_all.append(roc_auc)
		plt.plot(fpr_prs[i], tpr_prs[i], color=colorlist[i],
			lw=1, label= 'Fold-%d (AUPR = %0.4f)' % (i+1, roc_auc))
		plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

	auc_mean = np.mean(roc_auc_all)
	mean_tpr_all = [i/len(tpr_prs) for i in mean_tpr]
	mean_fpr_all = [i/len(tpr_prs) for i in mean_fpr]

	
	plt.plot(mean_fpr_all, mean_tpr_all, color=colorlist[5],
			lw=1, label= 'Mean' + ' (AUPR = %0.4f)' % (auc_mean))
	
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	fontsize = 14
	plt.xlabel('Recall', fontsize = fontsize)
	plt.ylabel('Precision', fontsize = fontsize)
	plt.title('Precision Recall Curve (PR)')
	plt.legend(loc="lower right")

	plt.savefig(save_path +'/{}_all_pr.jpg'.format(dataset_class),dpi=300)
	plt.savefig(save_path +'/{}_all_pr.pdf'.format(dataset_class),dpi=300)

	np.save(save_path+'/{}_Mean_fpr_pr.npy'.format(dataset_class),np.array(mean_fpr_all))
	np.save(save_path+'/{}_Mean_tpr_pr.npy'.format(dataset_class),np.array(mean_tpr_all))

	return 


