import os
import torch
import hydra
import torch.nn as nn
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from data import *
from tensorboardX import SummaryWriter

from hydra.utils import get_original_cwd, to_absolute_path
import warnings
warnings.filterwarnings('ignore')
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

from model import *

import torch.optim as optim
from torch.utils.data import DataLoader
from utils.DatasetFunction import CustomDataSet, collate_fn
from utils.EarlyStopping import EarlyStopping
from utils.TestModel import *
from utils.ShowResult import show_result
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
import torch.nn.functional as F
import time 


class Trainer(object):

    def __init__(self, args):
        
        self.args = args
        self.console = Console()

        self.console.log('=> [0] Initial Task')
        self.writer = SummaryWriter(comment = f'Task: {args.ds.task}, Data: {args.ds.data}')

        self.console.log('=> [1] Initial Settings')
        np.random.seed(args.basic.seed)
        torch.manual_seed(args.basic.seed)
        torch.cuda.manual_seed(args.basic.seed)
        cudnn.enabled = True

        self.console.log('=> [2] Initial Models & Optimizers')
        self.weight_loss = weight_loss(self.args)
        self.loss_fn  = get_loss_fn(self.args, self.weight_loss)

        self.console.log('=> [3] Preparing Dataset & Shuffle Dataset')
        self.console.log(f'[green]    => Train in {args.ds.task}')
        self.dataset  = load_data(args)

        self.console.log('=> [4] Split Dataset to Train-Valid-Test')
        self.train_data_list, self.test_data_list, len_train, len_test = split_data(self.dataset)
        self.console.log(f"[green]    => Number of Train&Val set: {len_train}")                                             
        self.console.log(f"[green]    => Number of Test set: {len_test}")                                             

    def run(self):
        
        device = torch.device('cuda')

        Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

        for i_fold in range(self.args.ds.fold):
            self.console.log(f'    => Training...: ')
            self.console.log(f'    => *********** The {i_fold} fold ***********')
            train_dataset, valid_dataset = get_kfold_data(i_fold, self.train_data_list, k=self.args.ds.fold)
            
            train_dataset = CustomDataSet(train_dataset)
            valid_dataset = CustomDataSet(valid_dataset)
            test_dataset = CustomDataSet(self.test_data_list)
            train_size = len(train_dataset)

            train_dataset_loader = DataLoader(
                train_dataset, 
                batch_size=self.args.ds.batch_size, 
                shuffle=True, 
                num_workers=0,
                collate_fn=collate_fn, 
                drop_last=True
            )

            valid_dataset_loader = DataLoader(
                valid_dataset, 
                batch_size=self.args.ds.batch_size, 
                shuffle=False, 
                num_workers=0,
                collate_fn=collate_fn, 
                 drop_last=True
            )

            test_dataset_loader = DataLoader(
                test_dataset, 
                batch_size=self.args.ds.batch_size, 
                shuffle=False, 
                num_workers=0,
                collate_fn=collate_fn, 
                 drop_last=True
            )

            self.model = ModelMAAR(self.args, device).to(device)
            para_mb = str(float(count_parameters_in_MB(self.model)))
            # print(para_mb)
            self.console.log(f'[red]=> Supernet Parameters: {count_parameters_in_MB(self.model):.4f} MB')

            # print(self.model)
            """Initialize weights"""
            weight_p, bias_p = [], []
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for name, p in self.model.named_parameters():
                if 'bias' in name:
                    bias_p += [p]
                else:
                    weight_p += [p]

            # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.ds.Learning_rate)
            
            self.optimizer = optim.AdamW(
                [{
                 'params'       : weight_p, 
                 'weight_decay' : self.args.ds.weight_decay},
                {
                 'params'       : bias_p, 
                 'weight_decay' : 0}], 
                lr = self.args.ds.Learning_rate
            )
            
            # self.optimizer = optim.RAdam(
            #     [{'params'      : weight_p, 
            #       'weight_decay': self.args.ds.weight_decay}, 
            #     {
            #       'params'      : bias_p, 
            #       'weight_decay': 0}], 
            #     lr= self.args.ds.Learning_rate)

            self.scheduler = optim.lr_scheduler.CyclicLR(
                self.optimizer, 
                base_lr        = self.args.ds.Learning_rate, 
                max_lr         = self.args.ds.Learning_rate * 10, 
                cycle_momentum = False,
                step_size_up   = train_size // self.args.ds.batch_size
            )
            
            """Output files"""
            path = os.path.join(get_original_cwd(), self.args.ds.r_savepath, self.args.ds.data, self.args.ds.d_path)
            
            save_path = path + "/{}".format(i_fold+1)
            writer = SummaryWriter(log_dir=save_path, comment='')
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
     
            file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

            early_stopping = EarlyStopping(savepath=save_path, patience=self.args.ds.Patience, verbose=True, delta=0)
            accuracy_record, epoch_record = [], []
            
            for epoch in range(1, self.args.ds.Epoch + 1):
                if early_stopping.early_stop == True:
                    break
                train_pbar = tqdm(
                    enumerate(BackgroundGenerator(train_dataset_loader)),
                    total =len(train_dataset_loader))

                """train"""
                train_losses_in_epoch = []
                self.model.train()
                for train_i, train_data in train_pbar:
                    train_compounds, train_proteins, train_compounds_mask, train_proteins_mask, train_labels = train_data

                    train_compounds = train_compounds.to(device)
                    train_proteins = train_proteins.to(device)
                    #mask
                    train_compounds_mask = train_compounds_mask.to(device)
                    train_proteins_mask = train_proteins_mask.to(device)

                    train_labels = train_labels.to(device)

                    self.optimizer.zero_grad()

                    predicted_interaction = self.model(train_compounds, train_proteins, train_compounds_mask, train_proteins_mask)
                    train_loss = self.loss_fn(predicted_interaction, train_labels)
                    train_losses_in_epoch.append(train_loss.item())
                    train_loss.backward()
                    
                    self.optimizer.step()
                    self.scheduler.step()

                train_loss_a_epoch = np.average(train_losses_in_epoch)  #一次epoch的平均训练loss
                # writer.add_scalar('TrainLoss', train_loss_a_epoch, epoch)

                """valid"""
                valid_pbar = tqdm(
                    enumerate(BackgroundGenerator(valid_dataset_loader)),
                    total=len(valid_dataset_loader))
                valid_losses_in_epoch = []
                self.model.eval()
                
                Y, P, S = [], [], []
                with torch.no_grad():
                    for valid_i, valid_data in valid_pbar:

                        valid_compounds, valid_proteins, valid_compounds_mask, valid_proteins_mask, valid_labels = valid_data

                        valid_compounds = valid_compounds.to(device)
                        valid_proteins = valid_proteins.to(device)

                         #mask
                        valid_compounds_mask = valid_compounds_mask.to(device)
                        valid_proteins_mask = valid_proteins_mask.to(device)

                        valid_labels = valid_labels.to(device)

                        valid_scores = self.model(valid_compounds, valid_proteins, valid_compounds_mask, valid_proteins_mask)
                        valid_loss = self.loss_fn(valid_scores, valid_labels)
                        valid_losses_in_epoch.append(valid_loss.item())
                        valid_labels = valid_labels.to('cpu').data.numpy()
                        valid_scores = F.softmax(
                            valid_scores, 1).to('cpu').data.numpy()
                        valid_predictions = np.argmax(valid_scores, axis=1)
                        valid_scores = valid_scores[:, 1]

                        Y.extend(valid_labels)
                        P.extend(valid_predictions)
                        S.extend(valid_scores)

                Precision_dev = precision_score(Y, P)
                Reacll_dev = recall_score(Y, P)
                Accuracy_dev = accuracy_score(Y, P)
                AUC_dev = roc_auc_score(Y, S)
                tpr, fpr, _ = precision_recall_curve(Y, S)
                PRC_dev = auc(fpr, tpr)
                valid_loss_a_epoch = np.average(valid_losses_in_epoch)
                
                epoch_record.append([epoch, Accuracy_dev, valid_loss_a_epoch, AUC_dev, PRC_dev])

                epoch_len = len(str(self.args.ds.Epoch))
                print_msg = (f'[{epoch:>{epoch_len}}/{self.args.ds.Epoch:>{epoch_len}}] ' +
                            f'train_loss: {train_loss_a_epoch:.5f} ' +
                            f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                            f'valid_AUC: {AUC_dev:.5f} ' +
                            f'valid_PRC: {PRC_dev:.5f} ' +
                            f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                            f'valid_Precision: {Precision_dev:.5f} ' +
                            f'valid_Reacll: {Reacll_dev:.5f} ')
                
                # writer.add_scalar('Valid_Loss', valid_loss_a_epoch, epoch)
                # writer.add_scalar('Valid_AUC', AUC_dev, epoch)
                # writer.add_scalar('Valid_AUPR', PRC_dev, epoch)
                # writer.add_scalar('Valid_Accuracy', Accuracy_dev, epoch)
                # writer.add_scalar('Valid_Precision', Precision_dev, epoch)
                # writer.add_scalar('Valid_Reacll', Reacll_dev, epoch)
                # writer.add_scalar('Learn_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                print(print_msg)

                '''save checkpoint and make decision when early stop'''
                # early_stopping(valid_loss_a_epoch, self.model, epoch)
                early_stopping(Accuracy_dev, self.model, epoch)

            '''load best checkpoint'''
            self.model.load_state_dict(torch.load(early_stopping.savepath + '/valid_best_checkpoint.pth'))

            with open(save_path + '/' + 'result_valid_record.txt', 'a') as f:
                for i in range(len(epoch_record)):
                    f.write(str(epoch_record[i][0]) + " " + str(epoch_record[i][1]) + " " + str(epoch_record[i][2]) +'\n')

            '''test model'''
            trainset_test_stable_results, _, _, _, _, _  = test_model(
                self.model, train_dataset_loader, save_path, self.args.ds.data, self.loss_fn, path, dataset_class="Train", FOLD_NUM=1, PLOT=2,)
            
            validset_test_stable_results, _, _, _, _, _ = test_model(
                self.model, valid_dataset_loader, save_path, self.args.ds.data, self.loss_fn, path, dataset_class="Valid", FOLD_NUM=1, PLOT=2)
            
            testset_test_stable_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
                self.model, test_dataset_loader, save_path, self.args.ds.data, self.loss_fn, path, dataset_class="Test", FOLD_NUM=1, PLOT=1)
            
            AUC_List_stable.append(AUC_test)
            Accuracy_List_stable.append(Accuracy_test)
            AUPR_List_stable.append(PRC_test)
            Recall_List_stable.append(Recall_test)
            Precision_List_stable.append(Precision_test)
            with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:                
                f.write("Test the stable model" + '\n')
                f.write("The Parameter is {} MB".format(para_mb) + '\n')
                f.write(trainset_test_stable_results + '\n')
                f.write(validset_test_stable_results + '\n')
                f.write(testset_test_stable_results + '\n')

        show_result(self.args.ds.data, Accuracy_List_stable, Precision_List_stable,
                    Recall_List_stable, AUC_List_stable, AUPR_List_stable, path)


@hydra.main(config_path = 'conf', config_name = 'defaults')
def app(args):
    start_time = time.time()
    OmegaConf.set_struct(args, False)
    console = Console()
    vis = Syntax(OmegaConf.to_yaml(args), "yaml", theme="monokai", line_numbers=True)
    richPanel = Panel.fit(vis)
    console.print(richPanel)

    data_path = os.path.join(get_original_cwd(), args.ds.r_savepath, args.ds.data, args.ds.d_path)

    Path(data_path).mkdir(parents = True, exist_ok = True)
    with open(os.path.join(data_path, "configs.txt"), "w") as f:
        f.write(str(args))

    Trainer(args).run()

    end_time = time.time()  # 记录程序结束运行时间
    console.log(f'[red]    => Completely Cost Time: {(end_time - start_time)/3600} h')
    with open(os.path.join(data_path, "result.txt"), "a") as f:
        f.write("Complete Time:{} h".format((end_time-start_time)/3600))

if __name__ == '__main__':
    app()
