
import time
import tqdm
import copy
import torch
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import KFold

import os, sys; sys.path.append(os.path.dirname(__file__))
from common import eegmix
from common import psemix
from common.eegdataset import EEGDataset
from models.eegmixer import EEGMixer
from models.retmil import RetMIL
import engine as utils


class CrossValidation:
    def __init__(self, args):
        self.args = args
        if args.aug == 'none':
            self.args.augment = None
        else:
            # self.args.augment = eegmix.CutMix(method='mixup')
            self.args.augment = psemix.PSEMix(alpha=1.0, method=args.aug, n_pseb=5, n_pheno=5)

    def k_fold_cv(self, data, labels, lengths, subjects, folds=10, verbose=True):
        # data: subjects of [trials x segments x timepoints x channels]
        # labels: subjects of [trials x segments]
        if folds == len(subjects): 
            # leave-one-subject-out cross validation
            splits = []
            for i in range(folds):
                idx_train = np.array([j for j in range(folds) if j != i])
                idx_test = np.array([i])
                splits.append((idx_train, idx_test))
        else:
            # k-fold cross validation
            kfold = KFold(n_splits=folds, shuffle=True, random_state=42)
            splits = list(kfold.split(data))

        return self.do_cv(data, labels, lengths, folds, splits, verbose)

    def do_cv(self, data, labels, lengths, folds, splits, verbose=True):
        args = self.args
        train_losses, train_accus, train_f1s, train_aucs = np.zeros(folds), np.zeros(folds), np.zeros(folds), np.zeros(folds)
        valid_losses, valid_accus, valid_f1s, valid_aucs = np.zeros(folds), np.zeros(folds), np.zeros(folds), np.zeros(folds)
        test_losses, test_accus, test_f1s, test_aucs = np.zeros(folds), np.zeros(folds), np.zeros(folds), np.zeros(folds)
        train_ytrues, train_yprobs, valid_ytrues, valid_yprobs, test_ytrues, test_yprobs = [], [], [], [], [], []
        
        for fold, (idx_train, idx_test) in enumerate(splits):
            # split data into training and testing
            data_train, labels_train, lengths_train, data_test, labels_test, lengths_test = self.prepare_data(
                data, labels, lengths, idx_train, idx_test
            )
            # split training data into training and validation
            data_train, labels_train, lengths_train, data_valid, labels_valid, lengths_valid = self.split_balance_class(
                data_train, labels_train, lengths_train, train_ratio=0.8, random=True
            )

            
            # create dataset and dataloader
            train_dataset = EEGDataset(data_train, labels_train, args.tf_epoch)
            valid_dataset = EEGDataset(data_valid, labels_valid, args.tf_epoch)
            test_dataset  = EEGDataset(data_test, labels_test, args.tf_epoch)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=args.collate_fn,
            )
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=args.collate_fn,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=args.collate_fn,
            )

            # create model
            print(f"=> creating model {args.arch}")
            
            # instance feature encoder
            encoder = EEGMixer(
                0, args.patch_size, args.n_channels, embed_dim=args.embed_dim, pooling=True,
                n_layers=args.num_layers, kernel_size=15,
            )
            
            # MIL pooling
            model = RetMIL(encoder, num_classes=args.num_classes, embed_dim=args.embed_dim)
            
            model = model.to(args.device)

            if hasattr(args, 'augment') and args.augment is not None:
                basecrit = torch.nn.CrossEntropyLoss(reduction='none').to(args.device)
                criterion = eegmix.MixupLoss(basecrit).to(args.device)
            else:
                criterion = torch.nn.CrossEntropyLoss().to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_accu, best_loss = 0., 0.
            best_modelpath = os.path.join(args.output_dir, f"checkpoint/task_{args.task}_fold_{fold}.pth")
            progress_bar = tqdm.tqdm(range(args.epochs)) if verbose else range(args.epochs)
            for epoch in progress_bar:
                start = time.time()
                utils.adjust_learning_rate(optimizer, epoch, args)
                lr = optimizer.param_groups[0]["lr"]

                train_losses[fold], train_accus[fold], train_ytrue, train_yprob = \
                    utils.train_epoch(train_loader, model, criterion, optimizer, epoch, args)
                
                valid_losses[fold], valid_accus[fold], valid_ytrue, valid_yprob = \
                    utils.evaluate(valid_loader, model, criterion, epoch, args)

                if hasattr(args, 'writer') and args.writer:
                    args.writer.add_scalar(f"Fold_{fold}/Accu/train", train_accus[fold], epoch)
                    args.writer.add_scalar(f"Fold_{fold}/Accu/valid", valid_accus[fold], epoch)
                    args.writer.add_scalar(f"Fold_{fold}/Loss/train", train_losses[fold], epoch)
                    args.writer.add_scalar(f"Fold_{fold}/Loss/valid", valid_losses[fold], epoch)
                    args.writer.add_scalar(f"Fold_{fold}/Misc/learning_rate", lr, epoch)

                if verbose:
                    progress_bar.set_description(
                        f"Fold: {fold}, Epoch time = {time.time() - start:.3f}s"
                    )
                
                if valid_accus[fold] > best_accu:
                    best_accu = valid_accus[fold]
                    utils.save_model(model, best_modelpath)

            utils.load_model(model, best_modelpath, strict=True)
            test_losses[fold], test_accus[fold], test_ytrue, test_yprob = utils.evaluate(
                test_loader, model, criterion, epoch, args, trialwise=False)
            
            train_ytrues.append(train_ytrue), train_yprobs.append(train_yprob)
            valid_ytrues.append(valid_ytrue), valid_yprobs.append(valid_yprob)
            test_ytrues.append(test_ytrue), test_yprobs.append(test_yprob)
            
            train_ypred = np.argmax(train_yprob, axis=1)
            valid_ypred = np.argmax(valid_yprob, axis=1)
            test_ypred = np.argmax(test_yprob, axis=1)
            train_accus[fold] = metrics.accuracy_score(train_ytrue, train_ypred)
            valid_accus[fold] = metrics.accuracy_score(valid_ytrue, valid_ypred)
            test_accus[fold] = metrics.accuracy_score(test_ytrue, test_ypred)
            train_f1s[fold] = metrics.f1_score(train_ytrue, train_ypred)
            valid_f1s[fold] = metrics.f1_score(valid_ytrue, valid_ypred)
            test_f1s[fold] = metrics.f1_score(test_ytrue, test_ypred)
            train_aucs[fold] = metrics.roc_auc_score(train_ytrue, train_yprob[:, 1])
            valid_aucs[fold] = metrics.roc_auc_score(valid_ytrue, valid_yprob[:, 1])
            test_aucs[fold] = metrics.roc_auc_score(test_ytrue, test_yprob[:, 1])

            if verbose:
                print(f"Fold: {fold}, "
                    f"Valid accu: {valid_accus[fold]:.3f}, loss: {valid_losses[fold]:.3f}, "
                    f"Test accu: {test_accus[fold]:.3f}, loss: {test_losses[fold]:.3f}")
                
        test_ytrues = np.concatenate(test_ytrues)
        test_yprobs = np.concatenate(test_yprobs)
        df_yy = pd.DataFrame({
            'y_true': test_ytrues,
            'y_prob': test_yprobs[:, 1],
        })
        df_yy.to_csv(os.path.join(args.output_dir, 'yy_' + args.mil + '.csv'))

        return train_losses, train_accus, train_f1s, train_aucs, \
            valid_losses, valid_accus, valid_f1s, valid_aucs, \
            test_losses, test_accus, test_f1s, test_aucs
    
    def prepare_data(self, data, labels, lengths, idx_train, idx_test):
        """
        1. get training and testing data according to the index
        2. numpy.array-->torch.tensor
        :param data: subjects of (trials, segments, timepoints, channels)
        :param labels: subjects of (trials, segments,)
        :param lengths: subjects of (trials,)
        :param idx_train: index of training subjects
        :param idx_test: index of testing subjects
        :return: data, labels, lengths for training and testing
        """
        data_train, labels_train = [], []
        for i in idx_train:
            data_i = data[i]
            labels_i = labels[i]
            lengths_1 = lengths[i]
            lengths_1 = np.cumsum(lengths_1) # allow different number of instances in trial
            data_i = np.split(data_i, lengths_1[:-1], axis=0)
            labels_i = np.split(labels_i, lengths_1[:-1], axis=0)
            data_train.append(data_i)
            labels_train.append(labels_i)
        data_test, labels_test = [], []
        for i in idx_test:
            data_i = data[i]
            labels_i = labels[i]
            lengths_1 = lengths[i]
            lengths_1 = np.cumsum(lengths_1) # allow different number of instances in trial
            data_i = np.split(data_i, lengths_1[:-1], axis=0)
            labels_i = np.split(labels_i, lengths_1[:-1], axis=0)
            data_test.append(data_i)
            labels_test.append(labels_i)
        data_train = sum(data_train, [])
        labels_train = sum(labels_train, [])
        data_test = sum(data_test, [])
        labels_test = sum(labels_test, [])
        lengths_train = [len(data_train[i]) for i in range(len(data_train))]
        lengths_test = [len(data_test[i]) for i in range(len(data_test))]
        labels_train = [lb[0] for lb in labels_train]
        labels_test = [lb[0] for lb in labels_test]
        labels_train, labels_test = np.array(labels_train), np.array(labels_test)
        # data_train, data_test = self.normalize(train=data_train, test=data_test)
        return data_train, labels_train, lengths_train, data_test, labels_test, lengths_test

    def normalize(self, train, test):
        """
        this function do standard normalization for EEG channel by channel
        :param train: training data
        :param test: testing data
        :return: normalized training and testing data
        """
        # data: ... x timepoints x channels
        mean = 0
        std = 0
        for channel in range(train.shape[-1]):
            mean = np.mean(train[..., channel])
            std = np.std(train[..., channel])
            train[..., channel] = (train[..., channel] - mean) / std
            test[..., channel] = (test[..., channel] - mean) / std
        return train, test

    def split_balance_class(self, data, labels, lengths, train_ratio=0.8, random=True):
        np.random.seed(0)
        # get index for each class
        index_0 = np.where(labels == 0)[0]
        index_1 = np.where(labels == 1)[0]
        # for class 0
        index_random_0 = copy.deepcopy(index_0)
        # for class 1
        index_random_1 = copy.deepcopy(index_1)
        # shuffle
        if random == True:
            np.random.shuffle(index_random_0)
            np.random.shuffle(index_random_1)
        # split
        idx_train = np.concatenate(
            (index_random_0[:int(len(index_random_0) * train_ratio)], 
            index_random_1[:int(len(index_random_1) * train_ratio)]), 
            axis=0
        )
        idx_valid = np.concatenate(
            (index_random_0[int(len(index_random_0) * train_ratio):],
            index_random_1[int(len(index_random_1) * train_ratio):]),
            axis=0
        )
        data_train = [data[i] for i in idx_train]
        labels_train = labels[idx_train]
        lengths_train = [lengths[i] for i in idx_train]
        data_valid = [data[i] for i in idx_valid]
        labels_valid = labels[idx_valid]
        lengths_valid = [lengths[i] for i in idx_valid]
        return data_train, labels_train, lengths_train, data_valid, labels_valid, lengths_valid
