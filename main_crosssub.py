
import os
import torch
import random
import datetime
import warnings
import argparse
import numpy as np
import pandas as pd

import sys; sys.path.append(os.path.dirname(__file__))
from common.eegdataset import ToTensor
from engine import MILSequenceCollator
from cv_crosssub_mil import CrossValidation as CV_MIL
import deapreader


emotion_datasets = {
    'deap' : {
        'data_dir' : 'e:/eegdata/emotion/deap/',
        'output_dir' : 'e:/eegdata/emotion/deap/output/',
    },
}

parser = argparse.ArgumentParser(description='EmotionNet Training from Scratch')
parser.add_argument('-D', '--dataset', default='deap', metavar='PATH',
                    help='dataset used')
parser.add_argument('--mil', default='retmil', metavar='MIL',
                    help="whether to use MIL or which MIL method ('transmil', 'retmil')"
                         "(default: none)")
parser.add_argument('--aug', metavar='AUGMENT', default='cutmix',
                    help='augmentation method (default: psebmix)')
parser.add_argument('-t', '--task', default='arousal', metavar='TASK',
                    help='emotion task (default: arousal)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='eegmixer',
                    help='model architecture (default: acrnn)')
parser.add_argument('--twin', default=2.0, type=float, metavar='N',
                    help='time window length in seconds (default: 1.0)')
parser.add_argument('--feature', metavar='FEATURE', default='processed',
                    help='input feature type (default: processed)')
parser.add_argument('-p', '--patch-size', default=16, type=int, metavar='N',
                    help='patch size (default: 16) when dividing the long signal into windows')
parser.add_argument('--embed_dim', default=64, type=int, metavar='N',
                    help='embedded feature dimension (default: 192)')
parser.add_argument('--num_layers', default=3, type=int, metavar='N',
                    help='number of transformer layers (default: 6)')
parser.add_argument('--num_heads', default=8, type=int, metavar='N',
                    help='number of heads for multi-head attention (default: 6)')
parser.add_argument('--global_pool', action='store_true', default=True)
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 1)')
parser.add_argument('--folds', default=10, type=int, metavar='N',
                    help='number of folds cross-valiation (default: 20)')
parser.add_argument('--splits', default='', type=str, metavar='PATH',
                    help='path to cross-validation splits file (default: none)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--optimizer', default='sgd', type=str,
                    choices=['adam', 'adamw', 'sgd', 'lars'],
                    help='optimizer used to learn the model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--min_lr', type=float, default=1e-8, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0')
parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                    help='epochs to warmup LR')
parser.add_argument('--schedule', default='cos', type=str,
                    choices=['cos', 'step'],
                    help='learning rate schedule (how to change lr)')
parser.add_argument('--lr_drop', default=[0.6, 0.8], nargs='*', type=float,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-s', '--save-freq', default=50, type=int,
                    metavar='N', help='save frequency (default: 100)')
parser.add_argument('-e', '--evaluate', action='store_true',
                    help='evaluate on the test dataset')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading code
    print("=> loading dataset {} from '{}'".format(args.dataset, args.data_dir))
    tasks = deapreader.tasks
    args.num_classes = 2
    args.collate_fn = MILSequenceCollator()
    data, labels, lengths, subjects = deapreader.load_dataset_preprocessed(args.data_dir+f"{args.feature}_{int(args.twin)}s/")
    labels = [labels[i][..., tasks[args.task]].copy() for i in range(len(labels))]
    
    print('Data for %d subjects has been loaded' % len(data))
    num_subjects = len(data)
    args.folds = num_subjects # leave-one-subject cross-validation
    args.n_trials = len(lengths[0])
    args.n_wavlen = data[0].shape[-2]
    args.n_channels = data[0].shape[-1]
    args.n_segments = data[0].shape[0] // args.n_trials
    args.max_seqlen = int(args.n_wavlen // args.patch_size)
    args.tf_epoch = ToTensor()

    cv = CV_MIL(args)
    train_losses, train_accus, train_f1s, train_aucs, \
    valid_losses, valid_accus, valid_f1s, valid_aucs, \
    test_losses, test_accus, test_f1s, test_aucs, = \
        cv.k_fold_cv(data, labels, lengths, subjects, folds=num_subjects, verbose=True)
    
    # Average over folds
    subjects = subjects + ['average']
    train_losses = np.append(train_losses, np.mean(train_losses))
    train_accus = np.append(train_accus, np.mean(train_accus))
    train_f1s = np.append(train_f1s, np.mean(train_f1s))
    train_aucs = np.append(train_aucs, np.mean(train_aucs))
    valid_losses = np.append(valid_losses, np.mean(valid_losses))
    valid_accus = np.append(valid_accus, np.mean(valid_accus))
    valid_f1s = np.append(valid_f1s, np.mean(valid_f1s))
    valid_aucs = np.append(valid_aucs, np.mean(valid_aucs))
    test_losses  = np.append(test_losses, np.mean(test_losses))
    test_accus  = np.append(test_accus, np.mean(test_accus))
    test_f1s  = np.append(test_f1s, np.mean(test_f1s))
    test_aucs  = np.append(test_aucs, np.mean(test_aucs))
    df_results_subs = pd.DataFrame({
        'subject': subjects,
        'train_losses': train_losses,
        'train_accus': train_accus,
        'train_f1s': train_f1s,
        'train_aucs': train_aucs,
        'valid_losses': valid_losses,
        'valid_accus': valid_accus,
        'valid_f1s': valid_f1s,
        'valid_aucs': valid_aucs,
        'test_losses' : test_losses,
        'test_accus' : test_accus,
        'test_f1s' : test_f1s,
        'test_aucs' : test_aucs,
    })
    df_results_subs.to_csv(os.path.join(
        args.output_dir, f"results_task_{args.task}_{args.arch}.csv")
    )


if __name__ == '__main__':

    args = parser.parse_args()

    args.data_dir = emotion_datasets[args.dataset]['data_dir']
    args.output_dir = emotion_datasets[args.dataset]['output_dir']

    output_prefix = f"crosssub_{args.arch}" if args.mil == "none" else f"crosssub_{args.arch}_{args.mil}"
    output_prefix += "/session_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    if not hasattr(args, 'output_dir'):
        args.output_dir = args.data_dir
    args.output_dir = os.path.join(args.output_dir, output_prefix)
    os.makedirs(args.output_dir)
    print("=> results will be saved to {}".format(args.output_dir))

    main(args)
