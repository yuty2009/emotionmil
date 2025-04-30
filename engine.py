
import os
import math
import torch
import numpy as np


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule after warmup"""
    if not hasattr(args, 'warmup_epochs'):
        args.warmup_epochs = 0
    if not hasattr(args, 'min_lr'):
        args.min_lr = 0.
    if epoch < args.warmup_epochs:
        lr = max(args.min_lr, args.lr * epoch / args.warmup_epochs)
    else:
        lr = args.lr
        if args.schedule in ['cos', 'cosine']:  # cosine lr schedule
            # lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) # without warmup
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        elif args.schedule in ['step', 'stepwise']:  # stepwise lr schedule
            for milestone in args.lr_drop:
                lr *= 0.1 if epoch >= int(milestone * args.epochs) else 1.
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def save_model(model, savepath):
    if not os.path.exists(os.path.dirname(savepath)):
        os.makedirs(os.path.dirname(savepath))
    torch.save(model.state_dict(), savepath)


def load_model(model, loadpath, strict=False):
    if os.path.isfile(loadpath):
        checkpoint = torch.load(loadpath, map_location='cpu')
        msg = model.load_state_dict(checkpoint, strict=strict)
        if not strict: print(msg.missing_keys)
        print("=> loaded checkpoint '{}'".format(loadpath))
    else:
        print("=> no checkpoint found at '{}'".format(loadpath))


def train_epoch(data_loader, model, criterion, optimizer, epoch, args):
    model.train()
    
    y_trues, y_preds, y_probs = [], [], []
    total_loss = 0.0
    for batch in data_loader:
        if len(batch) == 2:
            data, target = batch
            mask = None
        elif len(batch) == 3:
            data, target, mask = batch
        data = data.to(args.device)
        target = target.to(args.device)
        mask = mask.to(args.device) if mask is not None else None

        target_aug = target.clone()
        if hasattr(args, 'augment') and args.augment is not None:
            data, target_aug = args.augment(data, target_aug)
            
        # compute output
        if mask is None:
            output = model(data)
        else:
            output = model(data, mask)
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        if hasattr(args, 'augment') and args.augment is not None:
            loss = criterion(output, target_aug, True)
        else:
            loss = criterion(output, target_aug)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        logits = output[0] if isinstance(output, tuple) else output
        pred = torch.argmax(logits, dim=-1)
        prob = torch.nn.functional.softmax(output, dim=-1)
        y_trues.append(target.detach().cpu())
        y_preds.append(pred.detach().cpu())
        y_probs.append(prob.detach().cpu())

    y_trues = torch.concatenate(y_trues).numpy()
    y_preds = torch.concatenate(y_preds).numpy()
    y_probs = torch.cat(y_probs, dim=0).numpy()

    accu = 100 * np.mean(y_trues == y_preds)
    
    return total_loss/len(data_loader), accu, y_trues, y_probs


def evaluate(data_loader, model, criterion, epoch, args, trialwise=False, lengths=None):
    model.eval()

    y_trues, y_preds, y_probs = [], [], []
    total_loss = 0.0
    for batch in data_loader:
        if len(batch) == 2:
            data, target = batch
            mask = None
        elif len(batch) == 3:
            data, target, mask = batch
        data = data.to(args.device)
        target = target.to(args.device)
        mask = mask.to(args.device) if mask is not None else None
        # compute output
        if mask is None:
            output = model(data)
        else:
            output = model(data, mask)
        if isinstance(output, (list, tuple)):
            output = output[0]
        
        if hasattr(args, 'augment') and args.augment is not None:
            loss = criterion(output, target, False)
        else:
            loss = criterion(output, target)

        total_loss += loss.item()
        logits = output[0] if isinstance(output, tuple) else output
        pred = torch.argmax(logits, dim=-1)
        prob = torch.nn.functional.softmax(output, dim=-1)
        y_trues.append(target.detach().cpu())
        y_preds.append(pred.detach().cpu())
        y_probs.append(prob.detach().cpu())

    y_trues = torch.concatenate(y_trues).numpy()
    y_preds = torch.concatenate(y_preds).numpy()
    y_probs = torch.cat(y_probs, dim=0).numpy()

    if trialwise:
        y_trues, y_preds = trialwise_voting(y_trues, y_preds, lengths)
        # y_probs is not available in trialwise mode
        y_probs = np.zeros((len(y_trues), 2)) 
        for i in range(len(y_trues)):
            y_probs[i, y_preds[i]] = 1.0

    accu = 100 * np.mean(y_trues == y_preds)
    
    return total_loss/len(data_loader), accu, y_trues, y_probs


def trialwise_voting(ytrue, ypred, lengths):
    assert len(ytrue) == sum(lengths), "length mismatch"
    lengths_cumsum = np.cumsum(lengths)
    ytrue = np.split(ytrue, lengths_cumsum[:-1])
    ypred = np.split(ypred, lengths_cumsum[:-1])
    ytrue_trial = [ytrue[i][0] for i in range(len(ytrue))]
    ypred_trial = []
    for trial in ypred:
        index_0 = np.where(trial == 0)[0]
        index_1 = np.where(trial == 1)[0]
        if len(index_1) >= len(index_0):
            label = 1
        else:
            label = 0
        ypred_trial.append(label)
    ypred_trial = np.array(ypred_trial)
    return ytrue_trial, ypred_trial


class MILSequenceCollator(object):
    def __init__(self, pad_value=0, padding_side='right'):
        self.pad_value = pad_value
        self.padding_side = padding_side

    def __call__(self, batch):
        batch_x, batch_y = zip(*batch)

        batch_size = len(batch_x)
        xdim = batch_x[0].shape[1:]
        lengths = [x.size(0) for x in batch_x] # length of each sequence
        max_length = max(lengths) # max length of all sequences
        
        sequences = self.pad_value * torch.ones(
            [batch_size, max_length, *xdim], dtype=batch_x[0].dtype)
        masks = torch.zeros(batch_size, max_length)
        labels = torch.LongTensor(batch_y)

        for k in range(batch_size):
            if self.padding_side == 'right':
                sequences[k, :lengths[k]] = batch_x[k]
                masks[k, :lengths[k]] = 1
            elif self.padding_side == 'left':
                sequences[k, -lengths[k]:] = batch_x[k]
                masks[k, -lengths[k]:] = 1
            else:
                raise ValueError("Padding side should be either left or right")
            
        masks = masks.bool()

        return sequences, labels, masks
    

def collate_fn_min(data): # 按最小长度截断
    data.sort(key=lambda x: len(x[0]), reverse=False)  # 按照数据长度升序排序
    data_list = []
    label_list = []
    min_len = len(data[0][0]) # 最短的数据长度 
    for i in range(len(data)): #
        data_list.append(data[i][0][:min_len].numpy())
        label_list.append(data[i][1].numpy())
    data_tensor = torch.FloatTensor(np.array(data_list))
    label_tensor = torch.LongTensor(np.array(label_list))
    data_copy = (data_tensor, label_tensor)
    return data_copy


def collate_fn_max(data): # 按最大长度补零
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_list = []
    label_list = []
    max_len = len(data[0][0]) # 最长的数据长度 
    for i in range(len(data)): #
        len_i = len(data[i][0])
        zeropad = np.zeros((max_len-len_i, *data[i][0].shape[1:]))
        data_padded = np.concatenate((data[i][0].numpy(), zeropad), axis=0)
        data_list.append(data_padded)
        label_list.append(data[i][1].numpy())
    data_tensor = torch.FloatTensor(np.array(data_list))
    label_tensor = torch.LongTensor(np.array(label_list))
    data_copy = (data_tensor, label_tensor)
    return data_copy
