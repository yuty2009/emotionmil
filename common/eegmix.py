# Copy from https://github.com/timeseriesAI/timeseriesAI1/blob/master/fastai_timeseries/exp/nb_TSCallbacks.py
import torch
import numpy as np


""" We assume a (n_batch x timepoints x channels) or (B, T, C) EEG epoch as input here """
    
class CutMix(object):
    """ CutMix augmentation for EEG data 
    Args:
        alpha (float): parameter for beta distribution
        method (str): 'cutout', 'cutmix' or 'mixup'
    """
    def __init__(self, alpha=1.0, method='cutmix'):
        self.alpha = alpha
        self.alpha2 = 0.0
        self.method = method

    def __call__(self, input, target=None):
        λ = np.random.beta(self.alpha, self.alpha)
        λ = max(λ, 1- λ)
        inshape = input.size()
        new_input = input.clone()
        idx = torch.randperm(input.size(0)).to(input.device)
        if self.method == 'cutout': # cutout https://arxiv.org/abs/1708.04552
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(inshape, λ)
            new_input[..., bby1:bby2, bbx1:bbx2] = 0
        elif self.method == 'cutmix': # cutmix https://arxiv.org/pdf/1905.04899
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(inshape, λ)
            new_input[..., bby1:bby2, bbx1:bbx2] = input[idx][..., bby1:bby2, bbx1:bbx2]
            λ = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inshape[-1] * inshape[-2]))
        elif self.method == 'mixup':
            if self.alpha2 == 0: self.alpha2 = self.alpha
            λ2 = np.random.beta(self.alpha2, self.alpha2)
            λ2 = λ + (1 - λ) * λ2
            λ = λ / λ2
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(inshape, λ)
            new_input[..., bby1:bby2, bbx1:bbx2] = λ2 * input[..., bby1:bby2, bbx1:bbx2] + \
                (1 - λ2) * input[idx][..., bby1:bby2, bbx1:bbx2]
            λ = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inshape[-1] * inshape[-2]))
            λ = λ * λ2
        λ = input.new([λ])
        if target is not None:
            if len(target.shape) == 1:
                # binary classification with integer target
                # output target is a 3-tuple: target, target[idx], λ
                new_target = torch.cat([
                    target.unsqueeze(1).float(), 
                    target[idx].unsqueeze(1).float(), 
                    λ.repeat(inshape[0]).unsqueeze(1).float()
                ], 1)
            else:
                # multi-class classification with one-hot encoding target
                # output target is soft label
                λ = λ.unsqueeze(1).float()
                new_target = target.float() * λ + target[idx].float() * (1-λ)
        return new_input, new_target
    
    def rand_bbox(self, inshpae, λ=0.5):
        '''lambd is always between .5 and 1'''
        W, H = inshpae[-1], inshpae[-2]
        cut_rat = np.sqrt(1. - λ) # 0. - .707
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        if len(inshpae) == 4:
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bby2 = np.clip(cy + cut_h // 2, 0, H)
        else:
            bby1 = 0
            bby2 = inshpae[1]
        return bbx1, bby1, bbx2, bby2
    

class MixupLoss(torch.nn.Module):
    def __init__(self, criterion):
        super(MixupLoss, self).__init__()
        self.criterion = criterion # reduction must be 'none'

    def forward(self, output, target, train=False):
        if train:
            # binary classification with integer target
            target1, target2, λ = target[:, 0], target[:, 1], target[:, 2]
            loss = λ * self.criterion(output, target1.long()) + (1 - λ) * self.criterion(output, target2.long())
            return loss.mean()
        else:
            return self.criterion(output, target).mean()
    

class NaiveMix(object):
    def __init__(self, n_windows=4):
        self.n_windows = n_windows

    def __call__(self, data, labels=None, n_augments=1000):
        if labels is None:
            n_classes = 1
            labels = np.zeros(data.shape[0])
        else:
            n_classes = len(np.unique(labels))
        n_timepoints = data.shape[1]
        n_winlen = np.ceil(n_timepoints / self.n_windows).astype(int)
        data_aug = []
        labels_aug = []
        for c in range(n_classes):
            idx = np.where(labels == c)[0]
            data_cls = data[idx]
            data_new = np.zeros((n_augments, *data.shape[1:]))
            for ri in range(n_augments):
                for rj in range(self.n_windows):
                    rand_idx = np.random.choice(data_cls.shape[0], 1, replace=False)
                    data_new[ri, rj*n_winlen:(rj+1)*n_winlen, :] = data_cls[rand_idx, rj*n_winlen:(rj+1)*n_winlen, :]
            labels_new = np.ones(n_augments) * c
            data_aug.append(data_new)
            labels_aug.append(labels_new)
        data_aug = np.concatenate(data_aug, axis=0)
        labels_aug = np.concatenate(labels_aug, axis=0)
        # shuffle
        idx = np.random.permutation(data_aug.shape[0])
        data_aug = data_aug[idx]
        labels_aug = labels_aug[idx]
        return data_aug, labels_aug
    

if __name__ == "__main__":

    x = torch.randn(10, 128, 30)
    y = torch.randint(0, 2, (10,))
    y_onehot = torch.nn.functional.one_hot(y, num_classes=2).float()

    cutmix = CutMix(alpha=1.0, method='cutout')
    x_new, y_new = cutmix(x, y_onehot)
    print(x_new.shape, y_new.shape)
    print(y_new)
