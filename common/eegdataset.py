
import torch
import numpy as np


class ToTensor(object):
    """ 
    Turn a (timepoints x channels) or (T, C) epoch into 
    a (depth x timepoints x channels) or (D, T, C) image for torch.nn.Convnd
    """
    def __init__(self, expand_dim=True) -> None:
        self.expand_dim = expand_dim

    def __call__(self, epoch, target=None):
        if isinstance(epoch, np.ndarray):
            epoch = torch.FloatTensor(epoch.copy())
        if self.expand_dim:
            epoch = epoch.unsqueeze(-3)
        if target is not None:
            return epoch, torch.LongTensor(target)
        return epoch


class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, epochs, targets, transforms=None):
        super(EEGDataset, self).__init__()
        if transforms == None:
            self.epochs = epochs
        else:
            self.epochs = [transforms(epoch) for epoch in epochs]
        self.targets = torch.LongTensor(targets)

    def __getitem__(self, idx):
        return self.epochs[idx], self.targets[idx]

    def __len__(self):
        return len(self.targets)
    