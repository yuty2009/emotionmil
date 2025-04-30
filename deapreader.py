
import os
import glob
import pickle
import numpy as np
from scipy.stats import zscore
import sys; sys.path.append(os.getcwd())
from common.timefreq import bandpower, differentialentropy


tasks = {'arousal': 0, 'valence': 1, 'dominance': 2, 'liking': 3}

CHANLOC_RAW = [
    'FP1', 'AF3', 'F3' , 'F7' , 'FC5', 'FC1', 'C3' , 'T7' , 'CP5', 'CP1', 'P3' , 'P7' , 'PO3',
    'O1' , 'OZ' , 'PZ' , 'FP2', 'AF4', 'FZ' , 'F4' , 'F8' , 'FC6', 'FC2', 'CZ' , 'C4' , 'T8' , 'CP6',
    'CP2', 'P4' , 'P8' , 'PO4', 'O2'
]

CHANLOC_TS = [
    'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1',
    'FP2', 'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
]

CHANLOC_3D = [
    '-', '-', '-', 'FP1', '-', 'FP2', '-', '-', '-',
    '-', '-', '-', 'AF3', '-', 'AF4', '-', '-', '-',
    'F7', '-', 'F3', '-', 'FZ', '-', 'F4', '-', 'F8',
    '-', 'FC5', '-', 'FC1', '-', 'FC2', '-', 'FC6', '-',
    'T7', '-', 'C3', '-', 'CZ', '-', 'C4', '-', 'T8',
    '-', 'CP5', '-', 'CP1', '-', 'CP2', '-', 'CP6', '-',
    'P7', '-', 'P3', '-', 'PZ', '-', 'P4', '-', 'P8',
    '-', '-', '-', 'PO3', '-', 'PO4', '-', '-', '-',
    '-', '-', '-', 'O1', 'OZ', 'O2', '-', '-', '-'
]


def load_eegdata(filepath, chanset, raw_labels=False, channel_last=True, remove_baseline=True):
    # Load data from .dat file and return data and labels
    # data: [num_trials, num_samples, num_channels]
    # labels: [num_trials, num_tasks]
    f = open(filepath, 'rb')  # Read the file in Binary mode
    x = pickle.load(f, encoding='latin1')
    data, target = x['data'], x['labels']

    fs = 128
    if not raw_labels:
        target[target<=5] = 0
        target[target>5] = 1

    data_baseline = data[:, chanset, :3*fs]
    data_signal = data[:, chanset, 3*fs:]

    if remove_baseline:
        data_signal = data_signal - np.mean(data_baseline, axis=-1, keepdims=True)

    if channel_last:
        data_signal = np.transpose(data_signal, [0, 2, 1])
        data_baseline = np.transpose(data_baseline, [0, 2, 1])

    return data_signal, target


def split_eegdata(data, target, window, stride):
    # Split data into segments with window size and stride
    data_extracted = []
    labels_extracted = []
    data_lengths = []
    for i in range(len(data)):
        start = 0
        data_trial = []
        labels_trial = []
        data_used = data[i]
        while start + window <= data_used.shape[-2]:
            data_seg = data_used[start:start+window, :] # [t, c]
            data_trial.append(data_seg)
            labels_trial.append(target[i])
            start += stride
        data_trial = np.array(data_trial)
        labels_trial = np.array(labels_trial)
        data_extracted.append(data_trial)
        labels_extracted.append(labels_trial)
        data_lengths.append(data_trial.shape[0])
    data_extracted = np.concatenate(data_extracted, axis=0)
    labels_extracted = np.concatenate(labels_extracted, axis=0)
    data_lengths = np.array(data_lengths, dtype=np.int64)
    return data_extracted, labels_extracted, data_lengths


def reorder_channels(data, chanloc):
    # Reorder channels according to chanloc
    # data: [..., num_channels]
    # chanloc: [num_selected_channels]
    data_reorder = np.zeros((*data.shape[:-1], len(chanloc)))
    for i in range(len(chanloc)):
        if chanloc[i] != '-':
            idx = CHANLOC_RAW.index(chanloc[i])
            data_reorder[..., i] = data[..., idx]
    return data_reorder


def data_to_3d(data, nc_row=9, nc_col=9):
    data_3d = np.reshape(data, [*data.shape[:-1], nc_row, nc_col])
    return data_3d


def extract_differentialentropy(data, fbands=[4, 8, 14, 31, 49], fs=128):
    in_shape = data.shape
    data = np.reshape(data, (-1, in_shape[-2], in_shape[-1]))
    num_examples, num_samples, num_channels = data.shape
    num_features = len(fbands) - 1
    features = np.zeros((num_examples, num_features, num_channels))
    for i in range(num_examples):
        for j in range(num_channels):
            signal_ij = data[i, :, j]
            features[i, :, j] = differentialentropy(signal_ij, fs, fbands)
    features = np.reshape(features, (*in_shape[:-2], num_features, num_channels))
    return features


def extract_bandpower(data, fbands=[4, 8, 14, 31, 49], fs=128):
    in_shape = data.shape
    data = np.reshape(data, (-1, in_shape[-2], in_shape[-1]))
    num_examples, num_samples, num_channels = data.shape
    num_features = len(fbands) - 1
    features = np.zeros((num_examples, num_features, num_channels))
    for i in range(num_examples):
        for j in range(num_channels):
            signal_ij = data[i, :, j]
            features[i, :, j] = bandpower(signal_ij, fs, fbands)
    features = np.reshape(features, (*in_shape[:-2], num_features, num_channels))
    return features


def preprocess_subject(
        data, labels, window, stride, feature=None, chanloc=None, to3d=False, **kwargs
    ):
    # Split data into segments with window size and stride
    data, labels, lengths = split_eegdata(data, labels, window, stride)

    if feature == 'bp':
        data = extract_bandpower(data, fbands=kwargs.get('fbands', [4, 8, 14, 31, 49]))
    elif feature == 'de':
        data = extract_differentialentropy(data, fbands=kwargs.get('fbands', [4, 8, 14, 31, 49]))

    if chanloc is not None:
        data = reorder_channels(data, chanloc)

    if to3d:
        data = data_to_3d(data, 9, 9)

    return data, labels, lengths


def preprocess_dataset(
        data, labels, window, stride, feature=None, chanloc=None, to3d=False
    ):
    for i in range(len(data)):
        print(f"Preprocessing data of subject {i} ...")
        data[i], labels[i] = preprocess_subject(
            data[i], labels[i], window, stride, feature, chanloc, to3d
        )
    return data, labels


def load_dataset_preprocessed(datapath):
    npzfiles = glob.glob(os.path.join(datapath, "*.npz"))
    npzfiles.sort()
    subjects = [os.path.basename(npz_f)[:-4] for npz_f in npzfiles]

    data = []
    labels = []
    lengths = []
    for npz_f in npzfiles:
        print("Loading {} ...".format(npz_f))
        npz_data = np.load(npz_f, allow_pickle=True)
        tmp_data = npz_data['data']
        tmp_labels = npz_data['labels']
        tmp_lengths = npz_data['lengths']
        tmp_data = tmp_data.astype(np.float32)
        tmp_labels = tmp_labels.astype(np.int32)
        tmp_lengths = tmp_lengths.astype(np.int64)
        data.append(tmp_data)
        labels.append(tmp_labels)
        lengths.append(tmp_lengths)
    return data, labels, lengths, subjects


if __name__ == '__main__':

    # datapath = 'e:/eegdata/emotion/deap/'
    datapath = '/home/yuty2009/data/eegdata/emotion/deap/'

    fs = 128
    twin = 10.0
    eegchan = np.arange(32)
    # eegchan = [1,2,3,4,6,11,13,17,19,20,21,25,29,31] #14 Channels chosen to fit Emotiv Epoch+
    window = int(twin * fs)
    stride = int(twin * fs)
    # fbands = [4, 8, 14, 31, 49] # theta, alpha, beta, gamma
    fbands = [4, 8, 12, 16, 25, 45] # 5 bands

    savepath = datapath + f"processed_{int(twin)}s/"
    # savepath = datapath + f"processed_ts_{int(twin)}s/"
    # savepath = datapath + f"features_de_{int(twin)}s/"
    if not os.path.isdir(savepath):
        os.mkdir(savepath)

    for filepath in glob.glob(datapath + "data_preprocessed_python/*.dat"):
        subname = os.path.basename(filepath).split('.')[0]
        print('Load and extract feature for %s' % filepath)
        data, labels = load_eegdata(filepath, eegchan, raw_labels=False)
        # np.savez(datapath + f"raw/{subname}.npz", data=data, labels=labels)
        data, labels, lengths = preprocess_subject(
            data, labels, window, stride,
        )
        # data, labels, lengths = preprocess_subject(
        #     data, labels, window, stride,
        #     chanloc=CHANLOC_TS,
        # )
        # data, labels, lengths = preprocess_subject(
        #     data, labels, window, stride,
        #     chanloc=CHANLOC_3D, to3d=True,
        # )
        # data, labels, lengths = preprocess_subject(
        #     data, labels, window, stride,
        #     feature='de', fbands=fbands, # chanloc=CHANLOC_3D, to3d=False
        # )
        np.savez(f"{savepath}{subname}.npz", data=data, labels=labels, lengths=lengths)
