import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import pickle
import librosa
import librosa.display

def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


def repeat_padding(spec, ref_len):
    mul = int(np.ceil(ref_len / spec.shape[1]))
    spec = spec.repeat(1, mul)[:, :ref_len]
    return spec


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir, lfcc_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.lfcc_dir = lfcc_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # key = key.split(".")[0]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.wav"))

        D = np.abs(librosa.stft(X))
        P = np.angle(librosa.stft(X))
        perturbation_values = [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        i=0
        perturbed_P = P + np.random.uniform(-perturbation_values[i] / 2, perturbation_values[i] / 2, size=P.shape)
        X = librosa.istft(D * np.exp(1j * perturbed_P))

        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)

        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        
        with open(self.lfcc_dir + "train/"+key+"LFCC.pkl", 'rb') as feature_handle:
            feat_mat = pickle.load(feature_handle)
        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > 750:
            startp = np.random.randint(this_feat_len-750)
            feat_mat = feat_mat[:, startp:startp+750]
            
        if this_feat_len < 750:
            feat_mat = repeat_padding(feat_mat, 750)

        return x_inp, feat_mat, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir, lfcc_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.lfcc_dir = lfcc_dir

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # key = key.split(".")[0]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.wav"))

        D = np.abs(librosa.stft(X))
        P = np.angle(librosa.stft(X))
        perturbation_values = [np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
        i=3
        perturbed_P = P + np.random.uniform(-perturbation_values[i] / 2, perturbation_values[i] / 2, size=P.shape)
        X = librosa.istft(D * np.exp(1j * perturbed_P))

        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        
        if "valid" in str(self.base_dir):
            file = self.lfcc_dir + "dev/"+key+"LFCC.pkl"
        else:
            file = self.lfcc_dir + "eval/"+key+"LFCC.pkl"
            
        with open(file, 'rb') as feature_handle:
            feat_mat = pickle.load(feature_handle)
        feat_mat = torch.from_numpy(feat_mat)
        this_feat_len = feat_mat.shape[1]
        if this_feat_len > 750:
            feat_mat = feat_mat[:, :750]
            
        if this_feat_len < 750:
            feat_mat = repeat_padding(feat_mat, 750)
        
        return x_inp, feat_mat, key
