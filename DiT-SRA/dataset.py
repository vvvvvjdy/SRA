import os
import json

import torch
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
import PIL.Image
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    def __init__(self, data_dir,num_classes=1000):
        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}

        self.num_classes = num_classes
        self.features_dir = os.path.join(data_dir, 'vae-sd')


        # features
        self._feature_fnames = {
            os.path.relpath(os.path.join(root, fname), start=self.features_dir)
            for root, _dirs, files in os.walk(self.features_dir) for fname in files
            }
        self.feature_fnames = sorted(
            fname for fname in self._feature_fnames if self._file_ext(fname) in supported_ext
            )
        # labels
        fname = 'dataset.json'
        with open(os.path.join(self.features_dir, fname), 'rb') as f:
            labels = json.load(f)['labels']
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self.feature_fnames]
        labels = np.array(labels)
        self.labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])


    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        feature_fname = self.feature_fnames[idx]
        features = np.load(os.path.join(self.features_dir, feature_fname))
        return torch.from_numpy(features), torch.tensor(self.labels[idx])


