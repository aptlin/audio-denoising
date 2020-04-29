import functools
import numpy as np
from dataclasses import dataclass
import torch
from glob import glob
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from ipython_secrets import get_secret
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset


@dataclass(frozen=True)
class SpectogramPairedDataset(Dataset):
    dirname: str
    extension: str = "npy"
    clean: str = "clean"
    noisy: str = "noisy"

    @property
    @functools.lru_cache(1)
    def files(self):
        clean_files = glob(
            f"{self.dirname}/{self.clean}/**/*.{self.extension}", recursive=True
        )
        noisy_files = [
            f"{self.dirname}/{self.noisy}{f.split(self.clean)[1]}" for f in clean_files
        ]
        return list(zip(clean_files, noisy_files))

    def __len__(self):
        return len(self.files)

    def _load_raw(self, filename: str):
        return torch.from_numpy(np.load(filename)).unsqueeze(2)

    def __getitem__(self, idx):
        clean_file, noisy_file = self.files[idx]

        clean_mel = self._load_raw(clean_file)
        noisy_mel = self._load_raw(noisy_file)

        noise = noisy_mel - clean_mel

        return clean_mel, noise


def load(dirname, batch_size=32, extension="npy", clean="clean", noisy="noisy"):
    dataset = SpectogramPairedDataset(
        dirname, extension=extension, clean=clean, noisy=noisy
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader
