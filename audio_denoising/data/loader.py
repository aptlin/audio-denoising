import functools
from dataclasses import dataclass
from glob import glob

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision.transforms import RandomCrop


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
        return torch.from_numpy(np.load(filename)).unsqueeze(0)

    def __getitem__(self, idx):
        clean_file, noisy_file = self.files[idx]

        clean_mel = self._load_raw(clean_file)
        noisy_mel = self._load_raw(noisy_file)

        return clean_mel, noisy_mel


def _group_random_crop(img_group, crop_height, crop_width):
    if len(img_group) == 0:
        return ()
    else:
        _, height, width = img_group[0].shape
        random_height = (
            0 if height == crop_height else np.random.choice(height - crop_height)
        )
        random_width = (
            0 if width == crop_width else np.random.choice(width - crop_width)
        )
        return tuple(
            image[
                :,
                random_height : random_height + crop_height,
                random_width : random_width + crop_width,
            ]
            for image in img_group
        )


def _collate_with_cropping(batch):
    crop_height = min([item[0].shape[1] for item in batch])
    crop_width = min([item[0].shape[2] for item in batch])
    cropped_img_groups = [
        _group_random_crop(img_group, crop_height, crop_width) for img_group in batch
    ]
    return tuple(map(torch.stack, zip(*cropped_img_groups)))


def load(
    dirname,
    batch_size=32,
    extension="npy",
    clean="clean",
    noisy="noisy",
    collate_fn=_collate_with_cropping,
):
    dataset = SpectogramPairedDataset(
        dirname, extension=extension, clean=clean, noisy=noisy
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return loader
