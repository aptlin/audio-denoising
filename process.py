#!/usr/bin/env python

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from audio_denoising.data.loader import SpectogramDataset
from audio_denoising.model.rdn import ResidualDenseNetwork as Model
from pytorch_ssim import ssim

SOURCE_DIR = os.environ["SOURCE_DIR"] if "SOURCE_DIR" in os.environ else "/dataset"
TARGET_DIR = os.environ["TARGET_DIR"] if "TARGET_DIR" in os.environ else "/results"
WEIGHTS = (
    os.environ["WEIGHTS"]
    if "WEIGHTS" in os.environ
    else "./weights/audio_denoising_psnr_65.1736_epoch_15_D_20_C_6_G_16_G0_16.pth"
)


def process(args, verbose=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(args).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    dataset = SpectogramDataset(args.source_dir, extension=args.extension)

    files = dataset.files
    if verbose:
        print("Processing {} files.".format(len(files)))

    filenames = []
    results = []
    denoised_filenames = []

    target_dir = Path(args.target_dir)
    Path.mkdir(target_dir, exist_ok=True, parents=True)

    for file_idx in tqdm(range(len(dataset)), unit="files"):
        filename = files[file_idx]
        filenames.append(filename)

        img = dataset[file_idx].unsqueeze(0)
        img = img.to(device, dtype=torch.float)
        noise = model(img).to("cpu")
        img = img.to("cpu")

        if ssim(img - noise, img).data.item() >= args.threshold:
            results.append("clean")
            denoised_filenames.append("")
        else:
            results.append("noisy")

            denoised_filename = target_dir / (
                args.denoised_subdir + filename.split(str(Path(args.source_dir)))[1]
            )
            Path.mkdir(denoised_filename.parent, exist_ok=True, parents=True)
            denoised_filenames.append(str(denoised_filename))

            clean_img = img - noise

            np.save(denoised_filename, clean_img.to("cpu").detach().numpy())

    results_df = pd.DataFrame(
        {"file_name": filenames, "result": results, "denoised_file": denoised_filenames}
    )

    results_df.to_csv(target_dir / "results.csv", index=False)

    return results_df


def get_arg_parser():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument("--growth-rate", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--num-blocks", type=int, default=20)
    parser.add_argument("--num-channels", type=int, default=1)
    parser.add_argument("--num-features", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=6)

    # setup
    parser.add_argument(
        "--extension", type=str, default="npy",
    )
    parser.add_argument(
        "--source-dir", type=str, default=SOURCE_DIR,
    )
    parser.add_argument(
        "--target-dir", type=str, default=TARGET_DIR,
    )

    parser.add_argument(
        "--denoised-subdir", type=str, default="denoised",
    )
    parser.add_argument(
        "--weights", type=str, default=WEIGHTS,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.77,  # see ./demo.ipynb for the justification
    )

    return parser


if __name__ == "__main__":
    args = get_arg_parser().parse_args()
    print("*" * 80)
    print(
        "Starting to process the {} files in the '{}' directory".format(
            args.extension, args.source_dir
        )
    )
    print("*" * 80)
    print("Args")
    print("-" * 80)
    for key, value in vars(args).items():
        print("\t{}:\t{}".format(key, value))
    print("*" * 80)
    process(args)
    print("*" * 80)
    print("Done!")
    sys.stdout.flush()
