import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import argparse
import os
from pathlib import Path
from math import floor


def get_loaders(folder_path: Path, batch_size: int, test_size: float) -> (DataLoader, DataLoader):
    """Loads the dataset, performs a train/test split and returns the dataloaders for train and test."""

    if not os.path.exists(folder_path):
        print("The given folder path is invalid.")
        exit(-1)

    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    ds = datasets.ImageFolder(folder_path, transform=transform)

    test_len = floor(test_size * len(ds))
    train_len = len(ds) - test_len

    train_ds, test_ds = random_split(ds, (train_len, test_len))

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", help="The path to the folder where the dataset is.", type=Path)
    parser.add_argument("--bs", help="The batch size for the dataloader.", type=int, default=8)
    parser.add_argument("--test_size", help="The size of the test dataset (should be <1)", type=float, default=0.2)

    args = parser.parse_args()

    train_dl, test_dl = get_loaders(args.dataset_folder, args.bs, args.test_size)


if __name__ == '__main__':
    main()
