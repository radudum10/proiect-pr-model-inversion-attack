import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import argparse
import os
from pathlib import Path
from math import floor
import numpy as np


config = {
    'lr': 1e-3,
    'momentum': 0.8,
    'weight_decay': 1e-5,
    'epochs': 10
}

# The AT&T Database of Faces has 92x112 images of 40 people.
width = 92
height = 112
num_classes = 40


class SimpleLinearClassifier(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleLinearClassifier, self).__init__()

        self.l1 = nn.Linear(in_features=in_features, out_features=out_features)

    
    def forward(self, x): # b, height x width
        return self.l1(x) # b, num_classes


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


def eval(model: SimpleLinearClassifier, loss: nn.CrossEntropyLoss, dl: DataLoader, set_name: str):
    losses = []
    t, f = 0, 0

    for Xs, gt in dl:
        Xs, gt = Xs.cuda(), gt.cuda()
        Xs = Xs.reshape(Xs.size(0), -1) # b, height x width

        # forward
        with torch.no_grad():
            logits = model(Xs)
            J = loss(logits, gt)
            losses.append(J.item())

        yc = gt.cpu()
        y_hat = logits.detach().argmax(dim=1).cpu()

        t += int(yc.eq(y_hat).sum())
        f += int(yc.ne(y_hat).sum())

    acc = t / (t + f)
    print("=" * 5 + f"{set_name} evaluation" + "=" * 5)
    print(f"accuracy={acc * 100}%")
    print(f"loss={np.mean(losses)}")

    return acc


def train(model: SimpleLinearClassifier, loss: nn.CrossEntropyLoss, optimizer: optim.SGD,
          train_dl: DataLoader, test_dl: DataLoader):

    best_acc = 0.0
    
    for epoch in range(config['epochs']):
        model.train()
        losses = []

        for Xs, ys in train_dl:
            Xs, ys = Xs.cuda(), ys.cuda()
            Xs = Xs.reshape(Xs.size(0), -1) # b, height x width

            # cleaning the gradients
            model.zero_grad()

            # forward
            logits = model(Xs)

            # compute objective function
            J = loss(logits, ys)

            # compute the partial derivatives dJ/dparams
            J.backward()

            # step in the opposite direction of the gradinet
            optimizer.step()

            losses.append(J.item())
        
        print("=" * 10 + f"epoch: {epoch}" + "=" * 10)
        print(f"training_loss={np.mean(losses)}")
        train_acc = eval(model, loss, train_dl, 'train')
        test_acc = eval(model, loss, test_dl, 'test')

        if test_acc > 0.8 and test_acc > best_acc:
            print("Saving best model")
            best_acc = test_acc
            torch.save(
                model.state_dict(),
                f"best_model_train_acc{train_acc}_test_acc{test_acc}_epoch{epoch}.pth"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", help="The path to the folder where the dataset is.", type=Path)
    parser.add_argument("--bs", help="The batch size for the dataloader.", type=int, default=1)
    parser.add_argument("--test_size", help="The size of the test dataset (should be <1)", type=float, default=0.2)

    args = parser.parse_args()

    train_dl, test_dl = get_loaders(args.dataset_folder, args.bs, args.test_size)

    model = SimpleLinearClassifier(width * height, num_classes)
    model = model.cuda()
    print(model)

    params = model.parameters()
    optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], weight_decay=config['weight_decay'])
    loss = nn.CrossEntropyLoss(weight=torch.Tensor([1] * num_classes).cuda())

    train(
        model=model,
        loss=loss,
        optimizer=optimizer,
        train_dl=train_dl,
        test_dl=test_dl
    )


if __name__ == '__main__':
    main()
