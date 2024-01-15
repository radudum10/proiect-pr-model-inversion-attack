import torch
from torch import nn
from train import SimpleLinearClassifier, config, width, height, num_classes
import argparse
from pathlib import Path
from torchvision.utils import save_image
import os
from tqdm import tqdm


attack_iters = 5000
max_loss_not_decreasing = 1000
min_loss = 1e-4


def load_model(model_path: Path) -> SimpleLinearClassifier:
    model = SimpleLinearClassifier(width * height, num_classes)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    model.eval()

    print(model)

    return model


def attack(model: SimpleLinearClassifier, target: int, output_folder: Path):
    dummy_input = torch.zeros(1, width * height).cuda()
    dummy_input.requires_grad = True

    optimizer = torch.optim.SGD([dummy_input], lr=config['lr'], momentum=config['momentum'])

    last_loss = float('inf')
    loss_didnt_decrease = 0

    for _ in range(attack_iters):
        logits = model.forward(dummy_input)
        
        loss = nn.functional.cross_entropy(logits, torch.tensor([target]).cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        dummy_input = (dummy_input - dummy_input.min()) / (dummy_input.max() - dummy_input.min())
        dummy_input = torch.clamp(dummy_input.detach(), 0, 1)

        if loss < min_loss:
            break

        if loss >= last_loss:
            loss_didnt_decrease += 1
            
            if loss_didnt_decrease > max_loss_not_decreasing:
                break
        else:
            loss_didnt_decrease = 0
        
        last_loss = loss

    logits = model.forward(dummy_input)
    y_hat = logits.detach().argmax(dim=1).cpu()

    save_image(
        dummy_input.detach().reshape(height, width),
        os.path.join(output_folder, f"{y_hat.item()}.png")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        help="The state dict of the model.",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output_folder",
        help="The folder to output the image to.",
        type=Path,
        default=Path("output/")
    )

    args = parser.parse_args()
    model = load_model(args.model_file)

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)
    

    for i in tqdm(range(num_classes), desc="attacking"):
        attack(model, i, args.output_folder)


if __name__ == '__main__':
    main()
