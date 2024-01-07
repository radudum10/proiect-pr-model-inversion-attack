## Model inversion attack - AT&T Database of Faces Classifier

### Prerequisites:
* Python
* Cuda

If you already have PyTorch installed, just simply run the train.py script.

If not, please create a virtual environment, and activate it:
```shell
python -m venv <venv_name>
source <venv_name>/bin/activate
```
And install the requirements:
```shell
python -m pip install -r requirements.txt
```

### Train

#### Usage:
```shell
python train.py dataset/
```
The batch size and the size of test can also be customized like this:
```shell
python train.py dataset --bs=8 --test_size=0.3
```

#### Details about training
* The data is loaded with [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
with a simple transformation of grayscale images to PyTorch tensor, then it is
splitted into train and test and loaded into [DataLoader](https://pytorch.org/docs/stable/data.html).
* The model used is extremely simple, it has just a linear layer.
* The loss function is [CrossEntropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
and the optimizer is [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html).

### Attack

#### Usage:
```shell
python attack.py --model_file=<model.pth>
``` 

The output folder can be customized with `--output_folder` option.

#### Details about attacking
* It loads the model using a dict state saved in training.
* For each target (there are 40 of them), the attack is performed.
* We declare a dummy tensor (initialised with zeros).
* We declare an optimizer, but instead of the model's params we have the dummy tensor
as param.
* It makes a forward step and then computes the loss with the current target.
(e.g. we are on target 3, then the loss is computed with the result of the
forward step and the label 3).
* The 
* The loop can stop on the following cases:
    1. The current loss is less than the minimum allowed loss (1e-5).
    2. The lost didn't decrease for the last `max_loss_not_decreasing` steps.
    3. It performed `attack_iters` iterations.
* The 'dummy tensor' is now an image and it is saved to disk.

### Credits to AT&T Laboratories Cambridge for the dataset.
