## Model inversion attack - AT&T Database of Faces Classifier

### Train

## Prerequisites:
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

## Usage:
```shell
python train.py dataset/
```
The batch size and the size of test can also be customized like this:
```shell
python train.py dataset --bs=8 --test_size=0.3
```

## Details about training
* The data is loaded with [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
with a simple transformation of grayscale images to PyTorch tensor, then it is
splitted into train and test and loaded into [DataLoader](https://pytorch.org/docs/stable/data.html).
* The model used is extremely simple, it has just a linear layer.
* The loss function is [CrossEntropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
and the optimizer is [SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html).
