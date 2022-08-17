<p align="center">
    <img src="assets/Pytorch%20Boiler.png">
</p>

[![Python 3 version](https://img.shields.io/badge/python-%3E%3D3.6-blue)](https://www.python.org/downloads/release/python-360/)
[![Pytorch version](https://img.shields.io/badge/pytorch-%3E%3D%201.4.0-informational)](https://pytorch.org/get-started/previous-versions/)


[![Code Size](https://img.shields.io/github/languages/code-size/nmakes/pytorch_boiler)](https://github.com/nmakes/pytorch_boiler/)

[![LICENCE](https://img.shields.io/badge/licence-MIT-blueviolet)](LICENCE)


# Introduction
Pytorch Boiler is a minimalistic boiler plate code for training pytorch models.

## Quick Start

* Clone this repository

    ```
        git clone https://github.com/nmakes/pytorch_boiler
        cd pytorch_boiler
    ```

* Run sample experiments

    40-line [MNIST/CIFAR classification](example_projects/image_classifier/train.py):

    ```
        PYTHONPATH=$PYTHONPATH:./ python3 -m example_projects.image_classifier.train
    ```

    50-line [MNIST/CIFAR autoencoder](example_projects/image_autoencoder/train.py):

    ```
        PYTHONPATH=$PYTHONPATH:./ python3 -m example_projects.image_autoencoder.train
    ```

## Installation

Basic Requirements:

* `numpy`
* `pytorch`
* `torchvision`

Other Requirements:

* `nvidia-apex` [[install]](https://github.com/NVIDIA/apex#from-source) (for mixed-precision training)


## Supported Functionalities

* Customizable Train / Inference engine with forward and infer modes
* Tracking multiple training / validation losses and metrics
* Loading / Saving model, optimizer and trackers based on validation loss
* Training MNIST / CIFAR in 40-lines (see [train.py](example_projects/image_classifier/train.py))
* Supports Apex Amp for mixed precision training


## TODO

* Support for multiple loss optimization using multiple optimizers
* Support for tensorboard plots
* Support for torchscript
* Documentation: using metrics and losses beyond the example script


# Cite

If you found Pytorch Boiler useful in your project, please cite the following:

```
@software{Venkat_Pytorch_Boiler,
    author = {Venkat, Naveen},
    title = {{Pytorch Boiler}},
    url = {https://github.com/nmakes/pytorch_boiler},
    year = {2022}
}
```

Thanks!