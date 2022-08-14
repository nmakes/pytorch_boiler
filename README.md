<p align="center">
    <img src="assets/Pytorch%20Boiler.png">
</p>

[![Python 3 version](https://img.shields.io/badge/python-%3E%3D3.6-blue)](https://www.python.org/downloads/release/python-360/)
[![Pytorch version](https://img.shields.io/badge/pytorch-%3E%3D%201.4.0-informational)](https://pytorch.org/get-started/previous-versions/)


[![Code Size](https://img.shields.io/github/languages/code-size/nmakes/pytorch_boiler)]()

[![LICENCE](https://img.shields.io/badge/licence-MIT-blueviolet)]()


# Introduction
Pytorch Boiler is a minimalistic boiler plate code for training pytorch models.


## Current Functionalities

* Train / Inference loop
* Separate forward and infer modes handling no_grad
* Tracking training / validation loss
* Loading / Saving model, optimizer and tracker state_dict
* 40-line example
* Tracking custom performance metrics


## TODO

* Writing individual metrics for tracking
* Saving model based on validation metric
* Support Apex Amp training
* Support for torchscript
