# Install

## 1. Install / update conda

For a fresh conda installation:

1. [Download conda version](https://docs.conda.io/en/latest/miniconda.html#linux-installers)
2. Run installer: `bash Miniconda3-latest-Linux-x86_64.sh`

If you have an existing conda installation, update it using:

```
conda update -n base -c defaults conda
```

## 2. Creating an environment

Create a conda environment and install the dependencies:

```
conda create -n boiler python=3.9
conda activate boiler
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
python3 -m pip install matplotlib
```

## 3. Download boiler

```
git clone git@github.com:nmakes/pytorch_boiler.git
```

## 4. Run sample model training
```
cd pytorch_boiler
PYTHONPATH=$PYTHONPATH:./ python3 -m \
    example_projects.image_classifier.train
```