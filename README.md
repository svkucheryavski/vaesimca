# VAESIMCA — one class classifier based on Variational Autoencoders and data driven SIMCA approach

**The package is currently in developing stage and break changes may occur. Do not use it so far for real cases. The stable release will come soon**

The package *vaesimca* implements a method for creating one-class classification (OCC) models (also known as *anomaly detectors* or *novelty detectors*) using [Variational Autoencoders](https://en.wikipedia.org/wiki/Variational_autoencoder) (VAE). The overall idea is based on another OCC method, [DD-SIMCA](http://dx.doi.org/10.1002/cem.3556), and hence can be considered as a adaptation of the DD-SIMCA approach using VAE for data decomposition. The theoretical background and practical examples for the *vaesimca* methods are described in a paper which is currently under review (the link comes here later). It is recommended to read it first and then come back and try the examples below.

Although the method we proposed is versatile, the package implements VAESIMCA method for analysis of images. It can be installed from [PyPI](https://pypi.org) using `pip` or any other package manager compatible with PyPI, e.g.:

```
pip3 install vaesimca
```

It requires `numpy`, `scipy`, `torch`, `torchvision`, `pandas` and `matplotlib`,  which will be automatically installed as dependencies.


## Getting started

Use Jupyter notebook [demo.ipynb](https://github.com/svkucheryavski/vaesimca/blob/main/demo/demo.ipynb) in order to get started. To run the examples from this notebook you need to download zip file with simulated dataset (it is also used for illustration of the method in the paper). Here is [direct link](https://github.com/svkucheryavski/vaesimca/raw/main/demo/images_simulated.zip) to the archive with the dataset.

Simply download the dataset and unzip it to your working directory, where you have the notebook, and follow the guides.

## Releases

**0.3.7**
* fixed a bug in saving state dictionary of the best model during training loop
