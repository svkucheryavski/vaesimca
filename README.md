# VAESIMCA — one class classifier based on Variational Autoencoders and data driven SIMCA approach


The package *vaesimca* implements a method for creating one-class classification (OCC) models (also known as *anomaly detectors* or *novelty detectors*) using [Variational Autoencoders](https://en.wikipedia.org/wiki/Variational_autoencoder) (VAE). The overall idea is based on another OCC method, [DD-SIMCA](http://dx.doi.org/10.1002/cem.3556), and hence can be considered as a adaptation of the DD-SIMCA approach using VAE for data decomposition. The theoretical background and practical examples for the *vaesimca* methods are described in [this paper](https://doi.org/10.1016/j.chemolab.2024.105276), please use it for citation. The paper is freely available to everyone via open access option, it is recommended to read it first and then come back and try the examples from the demo notebook.

Although the method we proposed is versatile, the package implements VAESIMCA method for analysis of images. It can be installed from [PyPI](https://pypi.org) using `pip` or any other package manager compatible with PyPI, e.g.:

```
pip install vaesimca
```

It requires `numpy`, `scipy`, `torch`, `torchvision`, `pandas`, `torch_lr_finder` and `matplotlib`,  which will be automatically installed as dependencies.


## Getting started

Use Jupyter notebook [demo.ipynb](https://github.com/svkucheryavski/vaesimca/blob/main/demo/demo.ipynb) in order to get started. To run the examples from this notebook you need to download zip file with simulated dataset (it is also used for illustration of the method in the paper). Here is [direct link](https://github.com/svkucheryavski/vaesimca/raw/main/demo/images_simulated.zip) to the archive with the dataset.

Simply download the dataset and unzip it to your working directory, where you have the notebook, and follow the guides.  CSV files can be downloaded from GitHub as well.

## Releases

**1.0.0** (18/12/2025)
* fixed a bug leading to lack of reproducibility when `predict()` is called several times.
* when fitting a model, the loss value by default is now normalized to image size and batch size which makes it more stable and reproducible. If you want to use the previous way of computing loss, provide `loss_norm = False` to the method `fit()`.
* method `gridsearch()` can now be used with CSV based data.
* method `plotDistance()` shows objects in the same order as they were loaded without regrouping them.
* method `plotError()` now also works with object index (e.g. show error for object `12`).
* method `stat()` returns two outcomes instead of one: the number of accepted/rejected objects for each class (like in previous version) and the figures of merits (TN, FN, TP, FP, sensitivity, specificity and efficiency).
* method `gridsearch()` also returns two data frames, one with all class based details like in previous version and second one with figures of merits.
* method `summary()` now also shows figures of merits (sensitivity, specificity and efficiency).
* added memory and CUDA device cache cleaning after each grid search iteration to avoid memory leaks.
* several smaller improvements and bug fixes.
* see updated [demo.ipynb](https://github.com/svkucheryavski/vaesimca/blob/main/demo/demo) for all details.

**0.4.2** (2/6/2025)
* added possibility to load data from CSV files.
* added learning rate finder option.

**0.3.7**
* fixed a bug in saving state dictionary of the best model during training loop.

## Reference

A. Petersen, S. Kucheryavskiy, *VAE-SIMCA — Data-driven method for building one class classifiers with variational autoencoders*, Chemometrics and Intelligent Laboratory Systems, 256, 2025,
105276, DOI: [10.1016/j.chemolab.2024.105276](https://doi.org/10.1016/j.chemolab.2024.105276).