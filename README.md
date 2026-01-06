# VAESIMCA — one class classifier based on Variational Autoencoders and data driven SIMCA approach


The package *vaesimca* implements a method for creating one-class classification (OCC) models (also known as *anomaly detectors* or *novelty detectors*) using [Variational Autoencoders](https://en.wikipedia.org/wiki/Variational_autoencoder) (VAE). The overall idea is based on another OCC method, [DD-SIMCA](http://dx.doi.org/10.1002/cem.3556), and hence can be considered as a adaptation of the DD-SIMCA approach using VAE for data decomposition. The theoretical background and practical examples for the *vaesimca* methods are described in [this paper](https://doi.org/10.1016/j.chemolab.2024.105276), please use it for citation. The paper is freely available to everyone via open access option, it is recommended to read it first and then come back and try the examples from the demo notebook.

**Pay attention that version 1.1.0 introduces several breaking changes, mostly related to plots, please check the release notes below!**

Although the method we proposed is versatile, the package implements VAESIMCA method for analysis of images. It can be installed from [PyPI](https://pypi.org) using `pip` or any other package manager compatible with PyPI, e.g.:

```
pip install vaesimca
```

It requires `numpy`, `scipy`, `torch`, `torchvision`, `pandas`, `torch_lr_finder` and `matplotlib`,  which will be automatically installed as dependencies.


## Getting started

Use Jupyter notebook [demo.ipynb](https://github.com/svkucheryavski/vaesimca/blob/main/demo/demo.ipynb) in order to get started. To run the examples from this notebook you need to download zip file with simulated dataset (it is also used for illustration of the method in the paper). Here is [direct link](https://github.com/svkucheryavski/vaesimca/raw/main/demo/images_simulated.zip) to the archive with the dataset.

Simply download the dataset and unzip it to your working directory, where you have the notebook, and follow the guides.  CSV files can be downloaded from GitHub as well.

## Releases

**1.1.0** (05/01/2026)
* please pay attention that this version contains breaking changes, so you either have to amend your previously written code or use an older version. Check the short description below and the demo notebook for code examples.
* further improvements to loss computation, it was decided to drop the `log_norm` parameter, introduced in previous version, as the new loss works much better. However it will require smaller beta values for some of the cases, pay attention to that.
* method `predict()` got additional argument, `method`. By default it is set to `"moments"`, in this case the critical limit for classification is computed based on chi-square distribution (as in conventional DD-SIMCA). If you set it to `"quantiles"`, then the critical limit will be computed based on quantiles (e.g. for `alpha = 0.05` it will use 95th quantile computed for the training set distances). This can be useful when training set is large (> 1000). This option though works only with selected `alpha` values: 0.10, 0.05, 0.01 and 0.001.
* the `gridsearch()` has been simplified and improved. First of all now it returns only data frame with average figures of merits for each tested combination of hyperparameters. Second, the best set of hyperparameters is based on average (over iterations) FoM values.
* all plotting methods now require matplotlib's `Axes` object as method's argument (instead of `pyplot`) . The demo notebook has been amended correspondingly.
* optional parameter `show_boundary` for method `plotDistance()` is renamed to `show_crit` to make it more clear (as it shows critical value for corresponding distance).
* optional parameter `colors` for methods `plotDistance()`  and `plotAcceptance()` now should be provided as a dictionary (instead of a list), so every class has a specific color.
* several smaller improvements and bug fixes.
* see updated [demo.ipynb](https://github.com/svkucheryavski/vaesimca/blob/main/demo/demo.ipynb) for all details.


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

**0.4.2** (2/6/2025)
* added possibility to load data from CSV files.
* added learning rate finder option.

**0.3.7**
* fixed a bug in saving state dictionary of the best model during training loop.

## Reference

A. Petersen, S. Kucheryavskiy, *VAE-SIMCA — Data-driven method for building one class classifiers with variational autoencoders*, Chemometrics and Intelligent Laboratory Systems, 256, 2025,
105276, DOI: [10.1016/j.chemolab.2024.105276](https://doi.org/10.1016/j.chemolab.2024.105276).