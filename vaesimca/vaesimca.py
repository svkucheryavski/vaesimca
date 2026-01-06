import os
import sys
import math
import torch
import copy
import numpy as np
import torch.optim as optim
import pandas as pd
import functools
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy.typing as npt
import gc

from PIL import Image
from torch import nn
from torch.nn import Module, functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torch_lr_finder import LRFinder
from scipy.stats.distributions import chi2
from matplotlib.axes import Axes
from matplotlib.image import AxesImage

# how many epochs without improving best validation loss to run
# the training
MAX_EPOCHS_UNTIL_BEST_LOSS = 30

# default plot colors and markers
MARKERS = ['o', 's', '^', 'd', '>', 'h', 'p', 'v']
COLORS = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:olive', 'tab:pink',
          'tab:gray']

MAIN_COLOR = 'tab:blue'

# a small number to add to sd to avoid division by zero
EPSILON = 1e-12

# expected alpha values - this is needed for quantile based limits
alpha_expected = np.array([0.10, 0.05, 0.01, 0.001])

def plot_labels(ax:Axes, x:npt.NDArray, y:npt.NDArray,
               labels:npt.NDArray|list, dy:float = 0 )-> None:
    """
    Shows labels on top of data points with coordinates (x, y)
    """

    for i in range(0, len(x)):
        ax.text(x[i], y[i] + dy, labels[i], color="gray", ha = "center", fontsize = "medium")


def get_limits(u0:float, Nu:float, alpha:float = 0.05, up:npt.NDArray|None = None, method:str = "moments"):
    """
    Compute statistical limits for extreme objects and outliers based on chi-square distribution.
    """
    if method == "moments" or up is None:
        return chi2.ppf(1 - alpha, Nu) * u0 / Nu
    else:
        ind = np.where(np.abs(alpha_expected - alpha) < 0.00000001)[0]
        if len(ind) != 1:
            raise ValueError("Method 'quantiles' can only be used with alpha = 0.10, 0.05, 0.01, or 0.001")
        return up[ind[0]]


def get_group_colors(groups:list) -> dict:
   """
   Returns dictionary with dedicated color for each group name.
   """
   n = len(groups)

   if n == 1:
      colors = ['tab:blue']
   elif n == 2:
      colors = ['tab:blue', 'tab:red']
   elif n == 3:
      colors = ['tab:blue', 'tab:orange', 'tab:red']
   elif n <= 10:
      cmap = plt.get_cmap("tab10", n)
      colors = [cmap(i) for i in range(n)]
   elif n <= 20:
      cmap = plt.get_cmap("tab20", n)
      colors = [cmap(i) for i in range(n)]
   else:
      raise ValueError("Number of groups is too large (>20) for distingushing them with colors.")

   return dict(zip(groups, colors))


def get_distparams(u: npt.NDArray[np.float64]) -> tuple[float, float, npt.NDArray]:
    """
    Computes parameters of a scaled chi-square distribution that approximate the distribution of
    the distance values using the method of moments as well as a set of quantiles for quantile
    based estimation of critical values.

    Parameters
    ----------
        u : NDAarray
            A vector (1D array) of distances to compute the distribution parameters for.

    Returns
    -------
    A tuple containing two estimated parameters and array with quantiles:

    u0 : float
        The mean of the input distances.
    Nu : float
        The estimated number of degrees of freedom for chi-squared distribution.
    up : NDArray
        Array with the 90th, 95th, 99th and 99.9th percentiles.

    Raises
    ------
    ValueError
        If the input array is empty.

    Notes
    -----
    The function calculates the mean (u0) and variance (vu) of the values in the input array `u`.
    If the coefficient of variation is very small (less than 1e-6), the function returns (u0, 1) to avoid
    division by zero in subsequent calculations. Otherwise, it calculates using the moments approach
    described in DD-SIMCA tutorial.
    """

    if u.size == 0:
        raise ValueError("Input array 'u' must not be empty.")

    u0 = u.mean()
    vu = ((u - u0)**2).mean()
    u02 = u0 ** 2
    up = np.quantile(u, 1 - alpha_expected)

    if u02 < EPSILON or vu < EPSILON or math.sqrt(vu/u02) < EPSILON:
        return (u0, 1, up)

    return (u0, 2 * u02 / vu, up)


class VAEInputTargetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, _ = self.dataset[index]  # discard label
        return x, x  # input and target are the same for VAE

    def __len__(self):
        return len(self.dataset)



class VAELoss:
    """ Class for computing loss values """

    def __init__(self, beta: float = 1.0):
        """
        VAE loss for continuous-valued images (grayscale or RGB) using Gaussian likelihood.

        Parameters
        ----------
        beta : float
            Weight for the KL divergence term (beta-VAE)
        """
        if beta < 0:
            raise ValueError(f"Beta must be non-negative, got {beta}")
        self.beta = beta

    def __call__(self, model_output, target):
        """
        Compute per-sample VAE loss.

        Parameters
        ----------
        model_output : tuple
            Tuple of (reconstructed_images, mu, logvar) from the VAE decoder/encoder
        target : torch.Tensor
            Original images, shape (batch_size, channels, height, width)
        """
        recon_x, mu, logvar = model_output
        batch_size = recon_x.size(0)

        # reconstruction loss per pixel per sample
        recon_loss = F.mse_loss(recon_x, target, reduction='none')
        recon_loss = recon_loss.view(batch_size, -1).sum(dim=1)

        # KL divergence per sample
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        # Return mean over batch (per-sample ELBO)
        return (recon_loss + self.beta * kl_div).mean()



class CSVImageDataset(Dataset):
    """ Class to handle image dataset stored as CSV file (each row is an image). """

    def __init__(self, csv_path:str, img_size:tuple, transform=None, index_col:int = 0):
        """
        Creates an instance of `CSVImageDataset` class. It is assumed that class names are located
        in the first column of the data frame.

        Parameters
        ----------
        csv_path : str
            Full path to CSV file with data values.
        transform : callable, optional
            A function/transform that preprocess the data.
        index_col : int, optional
            Index of column containing row labels.
        """

        # load data from CSV file
        self.data = pd.read_csv(csv_path, index_col=index_col)

        # check image size
        npixels_expected = np.prod(img_size)
        npixels_provided = self.data.iloc[:, 1:].values.shape[1]
        if npixels_expected != npixels_provided:
            raise ValueError(f"Image size mismatch: expected {npixels_expected} pixels, got {npixels_provided}")


        # extract class related information and labels
        self.classnames = sorted(self.data.iloc[:, 0].unique())
        self.classes = self.classnames
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classnames)}
        self.labels = self.data.iloc[:, 0].map(self.class_to_idx).values

        # extract images and related information
        self.images = self.data.iloc[:, 1:].values.astype(np.float32)
        self.samples = [(str(i), label) for i, label in enumerate(self.labels)]
        self.img_size = img_size
        self.transform = transform

        # required for compatibility with ImageFolder-like code
        self.samples = [(str(i), label) for i, label in enumerate(self.labels)]
        self.imgs = self.samples  # Alias .imgs to .samples
        self.targets = self.labels.tolist()  # Optional but standard

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):


        if self.img_size[2] == 1:
            image = self.images[idx].reshape(self.img_size[0], self.img_size[1]).astype(np.uint8)
            image = Image.fromarray(image, mode='L')
        else:
            image = self.images[idx].reshape(self.img_size[0], self.img_size[1], self.img_size[2]).astype(np.uint8)
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


class VAESIMCA(nn.Module):
    """ Class for VAESIMCA model. """

    def __init__(self, encoder_class:Module, decoder_class:Module, classname:str, img_size:tuple,
                 latent_dim:int, transform: transforms.Compose, device:str|None=None):
        """
        Creates an instance of `VAESIMCA` class.

        Parameters
        ----------
        encoder_class : nn.Module
            The class that defines the encoder structure. It must take 'img_size' and 'latent_dim' as arguments.
        decoder_class : nn.Module
            The class that defines the decoder structure. It must take 'img_size' and 'latent_dim' as arguments.
        classname : str
            The name of the target class the model should be trained for.
        img_size : tuple
            A tuple (width, height, num_channels) defining the input image size.
        latent_dim : int
            The dimensionality of the latent space.
        transform : callable
            A function/transform that preprocess the data.
        device: str, optional
            Name of device to run computation on (if not set, will be detected automatically).
        Raises
        ------
        ValueError
            If the input image size or latent dimensions do not meet the requirements.

        """

        super(VAESIMCA, self).__init__()

        # initialization checks
        if len(img_size) != 3:
            raise ValueError("Parameter 'img_size' should include three values (width, height, num_channels).")
        if img_size[0] < 2 or img_size[1] < 2 or img_size[2] < 1:
            raise ValueError("Image size should be at least 2x2 pixels with 1 channel.")
        if latent_dim < 2:
            raise ValueError("Latent dimension must be at least 2.")
        if latent_dim >= img_size[0] * img_size[1]:
            raise ValueError("Latent dimension must be smaller than the number of pixels in the image.")

        # set main model parameters
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.npixels = img_size[0] * img_size[1]
        self.transform = transform
        self.classname = classname

        # set the device and initialize encoder/decoder
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.encoder = encoder_class(img_size=img_size, latent_dim=latent_dim).to(self.device)
        self.decoder = decoder_class(img_size=img_size, latent_dim=latent_dim).to(self.device)



    def reparameterize(self, mu:torch.Tensor, logvar:torch.Tensor) -> torch.Tensor:
        """
        Performs the reparameterization trick to sample from the latent space.

        Parameters
        ----------
        mu : torch.Tensor
            The mean from the encoder's latent space.
        logvar : torch.Tensor
            The log variance from the encoder's latent space.

        Returns
        -------
        torch.Tensor
            The sampled latent vector.
        """

        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def forward(self, x:torch.Tensor)->tuple:
        """ Wrapper of _forward method to be used for LRFinder """
        recon_x, mu, logvar, _ = self._forward(x)
        return recon_x, mu, logvar

    def _forward(self, x:torch.Tensor)->tuple:
        """
        Maps data to a latent space using the trained the VAESIMCA model and computes the reconstructed version of the data.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        tuple
            A tuple containing the reconstructed input, the latent mean, the latent log variance, and the sampled latent vector.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z


    def _getdecomp(self, data:Dataset) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Decomposes the input data using the VAE model to get latent representations and reconstruction errors.

        Parameters
        ----------
        data : Dataset
            The dataset to decompose.

        Returns
        -------
        tuple with two elements

        Z : NDArray
            Latent representations of all data items (matrix nrows x latent dimension size).
        E : NDArray
            Reconstruction errors of all data items (matrix nrows x npredictors)
        """

        data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        Z = np.zeros((len(data_loader), self.latent_dim))
        E = np.zeros((len(data_loader), self.npixels * self.img_size[2]))

        self.eval()
        with torch.no_grad():
            for i, image in enumerate(data_loader):
                img = image[0].to(self.device)
                pred, _, _, z = self._forward(img.to(self.device))
                pred = pred.cpu().detach().numpy().squeeze()
                img = img.cpu().detach().numpy()
                z = z.cpu().detach().numpy().squeeze()
                Z[i, :] = z
                E[i, :] = (img - pred).reshape(1, self.npixels * self.img_size[2])

        return Z, E


    def _train_vae(self, data:Dataset, batch_size:int=10, nepochs:int=30, lr:float=0.001,
                   val_ratio:float = 0.2, tol:float = 0.05, beta:float = 1.0,
                   scheduler_step_size:int = 10, scheduler_gamma:float = 0.5,
                   verbose:bool=True):
        """
        Trains the VAE part of the model using the provided data.
        """

        dataset_size = len(data)
        if verbose:
            print(f"Training VAE model (nimg: {dataset_size} dim: {self.latent_dim}, epochs: {nepochs})")

        # create data indices for training and validation splits:
        indices = np.arange(dataset_size)
        split = int(np.floor(val_ratio * dataset_size))

        if split < 1 or (dataset_size - split) < 1:
            raise ValueError(f"Dataset too small for validation split: {dataset_size} samples")

        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # create data samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # create data loader
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler)

        # set up optimizer and scheduled to tune learning rate
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # loss method
        loss_fun = VAELoss(beta=beta)

        # loop over epochs
        best_loss = float('inf')
        best_loss_epochs = 0

        for epoch in range(nepochs):
            self.train()
            total_train_loss = 0.0
            total_train_samples = 0

            # loop throw batches
            for data_batch, _ in train_loader:
                data_batch = data_batch.to(self.device)
                optimizer.zero_grad()
                recon_batch, mu, logvar, _ = self._forward(data_batch)
                loss = loss_fun((recon_batch, mu, logvar), data_batch)
                loss.backward()
                optimizer.step()

                batch_size_local = data_batch.size(0)
                total_train_loss += loss.item() * batch_size_local
                total_train_samples += batch_size_local

            train_loss = total_train_loss / total_train_samples
            scheduler.step()

            # set model to evaluation mode and apply to validation set
            self.eval()
            total_val_loss = 0.0
            total_val_samples = 0
            with torch.no_grad():
                for data_batch, _ in val_loader:
                    data_batch = data_batch.to(self.device)
                    recon_batch, mu, logvar, _ = self._forward(data_batch)
                    loss = loss_fun((recon_batch, mu, logvar), data_batch)

                    batch_size_local = data_batch.size(0)
                    total_val_loss += loss.item() * batch_size_local
                    total_val_samples += batch_size_local

            val_loss = total_val_loss / total_val_samples

            # print the current state
            if verbose and (epoch % 10 == 0 or epoch == (nepochs - 1)):
                lr_loc = optimizer.param_groups[0]["lr"]
                print(f"Epoch {(epoch + 1):4d}/{nepochs:4d} - lr: {lr_loc:.10f} - train loss: {train_loss:7.2f} - val loss: {val_loss:7.2f}")

            if val_loss < best_loss:
                # if validation loss is better than the previous best, set the new values for
                # the best, best model and reset epochs counter
                best_loss = val_loss
                best_loss_epochs = 0
                best_model = copy.deepcopy(self.state_dict())
            elif ((val_loss - best_loss) / best_loss > tol):
                # if difference between current and best validation loss is too large, stop training
                if verbose:
                    print("The validation loss is getting worse —  stop training.")
                break
            elif best_loss_epochs >= MAX_EPOCHS_UNTIL_BEST_LOSS:
                if verbose:
                    print(f"No improvements during last {best_loss_epochs} epochs —  stop training.")
                break

            best_loss_epochs += 1

        # load parameters of the best model and set model to
        # evaluation mode
        self.load_state_dict(best_model)
        self.eval()

        if verbose:
            print(f"Finished. Best validation loss: {best_loss:.2f}.")


    def _train_simca(self, data:Dataset):
        """
        Trains the SIMCA part of the model and set the related object properties.
        """

        Z, E = self._getdecomp(data)
        self.z_mean = Z.mean(axis=0)
        self.z_sd = Z.std(axis=0)

        U, s, V = np.linalg.svd((Z - self.z_mean) / (self.z_sd + EPSILON), full_matrices=False)
        comp_ind = s > EPSILON
        self.s = s[comp_ind]
        self.V = np.transpose(V[comp_ind, :])
        U = U[:, comp_ind]
        h = (U ** 2).sum(axis=1)
        q = (E ** 2).sum(axis=1)

        h0, Nh, hp = get_distparams(h)
        q0, Nq, qp = get_distparams(q)


        f = h / h0 * Nh + q / q0 * Nq
        f0, Nf, fp = get_distparams(f)

        self.hParams = (h0, Nh, hp)
        self.qParams = (q0, Nq, qp)
        self.fParams = (f0, Nf, fp)
        self.n = len(q)


    def _get_dataset(self, data_path:str):
        """ Returns Dataset object """
        if os.path.isdir(data_path):
            return datasets.ImageFolder(root=data_path, transform=self.transform)
        else:
            return CSVImageDataset(csv_path=data_path, img_size=self.img_size, transform=self.transform)



    def findlr(self, data_path:str, batch_size:int=10, beta:float=1.0,
               num_iter:int=100, start_lr:float=1e-7, end_lr:float=100,
               weight_decay:float=0):
        """
        Applied FindLR method from "torch_lr_finder" package to find optimal learning rate.

        Parameters
        ----------
        data_path : str
            Path to the directory containing images for the training set or to a CSV file with image data as rows.
        batch_size : int, optional
            Batch size for training.
        beta: float, optional
            Regularization parameter for total loss (loss = reconstruction loss + beta * KL divergence).
        num_iter : int, optional
            Maximum number of iterations.
        start_lr : float, optional
            Initial learning rate for the finder.
        end_lr : float, optional
            Final learning rate for the finder.
        weight_decay : float, optional
            Weight decay parameter for Adam optimizer.

        Returns
        -------
        lrfinder:
            Object of LRFinder class which can be used to make plots, etc.

        """

        data = VAEInputTargetWrapper(self._get_dataset(data_path))
        train_loader = DataLoader(data, batch_size=batch_size)
        criterion = VAELoss(beta=beta)

        optimizer = optim.Adam(self.parameters(), lr=start_lr, weight_decay=weight_decay)
        lr_finder = LRFinder(self, optimizer, criterion, device=self.device)
        lr_finder.range_test(train_loader, end_lr=end_lr, num_iter=num_iter)
        lr_finder.reset()

        return lr_finder


    def fit(self, data_path:str, nepochs:int=30, batch_size:int=10, lr:float=0.001,
                val_ratio:float = 0.2, tol:float = 0.05, beta:float = 1.0,
                scheduler_step_size:int = 10, scheduler_gamma:float = 0.5,
                verbose:bool=True):
        """
        Train VAESIMCA model and set proper model parameters, so it is ready for predictions.

        Parameters
        ----------
        data_path : str
            Path to the directory containing images for the training set or to a CSV file with image data as rows.
        nepochs : int, optional
            The number of epochs to train the VAE part of the model.
        batch_size : int, optional
            Batch size for training.
        lr : float, optional
            Initial learning rate for the optimizer.
        val_ratio : float, optional
            Ratio of training data to use for validation set.
        tol: float, optional
            Tolerance, if validation loss is more that (tol * 100)% worse than the current best loss training process will stop.
        beta: float, optional
            Regularization parameter for total loss (loss = reconstruction loss + beta * KL divergence).
        scheduler_step_size : int, optional
            Step size for scheduler to adjust learning rate.
        scheduler_gamma : float, optional
            Gamma parameter for scheduler to adjust learning rate.
        verbose : bool, optional
            If True, prints detailed logs during training.


        Raises
        ------
        ValueError
            If no images found under the specified class name in the given path.
        """

        data_all = self._get_dataset(data_path)

        # get a subset of thr training set from the target class
        try:
            class_idx = data_all.class_to_idx[self.classname]
            filtered_indices = [i for i, (_, label) in enumerate(data_all.samples) if label == class_idx]
            data = torch.utils.data.Subset(data_all, filtered_indices)
        except KeyError:
            raise ValueError(f"Cannot find images with class name '{self.classname}' in the provided path.")

        if len(data) < 1:
            raise ValueError(f"No images found with class name '{self.classname}' in the provided path.")

        # train VAE and simca parts
        self._train_vae(data=data, batch_size=batch_size, nepochs=nepochs, lr=lr, val_ratio=val_ratio,
                        scheduler_step_size=scheduler_step_size, scheduler_gamma=scheduler_gamma,
                        tol=tol, beta=beta, verbose=verbose)
        self._train_simca(data=data)


    def predict(self, data_path:str, alpha:float=0.05, distance:str="f", method:str = "moments"):
        """
        Predicts using the trained model on the given dataset.

        Parameters
        ----------
        data_path : str
            Path to the directory with images or to CSV file with image data on which to perform predictions.
        alpha : float, optional
            Significance level to define expected sensitivity of the model.
        distance : str, optional
            Which distance to use for classification (use 'f': full, 'q': residual, 'h': explained).
        method: str, optional
            Name of method ("moments" or "quantiles") to use for computing critical limits.

        Returns
        -------
        VAESIMCARes
            A VAESIMCARes object containing the predictions and statistical analysis.
        """

        if alpha < 0.00001 or alpha > 0.999999:
            raise ValueError("Wrong value for parameter 'alpha' (must be between 0.00001 and 0.999999).")

        if distance not in ["f", "q", "h"]:
            raise ValueError(f"Invalid value for distance: '{distance}'. Must be 'f', 'q', or 'h'.")

        if method == "quantiles" and alpha not in [0.10, 0.05, 0.01, 0.001]:
            raise ValueError("For quantile based predictions alpha should have one of the following values: 0.10, 0.05, 0.01, or 0.001.")

        self.eval()
        data = self._get_dataset(data_path)

        labels = [os.path.splitext(os.path.basename(path))[0] for path, label in data.imgs]
        classes = data.classes
        class_labels = [classes[i] for i in data.targets]

        Z, E = self._getdecomp(data)
        T = np.dot((Z - self.z_mean) / (self.z_sd + EPSILON), self.V)
        U = np.dot(T, np.diag(1 / self.s))

        h = (U ** 2).sum(axis=1)
        q = (E ** 2).sum(axis=1)

        return VAESIMCARes(self.classname, self.img_size, Z, E, T, U, q, h, self.qParams, self.hParams,
                           self.fParams, alpha, labels, class_labels, classes, distance, method)


    @staticmethod
    def gridsearch(train_path:str, test_path:str, classname:str, encoder_class:nn.Module,
        decoder_class:nn.Module, img_size:tuple, transform:transforms.Compose, nepochs:int=30,
        scheduler_step_size = 10, scheduler_gamma = 0.5, verbose:bool=True,
        lr_seq = [0.001], ld_seq =[4, 8, 16], bs_seq = [10, 20], beta_seq = [0.5, 1.0],
        niter:int=3) -> pd.DataFrame:
        """
        Conducts a grid search over specified hyperparameters for training the VAESIMCA model.

        Parameters
        ----------
        train_path : str
            The path to the training data (directory with images or CSV file).
        test_path : str
            The path to the testing data (directory with images or CSV file).
        classname : str
            The target class name (only subset of training data which match this class will be used for training).
        encoder_class : nn.Module
            The class that defines the encoder structure.
        decoder_class : nn.Module
            The class that defines the decoder structure.
        img_size : tuple
            The dimensions of the input images (width, height, num_channels).
        transform : callable
            A function/transform that preprocesses the images.
        nepochs : int, optional
            The maximum number of epochs to train for each configuration.
        scheduler_step_size : int, optional
            Step size for scheduler to adjust learning rate.
        scheduler_gamma : float, optional
            Gamma parameter for scheduler to adjust learning rate.
        verbose : bool, optional
            If True, prints detailed logs during the grid search.
        lr_seq : list, optional
            A list of learning rates to try.
        ld_seq : list, optional
            A list of latent dimensions to try.
        bs_seq : list, optional
            A list of batch sizes to try.
        beta_seq : list, optional
            A list of regularization parameters (beta) to try.
        niter : int, optional
            Number of iterations to run each combination

        Returns
        -------
        DataFrame with following columns

            comb :
                Combination ID (identifies unique combination of parameters for optimization)
            class :
                Name of the class (results are computed separately for each target and alternative classes)
            beta :
                Beta value used to get this results
            bs :
                Batch size used to get this results
            lr :
                Learning rate used to get this results
            ld :
                Latent space dimension used to get this results
            sens :
                Average sensitivity for given combination
            spec :
                Average specificity
            eff :
                Average efficiency

        """

        def show_progress(p, n, sens, spec, eff):
            """ method to print current progress state """
            pct = p / n
            scaled_p = int(50 * pct)  # Scale p to fit within the range [0, 50]
            symbols_line = "[" + "#" * scaled_p + " " * (50 - scaled_p) + f"] {pct*100:5.1f}% - ({sens:.3f}/{spec:.3f}/{eff:.3f})"
            sys.stdout.write("\r" + symbols_line)
            sys.stdout.flush()

        # get test set classes and combine with training class
        test_data = datasets.ImageFolder(root=test_path, transform=transform) if os.path.isdir(test_path) \
            else CSVImageDataset(csv_path=test_path, img_size=img_size, transform=transform)
        test_classes = test_data.classes

        if len(test_classes) < 1:
            raise ValueError("No subdirectories found in the path specified by 'test_path' parameter.")

        # combine all arguments and calculate number of their combinations
        args = [beta_seq, bs_seq, lr_seq, ld_seq]
        nargs = len(args)
        ncombs = functools.reduce(lambda l, b: l * len(b), args, 1)

        # get all combinations of the parameters to optimize and repeat for each class
        params = np.array(np.meshgrid(beta_seq, bs_seq, lr_seq, ld_seq))
        params = params.reshape([nargs, ncombs])


        f_comb = []
        f_beta = []
        f_bs = []
        f_lr = []
        f_ld = []
        f_sens = []
        f_spec = []
        f_eff = []

        # best model
        sens_best = (0, [])
        spec_best = (0, [])
        eff_best = (0, [])


        if verbose:
            print(f"\nRun grid search with {ncombs} combinations x {niter} iterations:")
            print("-----------------------------------------------------------------")


        # loop over parameters:
        for p in range(0, ncombs):

            beta, batch_size, lr, ld = params[:, p]

            # loop over iterations
            sens = 0.
            spec = 0.
            eff  = 0.
            for i in range(0, niter):

                m = VAESIMCA(encoder_class=encoder_class, decoder_class=decoder_class, classname=classname,
                             img_size=img_size, latent_dim=int(ld), transform=transform)

                try:
                    m.fit(data_path=train_path, nepochs=nepochs, beta=beta, lr=lr, batch_size=int(batch_size),
                         scheduler_gamma=scheduler_gamma, scheduler_step_size=scheduler_step_size,
                         verbose=False)

                except Exception as error:
                    print("A critical problem occured when training the model with following parameters:")
                    print(f"lr = {lr} ld = {ld} beta = {beta} batch size = {batch_size}")
                    print(error)

                _, fomt = m.predict(test_path).stat()

                # get foms for test set
                sens = sens + (fomt["sens"] if fomt["sens"] is not None else 0)
                spec = spec + (fomt["spec"] if fomt["spec"] is not None else 0)
                eff  = eff  + (fomt["eff"]  if fomt["eff"]  is not None else 0)

                # free memory
                del m
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()

            sens = sens / niter
            spec = spec / niter
            eff  = eff / niter


            # adjust best FoM values
            if spec and spec > spec_best[0]:
                spec_best = (spec, [lr, ld, beta, batch_size])
            if sens and sens > sens_best[0]:
                sens_best = (sens, [lr, ld, beta, batch_size])
            if eff and eff > eff_best[0]:
                eff_best = (eff, [lr, ld, beta, batch_size])

            f_comb.append(p)
            f_beta.append(beta)
            f_bs.append(batch_size)
            f_lr.append(lr)
            f_ld.append(ld)
            f_sens.append(sens)
            f_spec.append(spec)
            f_eff.append(eff)

            if verbose:
                show_progress(p + 1, ncombs, sens_best[0], spec_best[0], eff_best[0])



        if verbose:
            print("\n-----------------------------------------------------------------\n")
            print(f"Best sensitivity:  {sens_best[0]:.3f} - lr={sens_best[1][0]:6f} ld={sens_best[1][1]:4.0f} beta={sens_best[1][2]:.1f} bs={sens_best[1][3]:4.0f}")
            print(f"Best specificity:  {spec_best[0]:.3f} - lr={spec_best[1][0]:6f} ld={spec_best[1][1]:4.0f} beta={spec_best[1][2]:.1f} bs={spec_best[1][3]:4.0f}")
            print(f"Best efficiency:   { eff_best[0]:.3f} - lr={ eff_best[1][0]:6f} ld={ eff_best[1][1]:4.0f} beta={ eff_best[1][2]:.1f} bs={ eff_best[1][3]:4.0f}")

        return pd.DataFrame({"comb": f_comb, "beta": f_beta, "bs": f_bs, "lr": f_lr, "ld": f_ld, "sens": f_sens, "spec": f_spec, "eff": f_eff})



class VAESIMCARes:
    """
    A class to hold and process results from the VAESIMCA model predictions. Do not use it manually, it is used by the
    'predict' method from the VAESIMCA class.

    Parameters
    ----------
    Z : numpy.ndarray
        2D array with latent representation of all images (in rows)
    E : numpy.ndarray
        2D array with reconstruction errors of all images (in rows)
    T : numpy.ndarray
        2D array with scores (left singular vectors) - projections of Z values to SVD space defined by the training set.
    U : numpy.ndarray
        2D array with standardized scores (like T but with unit variance).
    q : numpy.ndarray
        Array of residual distances.
    h : numpy.ndarray
        Array of explained distances.
    qParams : tuple
        Parameters of the distribution for the residual distances.
    hParams : tuple
        Parameters of the distribution for the explained distances.
    fParams : tuple
        Parameters of the distribution for the full distances.
    alpha : float
        The significance level used to make predictions.
    labels : list
        List of labels for the data points (image file names).
    class_labels : list
        List of class labels for the data points.
    classes : list
        List of classes in the dataset the predictions were made for.
    method: str
        Name of method ("moments" or "quantiles") to use for computing critical limits.

    Attributes
    ----------
    n : int
        Number of data points.
    Z : numpy.ndarray
        2D array with latent representation of all images (in rows)
    E : numpy.ndarray
        2D array with reconstruction errors of all images (in rows)
    T : numpy.ndarray
        2D array with scores (left singular vectors) - projections of Z values to SVD space defined by the training set.
    U : numpy.ndarray
        2D array with standardized scores (like T but with unit variance).
    q : numpy.ndarray
        Residual distances.
    h : numpy.ndarray
        Explained distances.
    f : numpy.ndarray
        Full distances based on q and h.
    hParams : tuple
        Parameters for explained distances.
    qParams : tuple
        Parameters for residual distances.
    fParams : tuple
        Parameters for full distances.
    alpha : float
        Significance level for statistical tests.
    regular : numpy.ndarray
        Boolean array indicating whether each data point is within the expected range.
    labels : list
        Labels of the data points.
    classes : list
        All classes in the dataset.
    class_labels : list
        Class labels for the data points.

    Methods
    -------
    stat()
        Computes statistics for the number of data points falling within expected ranges per class.
    summary()
        Prints a summary of the statistics.
    plotDistance(plt, distance="q", colors=None, legend_loc=2)
        Plots the specified type of distance for the data points.
    plotAcceptance(plt, do_log=False, colors=None, markers=None)
        Plots an acceptance graph showing the explained and residual distances and decision boundary.
    """
    def __init__(self, target_class:str, img_size:tuple, Z:npt.NDArray, E:npt.NDArray, T:npt.NDArray, U:npt.NDArray, q:npt.NDArray, h:npt.NDArray,
                 qParams:tuple, hParams:tuple, fParams:tuple, alpha:float, labels:list, class_labels:list, classes:list,
                 crit:str, method:str):

        n = len(q)
        h0, Nh, hp = hParams
        q0, Nq, qp = qParams
        f0, Nf, fp = fParams

        self.target_class = target_class
        self.img_size = img_size
        self.n = n
        self.Z = Z
        self.E = E
        self.T = T
        self.U = U
        self.q = q
        self.h = h
        self.f = h / h0 * Nh + q / q0 * Nq
        self.hParams = hParams
        self.qParams = qParams
        self.fParams = fParams
        self.alpha = alpha
        self.labels = labels
        self.classes = classes
        self.class_labels = class_labels
        self.method = method

        if crit == "f":
            self.regular = self.f < get_limits(f0, Nf, alpha=alpha, up=fp, method = method)
        elif crit == "q":
            self.regular = self.q < get_limits(q0, Nq, alpha=alpha, up=qp, method = method)
        elif crit == "h":
            self.regular = self.h < get_limits(h0, Nh, alpha=alpha, up=hp, method = method)
        else:
            raise ValueError("Wrong value for parameter 'crit' (must be 'f', 'q', or 'h').")


    def stat(self) -> tuple[dict, dict]:
        """
        Computes statistics for the number of data points falling within expected ranges per class.

        Returns
        -------
        tule
            A tuple with two dictionaries. First contains classes as keys and lists of counts: total,
            within range, and out of range. Second contains number of true positives, false negatives,
            true negatives and false positives.
        """
        stat = {}

        TN = 0
        TP = 0
        FN = 0
        FP = 0
        for c in self.classes:
            decisions = [self.regular[i] for i, label in enumerate(self.class_labels) if label == c]
            total = len(decisions)
            accepted = sum(decisions)
            rejected = total - accepted
            stat[c] = [total, accepted, rejected]

            if c == self.target_class:
                TP = TP + accepted
                FN = FN + rejected
            else:
                TN = TN + rejected
                FP = FP + accepted

        sens = TP / (TP + FN) if (TP + FN) > 0 else None
        spec = TN / (TN + FP) if (TN + FP) > 0 else None
        eff = math.sqrt(sens * spec) if sens and spec else None
        return stat, {"TP":TP, "FN":FN, "TN":TN, "FP":FP, "sens":sens, "spec":spec, "eff":eff}


    def summary(self):
        """
        Prints a summary of the statistics of the model predictions.

        Displays the number of data points per class and how many are accepted/rejected by the model.
        """
        stats, foms = self.stat()

        slen = max(6, max(len(x) for x in self.classes) + 1)
        dlen = max(4, int(math.log10(self.n)) + 1)
        print(f"\n{'class':<{slen}s} {'n':>{dlen}s} {'in':>{dlen}s} {'in (%)':>6s} {'out':>{dlen}s} {'out (%)':>7s}")
        print(f"{'':->{slen + 3 * dlen + 18}s}")
        for c, (total, in_range, out_of_range) in stats.items():
            in_pct = 100 * in_range / total if total > 0 else 0
            out_pct = 100 * out_of_range / total if total > 0 else 0
            print(f"{c:<{slen}s} {total:{dlen}d} {in_range:{dlen}d} {in_pct:6.1f} {out_of_range:{dlen}d} {out_pct:7.1f}")

        print("")
        if foms["sens"]:
            print(f"sensitivity: {foms['sens']:.3f}")
        if foms["spec"]:
            print(f"specificity: {foms['spec']:.3f}")
        if foms["eff"]:
            print(f"efficiency: {foms['eff']:.3f}")


    def as_df(self):
        """ returns the classification results in form of data frame """
        return pd.DataFrame({
            "sample": self.labels,
            "class": self.class_labels,
            "decision": self.regular,
            "h": self.h,
            "q": self.q,
            "f": self.f
        })


    def plotDistance(self, ax:Axes, distance:str="q", colors:dict|None=None, legend_loc:int=2,
                    show_crit:bool = False, show_labels:bool = False):
        """
        Plots the specified type of distance for the data points.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes instance.
        distance : str, optional
            The type of distance to plot ('q', 'h', or 'f').
        colors : dict, optional
            Dictionary with colors (one for each class).
        legend_loc : int, optional
            Location of the legend in the plot.
        show_crit : bool, optional
            Logical, show or not decision and outlier boundaries
        show_labels : bool, optional
            Logical, show or not object labels on top of each bar

        Raises
        ------
        ValueError
            If the specified distance type is not recognized.

        """

        if distance not in ('q', 'h', 'f'):
            raise ValueError("Invalid distance type specified. Choose 'q', 'h', or 'f'.")

        nclasses = len(self.classes)
        if colors is None:
            colors = get_group_colors(self.classes)
        elif len(colors) < nclasses:
            raise ValueError(f"Colors for each of the {nclasses} must be provided.")


        params = {'q': self.qParams, 'h': self.hParams, 'f': self.fParams}
        distances = {'q': self.q / self.qParams[0], 'h': self.h / self.hParams[0], 'f': self.f / self.fParams[0]}[distance]
        title_map = {'q': "Residual", 'h': "Explained", 'f': "Full"}


        x = np.arange(len(distances))
        y = distances
        lbs = self.labels
        dy = np.max(y) * 0.05

        if nclasses > 0 and colors is not None:
            g = self.class_labels
            gu = np.unique(g)
            for c in gu:
                class_points = [(x[j], y[j], lbs[j]) for j, gl in enumerate(g) if gl == c]
                if not class_points:
                    continue
                cx, cy, cl = zip(*class_points)
                cx = np.asarray(cx, dtype=float)
                cy = np.asarray(cy, dtype=float)
                cl = np.asarray(cl)
                ax.bar(cx, cy, color = colors[c], label = c)
                if show_labels:
                    plot_labels(ax, cx, cy, cl, dy)
        else:
            ax.bar(x, y, color = MAIN_COLOR)
            if show_labels:
                plot_labels(ax, x, y, lbs, dy)

        if show_crit:
            u0, Nu, up = params[distance]
            lim = get_limits(u0, Nu, alpha = self.alpha, up = up, method=self.method) / u0
            xr = ax.get_xlim()
            ax.plot(xr, [lim, lim], 'k--', linewidth=0.5)


        ax.legend(loc = legend_loc)
        ax.set_title(f"{title_map[distance]} distance")
        ax.set_ylabel(f"{distance}-distance")
        ax.set_xlabel("Objects")


    def plotAcceptance(self, ax:Axes, do_log:bool=False, colors:dict|None=None, markers:list|None=None,
                       legend_loc:str = "best", show_labels:bool = False):
        """
        Plots an acceptance graph showing scaled explained and residual distances and the decision boundary.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Matplotlib axes instance.
        do_log : bool, optional
            Whether to plot the original distances or log-transformed (log(1 + u)).
        colors : list, optional
            List of colors for each class.
        markers : list, optional
            List of markers for each class in the plot.
        legend_loc : str, optional
            Location of the legend (similar to parameter 'loc').
        show_labels : bool, optional
            Logical, show or not object labels on top of each bar.

        Raises
        ------
        ValueError
            If the number of colors or markers is smaller than the number of classes.

        """

        if markers is None:
            markers = MARKERS

        nclasses = len(self.classes)
        if colors is None:
            colors = get_group_colors(self.classes)
        elif len(colors) < nclasses:
            raise ValueError(f"Colors for each of the {nclasses} must be provided.")

        if len(markers) < len(self.classes):
            raise ValueError(f"Markers for each of the {len(self.classes)} must be provided.")

        h0, Nh, _ = self.hParams
        q0, Nq, _ = self.qParams
        f0, Nf, fp = self.fParams

        h_scaled = np.log1p(self.h / h0) if do_log else self.h / h0
        q_scaled = np.log1p(self.q / q0) if do_log else self.q / q0
        dy = np.max(q_scaled) * 0.05

        for i, c in enumerate(self.classes):
            class_points = [(h_scaled[j], q_scaled[j], self.labels[j]) for j, label in enumerate(self.class_labels) if label == c]
            cx, cy, cl = zip(*class_points)
            ax.scatter(cx, cy, label=c, marker=markers[i], edgecolors=colors[c], facecolors="none")
            if show_labels:
                plot_labels(ax, np.array(cx), np.array(cy), np.array(cl), dy)


        # show decision and outliers boundaries
        fCritE = get_limits(f0, Nf, alpha = self.alpha, up = fp, method=self.method)
        xqeMax = fCritE / Nh
        xqe = np.linspace(0, xqeMax, 200)
        yqe = (fCritE - xqe * Nh) / Nq

        if do_log:
            xqe = np.log1p(xqe)
            yqe = np.log1p(yqe)

        ax.plot(xqe, yqe, 'k--', linewidth=0.5)
        ax.legend(loc = legend_loc)
        ax.grid(color = "#e0e0e0", linestyle = ":")
        ax.set_title("Acceptance Plot")

        if do_log:
            ax.set_xlabel("Explained distance, log(1 + h/h0)")
            ax.set_ylabel("Residual distance, log(1 + q/q0)")
        else:
            ax.set_xlabel("Explained distance, h/h0")
            ax.set_ylabel("Residual distance, q/q0")


    def plotError(self, ax:Axes, ind:int|None = None, classname:str|None = None, object_label:str|None = None) -> AxesImage:
        """
        Show image with reconstruction error for object with given label and class name.

        Parameters
        ----------
        plt : matplotlib.pyplot
            Matplotlib plot module.
        classname : str
            Name of the object's class
        object_label : str
            Label of the object (filename without extension)

        Raises
        ------
        ValueError
            If the specified class or object can not be found.

        """

        if ind is None and (classname is None or object_label is None):
            raise ValueError("You need to specify either object index or class and object labels.")

        if classname and classname not in self.classes:
            raise ValueError("Can not find class with name '{classname}' in this result object.")

        if ind is not None:
            if ind < 0 or ind > self.n - 1:
                raise ValueError("Wrong value for object index.")
            classname = self.class_labels[ind]
            object_label = self.labels[ind]
        elif object_label is not None and classname is not None:
            ind_seq = [i for i in range(self.n) if self.class_labels[i] == classname and self.labels[i] == object_label]
            if len(ind_seq) < 1:
                raise ValueError("Can not find object with label '{object_label}' among elements of class '{classname}'")
            ind = ind_seq[0]

        mn = np.min(self.E)
        mx = np.max(self.E)
        e = self.E[ind, :].reshape(self.img_size[0], self.img_size[1])
        im = ax.imshow(e, vmin = mn, vmax = mx)
        ax.set_title(f"{classname}:{object_label}")
        return im

