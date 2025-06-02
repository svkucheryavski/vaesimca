import os
import sys
import math
import torch
import copy
import numpy as np
import matplotlib.pyplot
import numpy.matlib
import torch.optim as optim
import pandas as pd
import functools
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torch_lr_finder import LRFinder
from scipy.stats.distributions import chi2

# how many epochs without improving best validation loss to run
# the training
MAX_EPOCHS_UNTIL_BEST_LOSS = 30

# default plot colors and markers
MARKERS = ['o', 's', '^', 'd', '>', 'h', 'p', 'v']
COLORS = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple', 'tab:olive', 'tab:pink', 'tab:gray']


def plotlabels(plt, x, y, labels):
    """
    Shows labels on top of data points with coordinates (x, y)
    """
    for i in range(0, len(x)):
        plt.text(x[i], y[i], labels[i], color="gray", ha = "center", fontsize = "medium")


def getlimits(u0, Nu, alpha = 0.05):
    """
    Compute statistical limits for extreme objects and outliers based on chi-square distribution.
    """
    return chi2.ppf(1 - alpha, Nu) * u0 / Nu


def getdistparams(u: np.array) -> tuple[float, float]:
    """
    Computes parameters of a scaled chi-square distribution that approximate the distribution of the distance values using the method of moments.

    Parameters
    ----------
    u : np.array
        A vector (1D array) of distances to compute the distribution parameters for.

    Returns
    -------
    tuple
        A tuple containing two estimated parameters:
        - u0 (float): The mean of the input distances.
        - Nu (float): The estimated number of degrees of freedom for chi-squared distribution.

    Raises
    ------
    ValueError
        If the input array is empty.

    Notes
    -----
    The function calculates the mean (u0) and variance (vu) of the values in the input array `u`.
    If the coefficient of variation is very small (less than 1e-6), the function returns (u0, 1) to avoid
    division by zero in subsequent calculations. Otherwise, it calculates `Nu` as 2 * u0^2 / vu.
    """

    if u.size == 0:
        raise ValueError("Input array 'u' must not be empty.")

    u0 = u.mean()
    vu = ((u - u0)**2).mean()
    u02 = u0 ** 2

    if math.sqrt(vu/u02) < 1e-6:
        return (u0, 1)

    return (u0, 2 * u02 / vu)

class VAEInputTargetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        x, _ = self.dataset[index]  # discard label
        return x, x  # input and target are the same for VAE

    def __len__(self):
        return len(self.dataset)

class VAELoss:
    """ Class to compute loss value for VAE """

    def __init__(self, beta=1.0):
        self.beta = beta

    def __call__(self, model_output, target):
        recon_x, mu, logvar = model_output
        recon_loss = F.binary_cross_entropy(recon_x, target, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_div

class CSVImageDataset(Dataset):
    """
    Class to handle image dataset stored as CSV file (each row is an image)

    It is assumed that class names are located in the first column of the data frame.

    Parameters
    ----------
    csv_path : str
        Full path to CSV file with data values.
    transform : callable
        A function/transform that preprocess the data.
    """

    def __init__(self, csv_path:str, img_size:list, transform=None):
        self.data = pd.read_csv(csv_path, index_col=0)
        self.transform = transform

        self.classnames = sorted(self.data.iloc[:, 0].unique())
        self.classes = self.classnames
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classnames)}
        self.labels = self.data.iloc[:, 0].map(self.class_to_idx).values
        self.images = self.data.iloc[:, 1:].values.astype(np.float32)
        self.samples = [(str(i), label) for i, label in enumerate(self.labels)]
        self.img_size = img_size

        # Required for compatibility with ImageFolder-like code
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
            image = self.images[idx].reshape(self.img_size[1], self.img_size[0], self.img_size[2]).astype(np.uint8)
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


class VAESIMCA(nn.Module):
    """
    A variational autoencoder class with added functionalities to use it as DD-SIMCA one-class classifier.

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

    Raises
    ------
    ValueError
        If the input image size or latent dimensions do not meet the requirements.

    """

    def __init__(self, encoder_class:nn.Module, decoder_class:nn.Module, classname:str, img_size:tuple,
                 latent_dim:int, transform: callable, device=None):

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


    def _getdecomp(self, data:Dataset) -> tuple[np.array, np.array]:
        """
        Decomposes the input data using the VAE model to get latent representations and reconstruction errors.

        Parameters
        ----------
        data : Dataset
            The dataset to decompose.

        Returns
        -------
        tuple
            Z (numpy array): Latent representations of all data items.
            E (numpy array): Reconstruction errors of all data items.
        """

        data_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        Z = np.zeros((len(data), self.latent_dim))
        E = np.zeros((len(data), self.npixels * self.img_size[2]))

        for i, image in enumerate(data_loader):
            img = image[0]
            pred, _, _, z = self._forward(img.to(self.device))
            pred = pred.cpu().detach().numpy().squeeze()
            img = img.cpu().detach().numpy()
            z = z.cpu().detach().numpy().squeeze()
            Z[i, :] = z
            E[i, :] = (img - pred).reshape(1, self.npixels * self.img_size[2])

        return Z, E


    def _train_vae(self, data:Dataset, batch_size:int=10, nepochs:int=30, lr:float=0.001,
                   val_ratio:float = 0.2, tol:float = 0.05, beta:float = 1.0,
                   scheduler_step_size = 10, scheduler_gamma = 0.5,
                   verbose:bool=True):
        """
        Trains the VAE part of the model using the provided data.
        """

        dataset_size = len(data)
        if verbose:
            print(f"Training VAE model (nimg: {len(data)} dim: {self.latent_dim}, epochs: {nepochs})")

        # create data indices for training and validation splits:
        indices = list(range(dataset_size))
        split = int(np.floor(val_ratio * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # create data samplers
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        # create data loader
        train_loader = DataLoader(data, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(data, batch_size=batch_size, sampler=valid_sampler)

        # set up optimizer and scheduled to tune learnin rate
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

        # loop over epochs
        best_loss = float('inf')
        best_loss_epochs = 0
        for epoch in range(nepochs):
            self.train()
            train_loss = 0

            # loop throw batches
            for _, (data, _) in enumerate(train_loader):
                data = data.to(self.device)
                optimizer.zero_grad()
                recon_batch, mu, logvar, _ = self._forward(data)
                recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
                kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kl_divergence
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            # average training loss
            train_loss = train_loss / len(train_loader.dataset)
            scheduler.step()

            # set model to evaluation mode and apply to validation set
            self.eval()
            val_loss = 0
            with torch.no_grad():
                for _, (data, _) in enumerate(val_loader):
                    data = data.to(self.device)
                    recon_batch, mu, logvar, _ = self._forward(data)
                    recon_loss = F.binary_cross_entropy(recon_batch, data, reduction='sum')
                    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    val_loss += recon_loss + beta * kl_divergence

            # average validation loss
            val_loss = val_loss / len(val_loader.dataset)

            # print the current state
            if verbose and (epoch % 10 == 0 or epoch == (nepochs - 1)):
                lr_loc = optimizer.param_groups[0]["lr"]
                print(f"Epoch {(epoch + 1):4d}/{nepochs:4d} - lr: {lr_loc:.10f} - train loss: {train_loss:.2f} - val loss: {val_loss:.2f}")

            if val_loss < best_loss:
                # if validation loss is better than the previos best, set the new values for
                # the best, best model and reset epochs counter
                best_loss = val_loss
                best_loss_epochs = 0
                best_model = copy.deepcopy(self.state_dict())
            elif ((val_loss - best_loss) / best_loss > tol):
                # if difference between current and best validation loss is too large, stop training
                if verbose:
                    print(f"The validation loss is getting worse —  stop training.")
                break
            elif ((val_loss - best_loss) / best_loss > tol) or best_loss_epochs >= MAX_EPOCHS_UNTIL_BEST_LOSS:
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

        U, s, V = np.linalg.svd((Z - self.z_mean) / self.z_sd, full_matrices=False)
        self.s = s
        self.V = np.transpose(V)

        h = (U ** 2).sum(axis=1)
        q = (E ** 2).sum(axis=1)

        h0, Nh = getdistparams(h)
        q0, Nq = getdistparams(q)

        f = h / h0 * Nh + q / q0 * Nq
        f0, Nf = getdistparams(f)
        #Nf = Nh + Nq
        #f0 = Nf

        self.hParams = (h0, Nh)
        self.qParams = (q0, Nq)
        self.fParams = (f0, Nf)
        self.n = len(q)


    def _get_dataset(self, data_path:str):
        """ Returns Dataset object """
        if os.path.isdir(data_path):
            return datasets.ImageFolder(root=data_path, transform=self.transform)
        else:
            return CSVImageDataset(csv_path=data_path, img_size=self.img_size, transform=self.transform)



    def findlr(self, data_path:str, batch_size:int=10, beta:float=1.0,
               num_iter:int=100, start_lr:float=1e-7, end_lr:float=100, weight_decay:float=1e-2):
        """
        Applied FindLR method from "torch_lr_finder" package to find optimal learning rate.

        Parameters
        ----------
        data_path : str
            Path to the directory containing images for the training set.
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
        criterion = VAELoss(beta = beta)

        optimizer = optim.Adam(self.parameters(), lr=1e-7, weight_decay=weight_decay)
        lr_finder = LRFinder(self, optimizer, criterion, device=self.device)
        lr_finder.range_test(train_loader, end_lr=100, num_iter=num_iter)
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
            Path to the directory containing images for the training set.
        nepochs : int, optional
            The number of epochs to train the VAE pare of model.
        batch_size : int, optional
            Batch size for training.
        lr : float, optional
            Initial learning rate for the optimizer.
        val_ratio : int, optional
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
            If True, print detailed logs during training.

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


    def predict(self, data_path:str, alpha:float=0.05, distance:str="f"):
        """
        Predicts using the trained model on the given dataset.

        Parameters
        ----------
        data_path : str
            Path to the dataset with images on which to perform predictions.
        alpha : float, optional
            Significance level to define expected sensitivity of the model.
        crit : str, optional
            Which distance to use for classification (use 'f': full, 'q': residual, 'h': explained).

        Returns
        -------
        VAESIMCARes
            A VAESIMCARes object containing the predictions and statistical analysis.
        """

        if alpha < 0.00001 or alpha > 0.999999:
            raise ValueError("Wrong value for parameter 'alpha' (must be between 0.00001 and 0.999999).")

        data = self._get_dataset(data_path)

        labels = [os.path.splitext(os.path.basename(path))[0] for path, label in data.imgs]
        classes = data.classes
        class_labels = [classes[i] for i in data.targets]

        Z, E = self._getdecomp(data)
        T = np.dot((Z - self.z_mean) / self.z_sd, self.V)
        U = np.dot(T, np.diag(1 / self.s))

        h = (U ** 2).sum(axis=1)
        q = (E ** 2).sum(axis=1)

        return VAESIMCARes(self.img_size, Z, E, T, U, q, h, self.qParams, self.hParams, self.fParams, alpha, labels, class_labels, classes, distance)


    @staticmethod
    def gridsearch(train_path:str, test_path:str, classname:str, encoder_class:nn.Module, decoder_class:nn.Module,
                   img_size:tuple, transform:callable, nepochs:int=30, scheduler_step_size = 10, scheduler_gamma = 0.5, verbose:bool=True,
                   lr_seq = [0.001], ld_seq =[4, 8, 16], bs_seq = [10, 20], beta_seq = [0.5, 1.0], niter:int=3,
                   ) -> pd.DataFrame:
        """
        Conducts a grid search over specified hyperparameters for training the VAESIMCA model.

        Parameters
        ----------
        train_path : str
            The path to the training data.
        test_path : str
            The path to the testing data.
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
            The number of epochs to train for each configuration.
        scheduler_step_size : int, optional
            Step size for scheduler to adjust learning rate.
        scheduler_gamma : float, optional
            Gamma parameter for scheduler to adjust learning rate.
        verbose : bool, optional
            If True, prints detailed logs during the grid search.

        Parameters for optimization
        ---------------------------

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
        DataFrame
            Contains values of parameters as well as prediction results (number of objects accepted
            by the model) for each class of training and test sets for each iteration and for each
            combination of the parameters.

            iter : iteration ID
            comb : combination ID (identifies unique combination of parameters for optimization)
            set  : name of the set ("train" or "test")
            class : name of the class (results are computed separately for each target and alternative classes)
            beta : beta value used to get this results
            bs : batch size used to get this results
            lr : learning rate used to get this results
            ld : latent space dimension used to get this results
            n : total number of images in this class
            in : number of images accepted as members
            inp : percent of images accepted as members

        """

        def show_progress(i, p, n, sens, spec, eff):
            """ method to print current progress state """
            pct = p / n
            scaled_p = int(50 * pct)  # Scale p to fit within the range [0, 50]
            symbols_line = f"Iteration {i:2d}: [" + "#" * scaled_p + " " * (50 - scaled_p) + f"] {pct*100:5.1f}% - ({sens:.3f}/{spec:.3f}/{eff:.3f})"
            sys.stdout.write("\r" + symbols_line)
            sys.stdout.flush()

        # get test set classes and combine with training class
        test_classes = [f for f in os.listdir(test_path) if os.path.isdir(test_path + "/" + f)]

        if len(test_classes) < 1:
            raise ValueError("No subdirectories found in the path specified by 'test_path' parameter.")

        # combine all arguments and calculate number of their combinations
        args = [beta_seq, bs_seq, lr_seq, ld_seq]
        nargs = len(args)
        ncombs = functools.reduce(lambda l, b: l * len(b), args, 1)

        # get all combinations of the parameters to optimize and repeat for each class
        params = np.array(np.meshgrid(beta_seq, bs_seq, lr_seq, ld_seq))
        params = params.reshape([nargs, ncombs])


        # global lists for results
        g_iter = []
        g_comb = []
        g_beta = []
        g_bs = []
        g_lr = []
        g_ld = []
        g_set = []
        g_class = []
        g_n = []
        g_in = []
        g_inp = []

        # best model
        sens_best = (0, [])
        spec_best = (0, [])
        eff_best = (0, [])

        # global counter for results
        n = 0

        if verbose:
            print(f"\nRun grid search with {ncombs} combinations x {niter} iterations:")
            print("-----------------------------------------------------------------")

        # loop over iterations
        for i in range(0, niter):

            # loop over parameters:
            for p in range(0, ncombs):

                if verbose:
                    show_progress(i, p, ncombs, sens_best[0], spec_best[0], eff_best[0]),

                beta, batch_size, lr, ld = params[:, p]

                m = VAESIMCA(encoder_class=encoder_class, decoder_class=decoder_class, classname=classname,
                             img_size=img_size, latent_dim=int(ld), transform=transform)

                try:
                    m.fit(data_path=train_path, nepochs=nepochs, beta=beta, lr=lr, batch_size=int(batch_size),
                         scheduler_gamma=scheduler_gamma, scheduler_step_size=scheduler_step_size, verbose=False)
                except Exception as error:
                    print("A critical problem occured when training the model with following parameters:")
                    print(f"lr = {lr} ld = {ld} beta = {beta} batch size = {batch_size}")
                    print(error)

                rc = m.predict(train_path).stat()
                rt = m.predict(test_path).stat()

                # save results for the calibration/training set
                g_iter.append(i)
                g_comb.append(p)
                g_beta.append(beta)
                g_bs.append(batch_size)
                g_lr.append(lr)
                g_ld.append(ld)
                g_set.append("train")
                g_class.append(classname)
                g_n.append(rc[classname][0])
                g_in.append(rc[classname][1])
                g_inp.append(rc[classname][1] / rc[classname][0])
                n = n + 1

                # save results for each class from the test set and accumulate results for alternative classes
                alt_in = 0
                alt_n = 0
                for c in test_classes:
                    g_iter.append(i)
                    g_comb.append(p)
                    g_beta.append(beta)
                    g_bs.append(batch_size)
                    g_lr.append(lr)
                    g_ld.append(ld)
                    g_set.append("test")
                    g_class.append(c)
                    g_n.append(rt[c][0])
                    g_in.append(rt[c][1])
                    g_inp.append(rt[c][1] / rt[c][0])
                    n = n + 1

                    if c == classname:
                        sens = rt[c][1] / rt[c][0]
                    else:
                        alt_in += rt[c][1]
                        alt_n += rt[c][0]

                # compute overall specificity and efficiency
                spec = 1 - alt_in / alt_n
                eff = math.sqrt(sens * spec)

                # adjust best FoM values
                if (spec > spec_best[0]):
                    spec_best = (spec, [i, lr, ld, beta, batch_size])
                if (sens > sens_best[0]):
                    sens_best = (sens, [i, lr, ld, beta, batch_size])
                if (eff > eff_best[0]):
                    eff_best = (eff, [i, lr, ld, beta, batch_size])

            if verbose:
                show_progress(i, ncombs, ncombs, sens_best[0], spec_best[0], eff_best[0]),
                print("")

        if verbose:
            print("-----------------------------------------------------------------\n")
            print(f"Best sensitivity:  {sens_best[0]:.3f} - lr={sens_best[1][1]:6f} ld={sens_best[1][2]:4.0f} beta={sens_best[1][3]:.1f} bs={sens_best[1][4]:4.0f}")
            print(f"Best specificity:  {spec_best[0]:.3f} - lr={spec_best[1][1]:6f} ld={spec_best[1][2]:4.0f} beta={spec_best[1][3]:.1f} bs={spec_best[1][4]:4.0f}")
            print(f"Best efficiency:   {eff_best[0]:.3f} - lr={eff_best[1][1]:6f} ld={eff_best[1][2]:4.0f} beta={eff_best[1][3]:.1f} bs={eff_best[1][4]:4.0f}")

        res_df = pd.DataFrame({"iter": g_iter, "comb": g_comb, "set": g_set, "class": g_class, "beta": g_beta, "bs": g_bs, "lr": g_lr, "ld": g_ld, "n": g_n,  "in": g_in, "inp": g_inp})
        return res_df



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
    plotAcceptance(plt, dolog=False, colors=None, markers=None)
        Plots an acceptance graph showing the explained and residual distances and decision boundary.
    """
    def __init__(self, img_size:tuple, Z:np.ndarray, E:np.ndarray, T:np.ndarray, U:np.ndarray, q:np.array, h:np.array,
                 qParams:tuple, hParams:tuple, fParams:tuple, alpha:float, labels:list, class_labels:list, classes:list,
                 crit:str):

        n = len(q)
        h0, Nh = hParams
        q0, Nq = qParams
        f0, Nf = fParams

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

        if crit == "f":
            self.regular = self.f < getlimits(f0, Nf, alpha=alpha)
        elif crit == "q":
            self.regular = self.q < getlimits(q0, Nq, alpha=alpha)
        elif crit == "h":
            self.regular = self.h < getlimits(h0, Nh, alpha=alpha)
        else:
            raise ValueError("Wrong value for parameter 'crit' (must be 'f', 'q', or 'h').")


    def stat(self):
        """
        Computes statistics for the number of data points falling within expected ranges per class.

        Returns
        -------
        dict
            A dictionary with classes as keys and lists of counts: total, within range, and out of range.
        """
        stat = {}
        for c in self.classes:
            decisions = [self.regular[i] for i, label in enumerate(self.class_labels) if label == c]
            total = len(decisions)
            accepted = sum(decisions)
            rejected = total - accepted
            stat[c] = [total, accepted, rejected]
        return stat


    def summary(self):
        """
        Prints a summary of the statistics of the model predictions.

        Displays the number of data points per class and how many are accepted/rejected by the model.
        """
        stats = self.stat()
        slen = max(6, max(len(x) for x in self.classes) + 1)
        dlen = max(4, int(math.log10(self.n)) + 1)
        print(f"\n{'class':<{slen}s} {'n':>{dlen}s} {'in':>{dlen}s} {'in (%)':>6s} {'out':>{dlen}s} {'out (%)':>7s}")
        print(f"{'':->{slen + 3 * dlen + 18}s}")
        for c, (total, in_range, out_of_range) in stats.items():
            in_pct = 100 * in_range / total if total > 0 else 0
            out_pct = 100 * out_of_range / total if total > 0 else 0
            print(f"{c:<{slen}s} {total:{dlen}d} {in_range:{dlen}d} {in_pct:6.1f} {out_of_range:{dlen}d} {out_pct:7.1f}")


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


    def plotDistance(self, plt:matplotlib.pyplot, distance:str="q", colors:list=None, legend_loc:int=2,
                    show_boundaries:bool = False, show_labels:bool = False):
        """
        Plots the specified type of distance for the data points.

        Parameters
        ----------
        plt : matplotlib.pyplot
            Matplotlib plot module.
        distance : str, optional
            The type of distance to plot ('q', 'h', or 'f').
        colors : list, optional
            List of colors for each class.
        legend_loc : int, optional
            Location of the legend in the plot.
        show_boundaries : bool, optional
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
        if colors is None:
            colors = COLORS
        if len(colors) < len(self.classes):
            raise ValueError(f"Colors for each of the {len(self.classes)} must be provided.")

        params = {'q': self.qParams, 'h': self.hParams, 'f': self.fParams}
        distances = {'q': self.q, 'h': self.h, 'f': self.f}[distance]
        title_map = {'q': "Residual", 'h': "Explained", 'f': "Full"}

        start = 0
        for i, c in enumerate(self.classes):
            y = [distances[j] for j, label in enumerate(self.class_labels) if label == c]
            x = range(start, start + len(y))
            plt.bar(x, y, color=colors[i % len(colors)], label=c)
            if show_labels:
                labels = [self.labels[j] for j, label in enumerate(self.class_labels) if label == c]
                plotlabels(plt, x, y, labels)
            start += len(y)

        if show_boundaries:
            u0, Nu = params[distance]
            lim = getlimits(u0, Nu, alpha = self.alpha)
            xr = plt.xlim()
            plt.plot(xr, [lim, lim], 'k--', linewidth=0.5)

        plt.legend(loc=legend_loc)
        plt.title(f"{title_map[distance]} distance")
        plt.ylabel(f"{distance}-distance")
        plt.xlabel("Objects")


    def plotAcceptance(self, plt:matplotlib.pyplot, dolog:bool=False, colors:list=None, markers:list=None,
                       legend_loc:str = "best", show_labels:bool = False):
        """
        Plots an acceptance graph showing scaled explained and residual distances and the decision boundary.

        Parameters
        ----------
        plt : matplotlib.pyplot
            Matplotlib plot module.
        dolog : bool, optional
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
        if colors is None:
            colors = COLORS

        if len(colors) < len(self.classes):
            raise ValueError(f"Colors for each of the {len(self.classes)} must be provided.")
        if len(markers) < len(self.classes):
            raise ValueError(f"Markers for each of the {len(self.classes)} must be provided.")

        h0, Nh = self.hParams
        q0, Nq = self.qParams
        f0, Nf = self.fParams

        h_scaled = np.log1p(self.h / h0) if dolog else self.h / h0
        q_scaled = np.log1p(self.q / q0) if dolog else self.q / q0

        for i, c in enumerate(self.classes):
            class_points = [(h_scaled[j], q_scaled[j], self.labels[j]) for j, label in enumerate(self.class_labels) if label == c]
            cx, cy, cl = zip(*class_points)
            plt.scatter(cx, cy, label=c, marker=markers[i], edgecolors=colors[i], facecolors='none')
            if show_labels:
                plotlabels(plt, cx, cy, cl)


        # show decision and outliers boundaries
        fCritE = getlimits(f0, Nf, alpha = self.alpha)
        xqeMax = fCritE / Nh
        xqe = np.linspace(0, xqeMax, 200)
        yqe = (fCritE - xqe * Nh) / Nq

        if dolog:
            xqe = np.log1p(xqe)
            yqe = np.log1p(yqe)

        plt.plot(xqe, yqe, 'k--', linewidth=0.5)
        plt.legend(loc = legend_loc)
        plt.grid(color = "#e0e0e0", linestyle = ":")
        plt.title("Acceptance Plot")

        if dolog:
            plt.xlabel("Explained distance, log(1 + h/h0)")
            plt.ylabel("Residual distance, log(1 + q/q0)")
        else:
            plt.xlabel("Explained distance, h/h0")
            plt.ylabel("Residual distance, q/q0")


    def plotError(self, plt:matplotlib.pyplot, classname:str, object_label:str):
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

        if not classname in self.classes:
            raise ValueError("Can not find class with name '{classname}' in this result object.")

        ind = [i for i in range(self.n) if self.class_labels[i] == classname and self.labels[i] == object_label]
        if len(ind) < 1:
            raise ValueError("Can not find object with label '{object_label}' among elements of class '{classname}'")

        mn = np.min(self.E)
        mx = np.max(self.E)
        e = self.E[ind[0], :].reshape(self.img_size[0], self.img_size[1])

        plt.imshow(e)
        plt.clim([mn, mx])
        plt.colorbar()
        plt.title(f"{classname}:{object_label}")

