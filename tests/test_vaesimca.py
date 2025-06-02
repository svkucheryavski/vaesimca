import unittest
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch import nn, sigmoid
from torch.nn import functional as F
from torch_lr_finder import LRFinder

from vaesimca import VAESIMCA, VAESIMCARes, getdistparams

class TestGetDistParams(unittest.TestCase):
    def test_normal_distribution(self):
        """ Test with a normal distribution of values """
        np.random.seed(0)
        u = np.random.normal(loc=0, scale=1, size=10000)
        mean, nu = getdistparams(u**2)
        self.assertAlmostEqual(mean, 1, places=1)
        self.assertAlmostEqual(nu, 1, places=1)

    def test_single_value_array(self):
        """ Test with an array where all elements are the same """
        u = np.array([5] * 100)
        mean, nu = getdistparams(u)
        self.assertEqual(mean, 5)
        self.assertEqual(nu, 1)


class TestVAESIMCA(unittest.TestCase):
    def setUp(self):
        # Dummy encoder and decoder classes
        class DummyEncoder(torch.nn.Module):
            def __init__(self, img_size, latent_dim):
                super().__init__()
                self.fc = torch.nn.Linear(np.prod(img_size), latent_dim)
            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x), torch.zeros_like(self.fc(x))

        class DummyDecoder(torch.nn.Module):
            def __init__(self, img_size, latent_dim):
                super().__init__()
                self.fc = torch.nn.Linear(latent_dim, np.prod(img_size))
            def forward(self, x):
                return self.fc(x).view(-1, *img_size)

        img_size = (28, 28, 1)
        latent_dim = 10
        transform = torch.nn.Identity()  # No transformation
        self.encoder = DummyEncoder
        self.decoder = DummyDecoder
        self.model = VAESIMCA(DummyEncoder, DummyDecoder, 'class_name', img_size, latent_dim, transform)

    def test_invalid_img_size(self):
        """ Test initialization with invalid image sizes """
        with self.assertRaises(ValueError):
            VAESIMCA(self.encoder, self.decoder, 'class_name', (28, 28, 0), 10, torch.nn.Identity())

    def test_latent_dim_greater_than_pixels(self):
        """ Test initialization with latent dimension greater than pixels """
        with self.assertRaises(ValueError):
            VAESIMCA(self.encoder, self.decoder, 'class_name', (28, 28, 1), 28 * 28 + 1, torch.nn.Identity())

    def test_zero_latent_dim(self):
        """ Test initialization with zero latent dimension """
        with self.assertRaises(ValueError):
            VAESIMCA(self.encoder, self.decoder, 'class_name', (28, 28, 1), 0, torch.nn.Identity())

class TestVAESIMCARes(unittest.TestCase):
    def setUp(self):
        self.Z = np.array([[1, 2, 3], [1, 2, 3]])
        self.E = np.array([[1, 2, 3], [1, 2, 3]])
        self.T = np.array([[1, 2, 3], [1, 2, 3]])
        self.U = np.array([[1, 2, 3], [1, 2, 3]])
        self.q = np.array([0.1, 0.2, 0.3])
        self.h = np.array([0.4, 0.5, 0.6])
        self.qParams = (0.2, 1)
        self.hParams = (0.5, 1)
        self.fParams = (0.3, 1)
        self.alpha = 0.05
        self.labels = ['sample1', 'sample2', 'sample3']
        self.class_labels = ['A', 'A', 'B']
        self.classes = ['A', 'B']

    def test_statistical_output(self):
        """ Test the statistics calculation """

        result = VAESIMCARes((2, 3), self.Z, self.E, self.T, self.U, self.q, self.h, self.qParams, self.hParams, self.fParams, self.alpha, self.labels, self.class_labels, self.classes, "f")
        stats = result.stat()
        self.assertEqual(stats['A'][0], 2)  # Total count in class A
        self.assertEqual(stats['B'][0], 1)  # Total count in class B

class TestVAESIMCAFull(unittest.TestCase):

    def setUp(self):

        class Encoder(nn.Module):
            def __init__(self, img_size, latent_dim):
                super(Encoder, self).__init__()
                img_width, img_height, nchannels = img_size
                self.conv1 = nn.Conv2d(nchannels, 32, kernel_size=4, stride=2, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
                self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
                self.fc_mu = nn.Linear(256 * (img_width // 16) * (img_height // 16), latent_dim)
                self.fc_logvar = nn.Linear(256 * (img_width // 16) * (img_height // 16), latent_dim)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.relu(self.conv2(x))
                x = F.relu(self.conv3(x))
                x = F.relu(self.conv4(x))
                x = x.view(x.size(0), -1) # flatten
                mu = self.fc_mu(x)
                logvar = self.fc_logvar(x)
                return mu, logvar

        # define the decoder class
        class Decoder(nn.Module):
            def __init__(self, img_size, latent_dim):
                super(Decoder, self).__init__()
                img_width, img_height, nchannels = img_size
                self.img_width = img_width
                self.img_height = img_height
                self.latent_dim = latent_dim
                self.fc = nn.Linear(latent_dim, 256 * (img_width // 16) * (img_height // 16))
                self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
                self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
                self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
                self.deconv4 = nn.ConvTranspose2d(32, nchannels, kernel_size=4, stride=2, padding=1)

            def forward(self, z):
                z = self.fc(z)
                z = z.view(z.size(0), 256, (self.img_width // 16), (self.img_height // 16))
                z = F.relu(self.deconv1(z))
                z = F.relu(self.deconv2(z))
                z = F.relu(self.deconv3(z))
                z = sigmoid(self.deconv4(z))
                return z

        self.encoder = Encoder
        self.decoder = Decoder
        self.img_size = (128, 128, 1)
        self.transform = transforms.Compose([
            transforms.Resize([self.img_size[0], self.img_size[1]]),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def test_lr(self):
        train_path = "demo/images_simulated/train"
        test_path = "demo/images_simulated/test"
        classname = "target"

        latent_dim = 6
        nepochs = 30
        lr = 0.0001

        # initialize the model object
        m = VAESIMCA(encoder_class=self.encoder, decoder_class=self.decoder, classname = classname,
                        img_size=self.img_size, latent_dim=latent_dim, transform=self.transform,
                        device = "cpu")

        # train the model
        lrf1 = m.findlr(data_path=train_path)
        lr1 = lrf1.history
        print(pd.DataFrame(lr1))


    # def test_image_csv(self):
    #     train_path = "demo/images_simulated_train.csv"
    #     test_path = "demo/images_simulated_test.csv"
    #     classname = "target"

    #     latent_dim = 6
    #     nepochs = 30
    #     lr = 0.0001

    #     # initialize the model object
    #     m = VAESIMCA(encoder_class=self.encoder, decoder_class=self.decoder, classname = classname,
    #                     img_size=self.img_size, latent_dim=latent_dim, transform=self.transform)

    #     # train the model
    #     m.fit(data_path=train_path, nepochs=nepochs, lr = lr)
    #     rc = m.predict(data_path=test_path)
    #     rc.summary()

    # def test_image_dir(self):
    #     train_path = "demo/images_simulated/train"
    #     test_path = "demo/images_simulated/test"
    #     classname = "target"

    #     latent_dim = 6
    #     nepochs = 30
    #     lr = 0.0001

    #     # initialize the model object
    #     m = VAESIMCA(encoder_class=self.encoder, decoder_class=self.decoder, classname = classname,
    #                     img_size=self.img_size, latent_dim=latent_dim, transform=self.transform)

    #     # train the model
    #     m.fit(data_path=train_path, nepochs=nepochs, lr = lr)
    #     rc = m.predict(data_path=test_path)
    #     rc.summary()


# Run the tests
if __name__ == '__main__':
    unittest.main()


