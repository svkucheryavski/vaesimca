import unittest
import torch
import numpy as np
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

# Run the tests
if __name__ == '__main__':
    unittest.main()
