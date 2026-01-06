"""
VAE-SIMCA: Variational Autoencoder with DD-SIMCA for One-Class Classification
"""

__version__ = "1.1.0"
__author__ = "Sergey Kucheryavskiy"
__email__ = "svkucheryavski@gmail.com"

from .vaesimca import VAESIMCA, VAESIMCARes, get_distparams

__all__ = ["VAESIMCA", "VAESIMCARes", "get_distparams"]