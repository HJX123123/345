from .base import *
from .sim_vae import *
from .sim_qrvae import SimQRVAE
from .sim_cvae import SimCVAE
from .sim_cqrvae import SimCQRVAE

# Aliases
# VAE = VanillaVAE
# GaussianVAE = VanillaVAE
# CVAE = ConditionalVAE
# GumbelVAE = CategoricalVAE

vae_models = {'SimVAE':SimVAE,
              'SimQRVAE':SimQRVAE,
              'SimCVAE':SimCVAE,
              'SimCQRVAE':SimCQRVAE,
              
}

# vae_models = {'HVAE':HVAE,
#               'LVAE':LVAE,
#               'IWAE':IWAE,
#               'SWAE':SWAE,
#               'MIWAE':MIWAE,
#               'VQVAE':VQVAE,
#               'DFCVAE':DFCVAE,
#               'DIPVAE':DIPVAE,
#               'BetaVAE':BetaVAE,
#               'InfoVAE':InfoVAE,
#               'WAE_MMD':WAE_MMD,
#               'VampVAE': VampVAE,
#               'GammaVAE':GammaVAE,
#               'MSSIMVAE':MSSIMVAE,
#               'JointVAE':JointVAE,
#               'BetaTCVAE':BetaTCVAE,
#               'FactorVAE':FactorVAE,
#               'LogCoshVAE':LogCoshVAE,
#               'VanillaVAE':VanillaVAE,
#               'ConditionalVAE':ConditionalVAE,
#               'CategoricalVAE':CategoricalVAE}