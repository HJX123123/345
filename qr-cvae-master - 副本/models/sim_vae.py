from typing import Sequence
from .types_ import *
from .base import BaseVAE
from torch import distributions as D

import torch
from torch import nn
from torch.nn import functional as F

class SimVAE(BaseVAE):
    '''
    This is mainly adapted from 
    https://github.com/SkafteNicki/john/blob/master/toy_vae.py
    '''
    def __init__(self, latent_dim=2, hidden_dim=50, *args, **kwargs) -> None:
        super(SimVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.enc_mu = nn.Sequential(nn.Linear(4, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, latent_dim))
        self.enc_var = nn.Sequential(nn.Linear(4, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, latent_dim),
                                     nn.Softplus())
        self.dec_mu = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 4))
        self.dec_var = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(hidden_dim, 4),
                                     nn.Softplus())
        
    def encode(self, inputs) -> Sequence[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        
        :param input: (Tensor) Input tensor to encoder [N x 4]
        :return: (Tensor) List of latent codes [N x 2] (Mean and Variance)
        """
        
        return self.enc_mu(inputs), self.enc_var(inputs)
    
    def forward(self, inputs: torch.tensor, *args, **kwargs) -> Sequence[torch.tensor]:
        mu, var = self.encode(inputs)
        z = self.reparameterize(mu, var)
        return  [self.decode(z), inputs, mu, var]
    
    def reparameterize(self, mu: Tensor, var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x 2]
        :param var: (Tensor) Standard deviation of the latent Gaussian [B x 2]
        :return: (Tensor) [B x 2]
        """
        std = torch.sqrt(var)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def decode(self, inputs: torch.tensor):
        return self.dec_mu(inputs), self.dec_var(inputs)
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x 4]
        :return: (Tensor) [B x 4]
        """

        return self.reparameterize(*self.forward(x)[0])
    
    def sample(self,
               num_samples:int,
               current_device, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        samples = self.reparameterize(*samples)
        return samples

    def sample_cond(self,
                num_samples:int,
                current_device, **kwargs) -> Tensor:
          """
          Samples from the latent space and return the corresponding
          image space map.
          :param num_samples: (Int) Number of samples
          :param current_device: (Int) Device to run the model
          :return: (Tensor)
          """
          return self.sample(num_samples, current_device, **kwargs)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args: recons, inputs, mu, log_var
        :param kwargs:
        :return: loss
        """
        epsilon=1e-5
        x_mu, x_var = args[0]
        inputs = args[1]
        mu = args[2]
        var = args[3]

        kld_weight = kwargs['kld_weight']
        
        q_dist = D.Independent(D.Normal(mu, var.sqrt()+epsilon), 1)
        z = q_dist.rsample()
        
        p_dist = D.Independent(D.Normal(x_mu, x_var.sqrt()+epsilon), 1)
        recons_loss = p_dist.log_prob(inputs)
        
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        
        kld_loss = q_dist.log_prob(z) - prior.log_prob(z)
        elbo = recons_loss - kld_weight*kld_loss

        loss = -elbo.mean()
        return {'loss': loss, 
                'Reconstruction_Loss':-recons_loss.detach().mean(), 
                'KLD':kld_loss.detach().mean(),
                'sigmaX':x_var.detach().sqrt().mean(),}
