from typing import Sequence
from .types_ import *
from .base import BaseVAE
from utils import clamp
from torch import distributions as D

import torch
from torch import nn
from torch.nn import functional as F


class SimQRVAE(BaseVAE):
    '''
    This is mainly adapted from 
    https://github.com/SkafteNicki/john/blob/master/toy_vae.py
    '''
    def __init__(self, latent_dim=2, hidden_dim=50, *args, **kwargs) -> None:
        super(SimQRVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.enc_mu = nn.Sequential(nn.Linear(4, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, latent_dim * 2))

        self.dec_mu = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, 8))

    def encode(self, inputs) -> Sequence[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        
        :param input: (Tensor) Input tensor to encoder [N x 4]
        :return: (Tensor) List of latent codes [N x 2] (Mean and Variance)
        """
        temp = self.enc_mu(inputs)
        return temp[:, :self.latent_dim],temp[:, self.latent_dim:]
    
    def forward(self, inputs: torch.tensor, *args, **kwargs) -> Sequence[torch.tensor]:
        '''
        :return Q50, Q15, inputs, mu, log_var
        '''
        mu, log_var = self.encode(inputs)
        
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), inputs, mu, log_var]
    
    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x 2]
        :param log_var: (Tensor) Standard deviation of the latent Gaussian [B x 2]
        :return: (Tensor) [B x 2]
        """
        std = torch.sqrt(torch.exp(log_var))
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def decode(self, inputs: torch.tensor):
        quantiles = self.dec_mu(inputs)
        return quantiles[:, :4],quantiles[:, 4:]
    
    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x 4]
        :return: (Tensor) [B x 4]
        """

        mu, sigma = self.forward(x)[0]
        sigma = mu-sigma
        sigma[sigma<0] = 0
        samples = self.reparameterize(mu, torch.log(sigma ** 2))
        return samples
    
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
        mu = samples[0]
        sigma = samples[0]-samples[1]
        sigma[sigma<0] = 0
        
        samples = self.reparameterize(mu, torch.log(sigma ** 2))
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
        q50, q15 = args[0]
        inputs = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['kld_weight']
        
        q_dist = D.Independent(D.Normal(mu, torch.exp(clamp(log_var)).sqrt()), 1)
        z = q_dist.rsample()

        loss_50 = torch.sum(torch.max(0.15 * (inputs-q15), (0.15 - 1) * (inputs-q15)).view(-1, 4),(1))
        loss_15 = torch.sum(torch.max(0.5 * (inputs-q50), (0.5 - 1) * (inputs-q50)).view(-1, 4),(1))
        recons_loss = (loss_50+loss_15)/2
        
        prior = D.Independent(D.Normal(torch.zeros_like(z),
                                       torch.ones_like(z)), 1)
        
        kld_loss = q_dist.log_prob(z) - prior.log_prob(z)
        # recons -> -log(p(x|z)) which is different from the SimVAE
        elbo = -recons_loss - kld_weight*kld_loss

        loss = -elbo.mean()
        
        
        return {'loss': loss, 
                'Reconstruction_Loss':recons_loss.detach().mean(), 
                'KLD':kld_loss.detach().mean(),
                'sigmaX':(q50.detach()-q15.detach()).mean()}
