from typing import Sequence
from .types_ import *
from .base import BaseVAE
from utils import clamp
from torch import distributions as D

import torch
from torch import nn
from torch.nn import functional as F

# def check_nan_in_tensors(ordered_dict):
#     """
#     检查 OrderedDict 中的 Tensor 是否包含 NaN, 并返回包含 NaN 的键列表。

#     参数:
#         ordered_dict (OrderedDict): 一个键值对, 其中值是 torch.Tensor。

#     返回:
#         list: 包含 NaN 的键列表。
#     """

#     nan_keys = []
#     for key, tensor in ordered_dict.items():
#         if not isinstance(tensor, torch.Tensor):
#             raise ValueError(f"键 {key} 的值不是 torch.Tensor 类型！")
        
#         # 检查是否包含 NaN
#         if torch.isnan(tensor).any():
#             nan_keys.append(key)
    
#         if torch.isinf(tensor).any():
#             nan_keys.append(key)
#     return nan_keys

class SimCVAE(BaseVAE):
    '''
    This is mainly adapted from 
    https://github.com/SkafteNicki/john/blob/master/toy_vae.py
    '''
    def __init__(self, latent_dim=2, hidden_dim=50, *args, **kwargs) -> None:
        super(SimCVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(4, hidden_dim), 
            nn.Tanh()
        )  # 第一隐藏层
        self.z_mean = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )  # z 的均值
        self.z_log_var = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim)
        )  # z 的对数方差
        
        self.r_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )  # r 的均值
        self.r_log_var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )  # r 的对数方差

        # p(z|r) 的生成器部分
        self.r_encoder = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh()
        )  # 第一隐藏层
        self.pz_mean = nn.Sequential(
            nn.Linear(2, latent_dim)
        )  # p(z|r) 的均值
        self.pz_log_var = nn.Sequential(
            nn.Linear(2, latent_dim)
        )  # p(z|r) 的对数方差

        # 解码器部分
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)  # 解码第一层隐藏层
        self.decoder_fc2 = nn.Linear(hidden_dim, 4 * 2)             # 重建输出层

    def encode(self, x):
        """
        对输入 x 进行编码，返回 z 和 r 的均值与对数方差。
        """
        x = self.encoder(x)  # 编码器第一层
        z_mean = self.z_mean(x)  # z 的均值
        z_log_var = self.z_log_var(x)  # z 的对数方差
        r_mean = self.r_mean(x)  # r 的均值
        r_log_var = self.r_log_var(x)  # r 的对数方差
        return z_mean, z_log_var, r_mean, r_log_var

    def decode(self, z):
        z = torch.tanh(self.decoder_fc1(z))               # 第一层
        outputs = self.decoder_fc2(z)                     # 输出: 4 + 4
        return outputs
    
    def forward(self, inputs: torch.tensor, *args, **kwargs) -> Sequence[torch.tensor]:
        '''
        :return: [decode, inputs, mu, var, r_mean, r_log_var, pz_mean, pz_logvar, labels]
        '''
        
        mu, log_var, r_mean, r_log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        
        pr_sample = self.reparameterize(r_mean, r_log_var)
        pr_sample = self.r_encoder(pr_sample)
        pz_mean = self.pz_mean(pr_sample)
        pz_log_var = self.pz_log_var(pr_sample)

        return  [self.decode(z), inputs, mu, log_var, r_mean, r_log_var, pz_mean, pz_log_var, kwargs['labels']]
    
    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x 2]
        :param log_var: (Tensor) Log variance of the latent Gaussian [B x 2]
        :return: (Tensor) [B x 2]
        """
        std = torch.exp(0.5 * clamp(log_var))  
        eps = torch.randn_like(std)    
        return mu + eps * std        

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x 4]
        :return: (Tensor) [B x 4]
        """
        temp = self.forward(x, **kwargs)[0]
        
        return self.reparameterize(temp[:, :4], temp[:, 4:])
    
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
        try:
            labels = kwargs['labels']
            labels = self.r_encoder(labels)
            pz_mean = self.pz_mean(labels)
            pz_log_var = self.pz_log_var(labels)
            z = self.reparameterize(pz_mean, pz_log_var)
            print('use cond!')
        except:
            z = torch.randn(num_samples,
                            self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        samples = self.reparameterize(samples[:, :4], samples[:, 4:])
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
        samples = self.reparameterize(samples[:, :4], samples[:, 4:])
        return samples


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
        x_mu, x_log_var = args[0][:, :4], args[0][:, 4:]
        inputs = args[1]
        mu = args[2]
        log_var = args[3]
        
        r_mean = args[4]
        r_log_var = args[5]
        pz_mean = args[6]
        pz_log_var = args[7]
        labels = args[8]

        kld_weight = kwargs['kld_weight']
            
        p_dist = D.Independent(D.Normal(x_mu, torch.exp(x_log_var).sqrt()+epsilon), 1)
        recons_loss = p_dist.log_prob(inputs) # NOTE: changes here
        # recons_loss = - 0.5 * ((x_mu - inputs)**2 / (torch.exp((x_log_var))) + x_log_var)
        # recons_loss = recons_loss.sum(1)
        
        # kld_loss = 1 + log_var - pz_log_var \
        #       - torch.square(mu - pz_mean) / (torch.exp((pz_log_var))) \
        #       - torch.exp((log_var)) / (torch.exp((pz_log_var)))
        # kld_loss = -0.5 * kld_loss.sum(dim=1)
        
        prior = D.Independent(D.Normal(pz_mean, torch.exp(0.5*pz_log_var)), 1)
        q_dist = D.Independent(D.Normal(mu, torch.exp(0.5*log_var)), 1)
        z = q_dist.rsample()
        
        kld_loss = q_dist.log_prob(z) - prior.log_prob(z)
                
        label_loss = - 0.5 * ((r_mean - labels)**2 / (torch.exp((r_log_var))) + r_log_var)
        label_loss = label_loss.mean(dim=1)
        
        elbo = recons_loss - kld_weight*kld_loss + 1.0 * label_loss

        loss = -elbo.mean()

        return {'loss': loss, 
                'Reconstruction_Loss':-recons_loss.detach().mean(), 
                'KLD':kld_loss.detach().mean(),
                'Label_Loss': -label_loss.detach().mean(),
                'sigmaX': (x_log_var.detach()/2).exp().mean(),
                }
