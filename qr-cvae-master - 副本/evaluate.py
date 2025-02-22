import os

from utils import get_kdeplot

from universal_divergence import estimate
import yaml
import argparse
import numpy as np
from utils import vae_datasets
from models import *
import pytorch_lightning as pl

parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/sim_vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

save_dir = os.path.join(config['logging_params']['save_dir'], config['exp_params']['name'])
model_dir = os.path.join(save_dir,
                         config['model_params']['name'],
                         f"seed_{config['exp_params']['manual_seed']}",
                         'checkpoints', 'last.ckpt')

# For reproducibility
pl.seed_everything(config['exp_params']['manual_seed'], True)
model = vae_models[config['model_params']['name']](**config['model_params'])

checkpoint = torch.load(model_dir)
model_weights = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}

model.load_state_dict(model_weights)

data = vae_datasets[config["data_params"]['name']](**config["data_params"], pin_memory=len(config['trainer_params']['devices']) != 0)
data.setup()

def calculate_kl_divergence(true_samples, generated_samples):
    kl_divergence = estimate(true_samples, generated_samples)
    return kl_divergence

with torch.no_grad():
    data_sampled = model.sample(500, current_device='cpu')
    data_origin = data.train_dataset.tensors[0]
    kld = calculate_kl_divergence(data_origin, data_sampled)

print(f"KL Divergence({config['model_params']['name']}): {kld}")

output_dir = os.path.join(save_dir,
                         config['model_params']['name'],
                        f"seed_{config['exp_params']['manual_seed']}",
                        )  
    
get_kdeplot(data_sampled.numpy(), output_dir, "sampled_data")
get_kdeplot(data_origin.numpy(), output_dir, "origin_data")

