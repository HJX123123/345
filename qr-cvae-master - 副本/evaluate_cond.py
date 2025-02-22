import os
import random
from universal_divergence import estimate
import yaml
import argparse
import numpy as np

from utils import get_kdeplot
from utils import vae_datasets
from models import *
import pytorch_lightning as pl

import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):

    current_run_dir = hydra.utils.to_absolute_path(".")
    save_dir = os.getcwd()
    print("log dir: ", save_dir)
    os.chdir(current_run_dir)
    
    model_dir = os.path.join(save_dir, 'checkpoints')
    model_name = os.listdir(model_dir)
    model_name = list(filter(lambda x: x!='last.ckpt', model_name))
    model_name = random.choice(model_name)
    model_dir = os.path.join(save_dir, 'checkpoints', model_name)
    # model_dir = os.path.join(save_dir,'checkpoints', 'last.ckpt')

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
        data_sampled = model.sample_cond(500, current_device='cpu', labels = data.train_dataset.tensors[1])
        data_origin = data.train_dataset.tensors[0]
        kld = calculate_kl_divergence(data_origin, data_sampled)

    print(f"KL Divergence({config['model_params']['name']}): {kld}")

    output_dir = save_dir
    np.savetxt(output_dir + '/kld.csv', np.array([kld]), delimiter=',')
        
    get_kdeplot(data_sampled.numpy(), output_dir, "sampled_data")
    get_kdeplot(data_origin.numpy(), output_dir, "origin_data")

if __name__ == "__main__":
    main()