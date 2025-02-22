

from typing import Optional
import numpy as np
import torch

from .dataset import VAEDataset


def get_sim_cond_datset(data_dir: str, split: str):
    if split == 'train':
        data = np.loadtxt(data_dir+'train.csv', delimiter=',')
    elif split == 'test':
        data = np.loadtxt(data_dir+'test.csv', delimiter=',')       
    
    X = data[:, :-2]
    y = data[:, -2:]
    
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), 
                                             torch.tensor(y, dtype=torch.float32))
    return dataset

class SimCondDataset(VAEDataset):
    def __init__(self, data_path: str, train_batch_size: int = 8, val_batch_size: int = 8, patch_size= (256, 256), num_workers: int = 0, pin_memory: bool = False, **kwargs):
        super().__init__(data_path, train_batch_size, val_batch_size, patch_size, num_workers, pin_memory, **kwargs)
        
    def setup(self, stage: Optional[str] = None) -> None:
        
        self.train_dataset = get_sim_cond_datset(
            self.data_dir,
            split='train',
        )
        
        self.val_dataset = get_sim_cond_datset(
            self.data_dir,
            split='test',
        )
