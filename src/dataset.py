from torch.utils.data import Dataset, DataLoader
import torch, pandas as pd, numpy as np
from generate_dataset import generate_data
import math


DATASET_PATH = '../scripts/data_32bits_polar.csv'


class PolarDecDataset(Dataset):

    def __init__(self, snr_db,  num_samples, seq_length, snr_noise_std=0.1, fixed_msg_bit_size = None, transform = None):
        super().__init__()
        self.snr_db = snr_db + np.random.normal(0, snr_noise_std)
        self.num_samples = num_samples
        self.fixed_msg_bit_size = fixed_msg_bit_size
        self.seq_length = seq_length
        

    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
         
         message_bit_size=np.random.choice([8, 16, 24])
         
         
         channel_observation_vector, frozen_bit_prior, target = generate_data(message_bit_size=message_bit_size if self.fixed_msg_bit_size is None else self.fixed_msg_bit_size, SNRs_db=[self.snr_db])
         channel_tensor = torch.tensor(channel_observation_vector, dtype=torch.float32)
         frozen_tensor = torch.tensor(frozen_bit_prior, dtype=torch.float32)
         snr_tensor = torch.tensor(self.snr_db, dtype=torch.float32)
         target_tensor = torch.tensor(target, dtype=torch.float32)

         code_rate = message_bit_size/self.seq_length

         llrs = 2* channel_observation_vector * math.pow(10, self.snr_db/10) * code_rate

        
         return llrs, frozen_tensor, snr_tensor, target_tensor
    


    

    


        
