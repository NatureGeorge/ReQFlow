import numpy as np
import pandas as pd
import logging
import tree
import torch
import random

from torch.utils.data import Dataset
from data import utils as du
from openfold.data import data_transforms
from openfold.utils import rigid_utils

from openfold.utils.rigid_utils import rot_to_quat, quat_to_rot

class RectifyProtDataset(Dataset):
    def __init__(
            self, 
            *,
            dataset_cfg,
            task,
            is_training):
        self._log = logging.getLogger(__name__)
        self._dataset_cfg = dataset_cfg
        self.is_training = is_training
        data = pd.read_csv(self._dataset_cfg.rectify_csv_path)
        self._create_split(data)
        
        
    
    def _create_split(self, data_csv):
        if self.is_training:
            self.csv = data_csv
            self._log.info(f'Loaded training dataset with {len(self.csv)} samples.')
        else:
            if self._dataset_cfg.max_eval_length is None and self._dataset_cfg.min_eval_length is None:
                # min and max are empty
                eval_lengths = data_csv.length
            elif self._dataset_cfg.max_eval_length is None:
                # min is not empty, max is empty
                eval_lengths = data_csv.length[
                    data_csv.length >= self._dataset_cfg.min_eval_length
                ]
            else:
                # max is not empty, min is empty
                eval_lengths = data_csv.length[
                    data_csv.length <= self._dataset_cfg.max_eval_length
                ]
            if self._dataset_cfg.min_eval_length is not None:
                eval_lengths = eval_lengths[eval_lengths >= self._dataset_cfg.min_eval_length]

            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self._dataset_cfg.num_eval_lengths)
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[data_csv.length.isin(eval_lengths)]
            eval_csv = eval_csv.groupby('length').sample(
                self._dataset_cfg.samples_per_eval_length,
                replace=True,
                random_state=123
            )
            eval_csv = eval_csv.sort_values('length', ascending=False)
            self.csv = eval_csv
            self._log.info(
                f'Validation: {len(self.csv)} examples with lengths {eval_lengths}')
        
        self.csv['index'] = list(range(len(self.csv)))




    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        prot_traj = torch.load(row['prot_path'], map_location='cpu')
        num_res = length = row['length']
        sample_batch = {}
        noise_batch = {}

        trans_0 = prot_traj[0][0]
        rotmats_0 = prot_traj[0][1]
        gt_trans_1 = prot_traj[1][0]
        gt_rotmats_1 = prot_traj[1][1]

        rotquats_0 = rot_to_quat(rotmats_0)
        gt_rotquats_1 = rot_to_quat(gt_rotmats_1)

        sample_batch['trans_1'] = gt_trans_1.squeeze(dim=0) # (1, num_res, 3) -> (num_res, 3)
        sample_batch['rotmats_1'] = gt_rotmats_1.squeeze(dim=0) # (1, num_res, 3, 3) -> (num_res, 3, 3)
        sample_batch['rotquats_1'] = gt_rotquats_1.squeeze(dim=0) # (1, num_res, 4) -> (num_res, 4)
        sample_batch['res_mask'] = torch.ones(num_res, dtype=torch.int)
        sample_batch['diffuse_mask'] = torch.ones(num_res, dtype=torch.int)
        sample_batch['res_idx'] = torch.arange(num_res, dtype=torch.int)
        sample_batch['csv_idx'] = torch.tensor([idx], dtype=torch.long) 
        sample_batch['chain_idx'] = torch.zeros(1, dtype=torch.int)

        noise_batch['trans_1'] = trans_0.squeeze(dim=0)
        noise_batch['rotmats_1'] = rotmats_0.squeeze(dim=0)
        noise_batch['rotquats_1'] = rotquats_0.squeeze(dim=0)
        noise_batch['res_mask'] = torch.ones(num_res, dtype=torch.int)
        noise_batch['diffuse_mask'] = torch.ones(num_res, dtype=torch.int)
        noise_batch['res_idx'] = torch.arange(num_res, dtype=torch.int)
        noise_batch['csv_idx'] = torch.tensor([idx], dtype=torch.long) 
        noise_batch['chain_idx'] = torch.zeros(1, dtype=torch.int)





        return {'sample': sample_batch, 'noise': noise_batch}
    


        
