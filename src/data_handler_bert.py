import torch
import clip
import pandas as pd
import pickle
from src.scaler import Scaler
from torch.utils.data import random_split, TensorDataset, DataLoader
import torch
import clip
import pandas as pd
import numpy as np
from src.scaler import Scaler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from .utils import N_max_elements, get_furthest_apart
import logging

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

class DataHandlerBERT:
    def __init__(self, csv_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.df = pd.read_csv(csv_path).dropna()
        config = OmegaConf.load(
                    "stable_diffusion/configs/stable-diffusion/v1-inference.yaml"
                )
        self.model = load_model_from_config(
            config, "stable_diffusion/models/ldm/stable-diffusion-v1/model.ckpt"
        )

    @torch.no_grad()
    def preprocess(self, word_type=None, scaling='none', word_batch_size=2048, channel=9):
        torch.cuda.empty_cache()
        fil_df = self.df[self.df["Type"] == word_type] if word_type else self.df
        self.words = list(fil_df["Word"])
        self.Z = torch.tensor([], device=self.device)
        torch.cuda.empty_cache()
        for m in range(
            int(np.ceil(len(self.words) / word_batch_size))
        ):  # looks weird but avoids all lenght problems
            words_subset = self.words[
                m
                * word_batch_size : min(len(self.words), (m + 1) * word_batch_size)
            ]
            with torch.no_grad():
                Z_subset = self.model.get_learned_conditioning(words_subset)[:,channel,:]
                self.Z = torch.cat([self.Z, Z_subset], 0)
                if len(self.words) <= (m + 1) * word_batch_size:
                    break
        
        dim_names = ['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']
        self.V = torch.tensor(
            fil_df[dim_names].values, device=self.device, dtype=torch.float32
        )
        sd_dim_names = ['V.SD.Sum', 'A.SD.Sum', 'D.SD.Sum']
        self.Vsd = torch.tensor(
            fil_df[sd_dim_names].values, device=self.device, dtype=torch.float32
        )

        self.scaler_Z = Scaler(self.Z, scaling)
        self.scaler_V = Scaler(self.V, scaling)

    @torch.no_grad()
    def build_datasets(self, train_ratio=0.7, batch_size=512):
        Z = self.scaler_Z.scale(self.Z)
        V = self.scaler_V.scale(self.V)
        dataset = TensorDataset(Z, V, self.Vsd)
        (ds_train, ds_test) = random_split(dataset, [round(len(dataset) * train_ratio), round(len(dataset) * (1 - train_ratio))])
        self.train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)