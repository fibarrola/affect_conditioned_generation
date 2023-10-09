import torch
import pandas as pd
from src.scaler import Scaler
from torch.utils.data import random_split, TensorDataset, DataLoader
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import pickle


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
    def __init__(self):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
             

    @torch.no_grad()
    def preprocess(self, csv_path, savepath, sdconfig, sdmodel, word_type=None, v_scaling=False, word_batch_size=2048, channel=9):
        self.df = pd.read_csv(csv_path).dropna()
        config = OmegaConf.load(sdconfig)
        self.model = load_model_from_config(config, sdmodel)
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
                Z_subset = self.model.get_learned_conditioning(words_subset)[:, channel, :]
                self.Z = torch.cat([self.Z, Z_subset], 0)
                if len(self.words) <= (m + 1) * word_batch_size:
                    break

        dim_names = ['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']
        self.V = torch.tensor(
            fil_df[dim_names].values, device=self.device, dtype=torch.float32
        )
        
        if v_scaling:
            self.vmax, _ = torch.max(self.V, dim=0)
            self.vmin, _ = torch.min(self.V, dim=0)
            self.V = (self.V-self.vmin)/(self.vmax-self.vmin+1e-9)

        data = {
            "Z": self.Z,
            "V": self.V,
            "vmax": self.vmax,
            "vmin": self.vmin,
        }
        with open(savepath, 'wb') as f:
            pickle.dump(data, f)    
    
    @torch.no_grad()
    def load_data(self, savepath):
        with open(savepath, 'rb') as f:
            data = pickle.load(f)
        self.Z = data["Z"]
        self.V = data["V"]
        self.vmin = data["vmin"]
        self.vmax = data["vmax"]

    @torch.no_grad()
    def build_datasets(self, train_ratio=0.7, batch_size=512):
        dataset = TensorDataset(self.Z, self.V)
        (ds_train, ds_test) = random_split(dataset, [round(len(dataset) * train_ratio), round(len(dataset) * (1 - train_ratio))])
        self.train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
