import torch
import clip
import pandas as pd
import pickle
from src.scaler import Scaler
from torch.utils.data import random_split, TensorDataset, DataLoader


class DataHandler:
    def __init__(self, csv_path):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.df = pd.read_csv(csv_path).dropna()
        self.clip_model, preprocess = clip.load('ViT-B/32', self.device, jit=False)
        with open("data/img_Vmn.pkl", "rb") as f:
            V = pickle.load(f)
        self.V_img = V.permute(1,0).to(torch.float32).to(self.device)
        with open("data/img_Vsd.pkl", "rb") as f:
            V = pickle.load(f)
        self.Vsd_img = V.permute(1,0).to(torch.float32).to(self.device)
        with open("data/img_Z.pkl", "rb") as f:
            Z = pickle.load(f)
        self.Z_img = Z.to(torch.float32).to(self.device)

    @torch.no_grad()
    def preprocess(self, word_type=None, scaling='none'):
        torch.cuda.empty_cache()
        fil_df = self.df[self.df["Type"] == word_type] if word_type else self.df
        self.words = list(fil_df["Word"])
        tokens = clip.tokenize(self.words).to(self.device)
        self.Z_txt = self.clip_model.encode_text(tokens).to(torch.float32)
        dim_names = ['V.Mean.Sum', 'A.Mean.Sum', 'D.Mean.Sum']
        self.V_txt = torch.tensor(
            fil_df[dim_names].values, device=self.device, dtype=torch.float32
        )
        sd_dim_names = ['V.SD.Sum', 'A.SD.Sum', 'D.SD.Sum']
        self.Vsd_txt = torch.tensor(
            fil_df[dim_names].values, device=self.device, dtype=torch.float32
        )
        self.scaler_Z = Scaler(self.Z_txt, scaling)
        self.scaler_V = Scaler(self.V_txt, scaling)

    @torch.no_grad()
    def build_datasets(self, train_ratio=0.7, batch_size=512):
        Z_txt = self.scaler_Z.scale(self.Z_txt)
        V_txt = self.scaler_V.scale(self.V_txt)
        Z_img = self.scaler_Z.scale(self.Z_img)
        V_img = self.scaler_V.scale(self.V_img)
        dataset_txt = TensorDataset(Z_txt, V_txt, self.Vsd_txt)
        dataset_img = TensorDataset(Z_img, V_img, self.Vsd_img)
        (ds_train, ds_test) = random_split(dataset_txt, [round(len(dataset_txt)*train_ratio), round(len(dataset_txt)*(1-train_ratio))])
        self.txt_train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
        self.txt_test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
        (ds_train, ds_test) = random_split(dataset_img, [round(len(dataset_img)*train_ratio), round(len(dataset_img)*(1-train_ratio))])
        self.img_train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=False)
        self.img_test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False)
