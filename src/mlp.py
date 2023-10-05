import torch
from torch import nn
import torch.nn.functional as F
import os
from dotenv import load_dotenv


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class MLP(nn.Module):
    def __init__(self, param_env, hh=[64, 32], h0=512):
        super(MLP, self).__init__()
        assert len(hh) >= 1
        load_dotenv(param_env)
        self.num_lay = len(hh) + 1
        hh = [h0] + hh + [os.environ.get("HF")]
        self.fc1 = nn.Linear(hh[0], hh[1])
        self.fc2 = nn.Linear(hh[1], hh[2])
        if self.num_lay >= 3:
            self.fc3 = nn.Linear(hh[2], hh[3])
        if self.num_lay >= 4:
            self.fc4 = nn.Linear(hh[3], hh[4])
        self.do = os.environ.get("USE_DROPOUT")
        self.sig = os.environ.get("USE_SIGMOID")
        if self.do:
            self.drop = nn.Dropout(p=0.2)

    def forward(self, x, train=False):
        x = self.fc1(x)
        x = F.relu(x)
        if self.do and train:
            x = self.drop(x)
        x = self.fc2(x)
        if self.num_lay >= 3:
            x = F.relu(x)
            if self.do and train:
                x = self.drop(x)
            x = self.fc3(x)
        if self.num_lay >= 4:
            x = F.relu(x)
            if self.do and train:
                x = self.drop(x)
            x = self.fc4(x)
        if self.sig:
            x = torch.sigmoid(x)
        return x
