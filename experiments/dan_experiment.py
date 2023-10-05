import pickle
import torch
import copy
import argparse
from src.mlp import MLP
from stable_diffusion.scripts.stable_diffuser import StableDiffuser
import plotly.graph_objects as go


device = "cuda:0" if torch.cuda.is_available() else "cpu"

fig = go.Figure()

for prompt in ["a puppy", "a dead puppy lying in the desert, away from his friends and family"]:
    xx = []

    with open(f'data/bert_nets/data_handler_bert_{0}.pkl', 'rb') as f:
        data_handler = pickle.load(f)

    mlp = MLP(param_env="mlp.env", h0=768).to(device)
    criterion = torch.nn.MSELoss(reduction='mean')

    z_0 = data_handler.model.get_learned_conditioning([prompt])
    z_0.requires_grad = False
    dim = 0

    for channel in range(77):
        with open(f'data/bert_nets/data_handler_bert_{channel}.pkl', 'rb') as f:
            data_handler = pickle.load(f)
        with torch.no_grad():
            mlp.load_state_dict(torch.load(f'data/bert_nets/model_{channel}.pt'))

            out = mlp(data_handler.scaler_Z.scale(z_0[:, channel, :]))[0,dim]
        
            xx.append(out.detach().item())

    fig.add_trace(go.Scatter(x=list(range(len(xx))), y=xx, name=prompt))

fig.show()