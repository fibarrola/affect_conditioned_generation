import pickle
import torch
import copy
import os
from src.mlp import MLP
import plotly.graph_objects as go


W = 0.4
# METHOD = 'full_vec'
METHOD = 'single_dim'
MAX_ITER = 500
LR = 0.1
PROMPTS = [
    # 'A dog in the forest',
    # 'A crocodile',
    # 'A colourful wild animal',
    # 'A dark forest',
    # 'A forest',
    # 'A house overlooking the sea',
    # 'A large wild animal',
    # 'A spaceship',
    # 'An elephant',
    # 'An UFO',
    # 'The sea at night',
    'The sea at sunrise',
]
PROMPT2 = "A messy sea at sunrise"
Vs = {
    'high_E': [1.0, 0.5, 0.5],
    'low_E': [0.0, 0.5, 0.5],
    'high_P': [0.5, 1.0, 0.5],
    'low_P': [0.5, 0.0, 0.5],
    'high_A': [0.5, 0.5, 1.0],
    'low_A': [0.5, 0.5, 0.0],
    'no_aff': [0.5, 0.5, 0.5],
}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
with open(f'data/bert_nets/data_handler_bert_{0}.pkl', 'rb') as f:
    data_handler = pickle.load(f)

mlp = MLP([64, 32], do=True, sig=False, h0=768).to(device)
criterion = torch.nn.MSELoss(reduction='mean')
mean_error = torch.nn.L1Loss(reduction='mean')

import os

zzz = torch.empty([len(Vs), 77, 768])
for prompt in PROMPTS:
    print(f"----- {prompt} -----")
    with torch.no_grad():
        z_0 = data_handler.model.get_learned_conditioning([prompt]).to('cpu')
        z_1 = data_handler.model.get_learned_conditioning([PROMPT2]).to('cpu')
        for i, v_name in enumerate(Vs):
            print(f"----- {v_name} -----")
            with open(
                f"data/diff_embeddings2/{METHOD}_{int(10*W)}/{prompt.replace(' ','_')}/{v_name}.pkl", 'rb'
            ) as f:
                zzz[i,:,:] = pickle.load(f).to('cpu').squeeze(0)
            

# fig = go.Figure()
# for i, v_name in enumerate(Vs):
#     yy = torch.norm((zzz[i,:,:]-z_1).squeeze(0), dim=1)
#     fig.add_trace(go.Box(y=yy, name=v_name))

# fig.show()

fig = go.Figure()
for i, v_name in enumerate(Vs):
    if v_name == 'low_A' or v_name == 'low_P' or v_name == 'low_E':
        yy = torch.norm((zzz[i-1,:,:]-z_1).squeeze(0), dim=1)-torch.norm((zzz[i,:,:]-z_1).squeeze(0), dim=1)
        fig.add_trace(go.Box(y=yy, name=f"d{list(Vs.keys())[i-1]}-d{v_name}"))
        fig.update_layout(title=PROMPT2)

fig.show()