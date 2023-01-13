import pickle
import torch
import copy
import os
from src.mlp import MLP


W = 0.2
# METHOD = 'full_vec'
METHOD = 'single_dim'
MAX_ITER = 500
LR = 0.1
PROMPTS = [
    'A dog in the forest',
    'A crocodile',
    'A colourful wild animal',
    'A dark forest',
    'A forest',
    'A house overlooking the sea',
    'A large wild animal',
    'A spaceship',
    'An elephant',
    'An UFO',
    'The sea at night',
    'The sea at sunrise',
]
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


for prompt in PROMPTS:
    print(f"----- {prompt} -----")
    z_0 = data_handler.model.get_learned_conditioning([prompt])
    z_0.requires_grad = False
    for v_name in Vs:
        print(f"----- {v_name} -----")
        zz = torch.zeros_like(z_0)
        for channel in range(77):
            print(f"Adjusting Channel {channel} ...")

            with open(f'data/bert_nets/data_handler_bert_{channel}.pkl', 'rb') as f:
                data_handler = pickle.load(f)
            with torch.no_grad():
                mlp.load_state_dict(torch.load(f'data/bert_nets/model_{channel}.pt'))

            z = copy.deepcopy(z_0[:, channel, :])
            z.requires_grad = True
            opt = torch.optim.Adam([z], lr=LR)

            v_0 = mlp(z_0[0, channel, :])
            v = [v_0[k] if Vs[v_name][k] is None else Vs[v_name][k] for k in range(3)]
            v = torch.tensor([v], device=device)

            if v_name != 'no_aff':
                for iter in range(MAX_ITER):
                    opt.zero_grad()
                    loss = 0
                    loss += criterion(z, z_0[:, channel, :])
                    if METHOD == 'full_vec':
                        loss += W * criterion(mlp(data_handler.scaler_Z.scale(z)), v)
                    elif METHOD == 'single_dim':
                        dim = 0 if v_name[-1] == 'E' else (1 if v_name[-1] == 'P' else 2)
                        loss += W * criterion(mlp(data_handler.scaler_Z.scale(z))[:, dim], v[:, dim])
                    loss.backward()
                    opt.step()

            with torch.no_grad():
                zz[0, channel, :] = copy.deepcopy(z.detach())
            
            # print('dim: ', dim)
            # print('mlp(z0): ', mlp(data_handler.scaler_Z.scale(z_0[:, channel, :])))
            # print('mlp(z)', mlp(data_handler.scaler_Z.scale(z)))

        zz = zz.to('cpu')
        os.makedirs(f"data/diff_embeddings2/{METHOD}_{int(10*W)}/{prompt.replace(' ','_')}/", exist_ok=True)
        with open(
            f"data/diff_embeddings2/{METHOD}_{int(10*W)}/{prompt.replace(' ','_')}/{v_name}.pkl", 'wb'
        ) as f:
            pickle.dump(zz, f)
