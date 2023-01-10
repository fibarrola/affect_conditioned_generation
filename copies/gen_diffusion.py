import pickle
import torch
import copy
from src.mlp import MLP


W = 0.3
MAX_ITER = 5000
LR = 0.1
PROMPTS = [
    'A dog in the forest',
    'An old building',
    'A fish swimming in the sea',
    'A tree on a hilltop',
    'A meal on a white plate',
]
# Vs = {
#     'high_E': [0, 1.],
#     'low_E': [0, 0.],
#     'high_P': [1, 1.],
#     'low_P': [1, 0.],
#     'high_A': [2, 1.],
#     'low_A': [2, 0.],
# }
# Vs = {
#     'high_E': [1.0, None, None],
#     'low_E': [0.0, None, None],
#     'high_P': [None, 1.0, None],
#     'low_P': [None, 0.0, None],
#     'high_A': [None, None, 1.0],
#     'low_A': [None, None, 0.0],
#     'no_aff': [None, None, None],
# }
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

for prompt in PROMPTS:
    print(f"----- {prompt} -----")
    z_0 = data_handler.model.get_learned_conditioning([prompt])
    z_0.requires_grad = False
    for v_name in Vs:
        print(f"----- {v_name} -----")
        # v = torch.tensor([Vs[v_name][1]], device=device)
        # v = torch.tensor([Vs[v_name]], device=device)
        # v_idx = Vs[v_name][0]
        zz = torch.zeros_like(z_0)
        for channel in range(77):
            print(f"----- Adjusting Channel {channel} -----")

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

            for iter in range(MAX_ITER):
                opt.zero_grad()

                loss = 0
                loss += criterion(z, z_0[:, channel, :])
                loss += W * criterion(mlp(data_handler.scaler_Z.scale(z)), v)
                loss.backward()
                opt.step()

            with torch.no_grad():
                zz[0, channel, :] = copy.deepcopy(z.detach())

        zz = zz.to('cpu')
        with open(
            f"data/diff_embeddings/{prompt.replace(' ','_')}_{v_name}_010_03.pkl", 'wb'
        ) as f:
            pickle.dump(zz, f)

    z_0 = z_0.to('cpu')
    with open(f"data/{prompt.replace(' ', '_')}_z0_010_03.pkl", 'wb') as f:
        pickle.dump(z_0, f)
