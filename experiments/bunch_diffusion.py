import pickle
import torch
import copy
from src.mlp import MLP
from stable_diffusion.scripts.stable_diffuser import StableDiffuser
from src.data_handler_bert import DataHandlerBERT
from src.utils import print_progress_bar
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    "Sea",
    "Forest",
    "Mountain",
    "Grassland",
    "Island",
    "Beach",
    "Desert",
    "City",
]
MAX_ITER = 1500
AFF_WEIGHT = 7
N_TRIALS = 1
vv = {
    'V90': [0.90, None, None],
    'V65': [0.65, None, None],
    'V50': [0.50, None, None],
    'V35': [0.35, None, None],
    'V10': [0.10, None, None],
    'A90': [None, 0.90, None],
    'A65': [None, 0.65, None],
    'A50': [None, 0.50, None],
    'A35': [None, 0.35, None],
    'A10': [None, 0.10, None],
    'D90': [None, None, 0.90],
    'D65': [None, None, 0.65],
    'D50': [None, None, 0.50],
    'D35': [None, None, 0.35],
    'D10': [None, None, 0.10],
    '000': [None, None, None],
}

criterion = torch.nn.MSELoss(reduction='mean')

data_handler = DataHandlerBERT("data/Ratings_Warriner_et_al.csv")

for prompt in PROMPTS:

    folder = f"results/stdiff_R1/{prompt.replace(' ','_')}"
    os.makedirs(folder, exist_ok=True)

    z_0 = data_handler.model.get_learned_conditioning([prompt])
    z_0.requires_grad = False

    for v_name in vv:
        print(f"Generating {prompt} with affect {v_name}...")

        target_dims = [k for k in range(3) if not vv[v_name][k] is None]
        print(target_dims)
        target_v = torch.tensor(
            [0.5 if v is None else v for v in vv[v_name]], device=device, requires_grad=False
        )

        zz = torch.zeros_like(z_0)
        
        for channel in range(77):
            print_progress_bar(channel+1, 77, channel, suffix= "-- Channel:")

            path = f"data/bert_nets/data_ch_{channel}.pkl"
            data_handler.load_data(savepath=path)

            with torch.no_grad():
                mlp = MLP(param_env="mlp.env", h0=768).to(device)
                mlp.load_state_dict(torch.load(f'data/bert_nets/model_ch_{channel}.pt'))

            z = copy.deepcopy(z_0[:, channel, :])

            if channel != 0: #channel 0 has no info
                z.requires_grad = True

                opt = torch.optim.Adam([z], lr=0.1)

                v_0 = mlp(z_0[0, channel, :])
                if len(target_dims) > 0:
                    for iter in range(MAX_ITER):
                        opt.zero_grad()
                        loss = 0
                        loss += criterion(z, z_0[:, channel, :])
                        for dim in target_dims:
                            loss += AFF_WEIGHT * criterion(
                                mlp(data_handler.scaler_Z.scale(z))[:, dim], target_v[dim:dim+1]
                            )
                        loss.backward()
                        opt.step()

            with torch.no_grad():
                zz[0, channel, :] = copy.deepcopy(z.detach())
            
            torch.cuda.empty_cache()

        zz = zz.to('cpu')

        stable_diffuser = StableDiffuser()
        stable_diffuser.initialize(prompt=prompt)
        stable_diffuser.override_zz(zz)
        stable_diffuser.run_diffusion(alt_savepath=f"{folder}/0_{v_name}.png")
