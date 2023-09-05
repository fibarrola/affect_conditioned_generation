import pickle
import torch
import copy
from src.mlp import MLP
from stable_diffusion.scripts.stable_diffuser import StableDiffuser
from src.data_handler_bert import DataHandlerBERT
from src.utils import print_progress_bar
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

RECOMPUTE_MEANS = False
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

# MEAN AFFECT COMPUTING
if RECOMPUTE_MEANS:
    mean_affects = {}
    with torch.no_grad():
        data_handler = DataHandlerBERT("data/Ratings_Warriner_et_al.csv")
        for prompt in PROMPTS:
            print(f"Generating mean affectf for {prompt} with affect")
            mlp = MLP([64, 32], do=True, sig=False, h0=768).to(device)
            z_0 = data_handler.model.get_learned_conditioning([prompt])
            aff_sum = torch.zeros((3), device=device)
            for channel in range(1, 77):
                data_handler.load_data(savepath=f"data/bert_nets/data_ch_{channel}.pkl")
                print_progress_bar(channel+1, 77, channel+1, suffix= "-- Channel:")
                mlp.load_state_dict(torch.load(f'data/bert_nets/model_ch_{channel}.pt'))
                z = data_handler.scaler_Z.scale(z_0[:, channel, :])
                aff_sum += mlp(z)[0,:]
                torch.cuda.empty_cache()
            
            mean_affects[prompt] = aff_sum/76

    with open(f'data/bert_nets/aff_means.pkl', 'wb') as f:
        pickle.dump(mean_affects, f)

else:
    with open(f'data/bert_nets/aff_means.pkl', 'rb') as f:
        mean_affects = pickle.load(f)
            

MAX_ITER = 1000
AFF_WEIGHT = 1
N_TRIALS = 1
vv = {
    '000': [None, None, None],
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
}
aff_names = ["V", "A", "D"]

criterion = torch.nn.MSELoss(reduction='mean')

data_handler = DataHandlerBERT("data/Ratings_Warriner_et_al.csv")
for prompt in PROMPTS:

    folder = f"results/stdiff_R1/{prompt.replace(' ','_')}"
    os.makedirs(folder, exist_ok=True)

    z_0 = data_handler.model.get_learned_conditioning([prompt])
    z_0.requires_grad = False

    dists = torch.min(0.95*torch.ones((3), device=device)-mean_affects[prompt], mean_affects[prompt]-0.05*torch.ones((3), device=device))

    for aff_idx in range(3):
        for tick in range(5):
            aff_val = mean_affects[prompt][aff_idx]-(2-tick)*dists[aff_idx]
            v_name = f"{aff_names[aff_idx]}_{round(100*aff_val.item())}"

            print(f"Generating {prompt} with affect {v_name}...")

            zz = torch.zeros_like(z_0)
            
            for channel in range(77):
                print_progress_bar(channel+1, 77, channel+1, suffix= "-- Channel:")

                path = f"data/bert_nets/data_ch_{channel}.pkl"
                data_handler.load_data(savepath=path)

                with torch.no_grad():
                    mlp = MLP([64, 32], do=True, sig=False, h0=768).to(device)
                    mlp.load_state_dict(torch.load(f'data/bert_nets/model_ch_{channel}.pt'))

                z = copy.deepcopy(z_0[:, channel, :])

                if channel != 0: #channel 0 has no info
                    z.requires_grad = True

                    opt = torch.optim.Adam([z], lr=0.1)

                    v_0 = mlp(z_0[0, channel, :])
                    if tick!=2:
                        for iter in range(MAX_ITER):
                            opt.zero_grad()
                            loss = 0
                            loss += criterion(z, z_0[:, channel, :])
                            loss += AFF_WEIGHT * criterion(
                                mlp(data_handler.scaler_Z.scale(z))[:, aff_idx], aff_val
                            )
                            loss.backward()
                            opt.step()

                        print(mlp(data_handler.scaler_Z.scale(z))[:, aff_idx])

                with torch.no_grad():
                    zz[0, channel, :] = copy.deepcopy(z.detach())
                
                torch.cuda.empty_cache()

            zz = zz.to('cpu')

            stable_diffuser = StableDiffuser()
            stable_diffuser.initialize(prompt=prompt)
            if tick != 2:
                stable_diffuser.override_zz(zz)
            stable_diffuser.run_diffusion(alt_savepath=f"{folder}/0_{v_name}.png")
