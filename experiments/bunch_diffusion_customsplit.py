import pickle
import torch
import copy
from src.mlp import MLP
from stable_diffusion.scripts.stable_diffuser import StableDiffuser
from src.data_handler_bert import DataHandlerBERT
from src.utils import print_progress_bar
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_ITER = 2000
AFF_WEIGHT = 500
FOLDER = "results/stdiff_R6"
RECOMPUTE_MEANS = False
N_SAMPLES = 3
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
            mlp = MLP(param_env="mlp.env", h0=768).to(device)
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
            

# MAIN starts here
aff_names = ["V", "A", "D"]

criterion = torch.nn.MSELoss(reduction='mean')

data_handler = DataHandlerBERT("data/Ratings_Warriner_et_al.csv")
for prompt in PROMPTS:

    folder = f"{FOLDER}/{prompt.replace(' ','_')}"
    os.makedirs(folder, exist_ok=True)

    z_0 = data_handler.model.get_learned_conditioning([prompt])
    z_0.requires_grad = False

    dists = torch.min(0.95*torch.ones((3), device=device)-mean_affects[prompt], mean_affects[prompt]-0.05*torch.ones((3), device=device))

    start_code = torch.randn(
        [N_SAMPLES, 4, 512 // 8, 512 // 8],
        device=device,
    )

    for aff_idx in range(3):
        for tick in range(5):
            for correction in [-0.02, 0, 0.02]:
                aff_val = mean_affects[prompt][aff_idx]-(1-0.5*tick)*dists[aff_idx] + correction
                v_name = f"{aff_names[aff_idx]}_{round(100*aff_val.item())}"

                print(f"Generating {prompt} with affect {v_name}...")

                zz = torch.zeros_like(z_0)
                
                for channel in range(77):
                    print_progress_bar(channel+1, 77, channel+1, suffix= "-- Channel:")

                    path = f"data/bert_nets/data_ch_{channel}.pkl"
                    data_handler.load_data(savepath=path)

                    with torch.no_grad():
                        mlp = MLP(param_env="mlp.env", h0=768).to(device)
                        mlp.load_state_dict(torch.load(f'data/bert_nets/model_ch_{channel}.pt'))

                    z = copy.deepcopy(z_0[:, channel, :])

                    if channel != 0: #channel 0 has no info
                        z.requires_grad = True

                        opt = torch.optim.Adam([z], lr=0.15)

                        v_0 = mlp(z_0[0, channel, :])
                        if tick!=2:
                            for iter in range(MAX_ITER):
                                opt.zero_grad()
                                loss = 0
                                loss += criterion(z, z_0[:, channel, :])
                                loss += AFF_WEIGHT * criterion(
                                    mlp(data_handler.scaler_Z.scale(z))[:, aff_idx].unsqueeze(0), aff_val
                                )
                                loss.backward()
                                opt.step()

                    with torch.no_grad():
                        zz[0, channel, :] = copy.deepcopy(z.detach())
                    
                    torch.cuda.empty_cache()

                # print(zz[:3,:3,:3])
                zz = zz.to('cpu')

                stable_diffuser = StableDiffuser()
                stable_diffuser.initialize(prompt=prompt, start_code = start_code)
                if tick != 2:
                    print("MOD!")
                    stable_diffuser.override_zz(zz)
                stable_diffuser.run_diffusion(alt_savepath=folder, im_name = f"_{v_name}")
