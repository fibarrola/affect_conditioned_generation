import torch
import copy
from src.mlp import MLP
from stable_diffusion.scripts.stable_diffuser import StableDiffuser
from src.data_handler_bert import DataHandlerBERT
from src.utils import print_progress_bar
import os
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_ITER = 500
AFF_WEIGHT = 500
FOLDER = "results/stdiff_even_7"
RECOMPUTE_MEANS = False
N_SAMPLES = 12
PROMPTS = [
    # "Sea",
    # "Forest",
    # "Mountain",
    "Grassland",
    # "Island",
    # "Beach",
    # "Desert",
    "City",
]
           

# MAIN starts here
aff_names = ["V", "A", "D"]
10
criterion = torch.nn.MSELoss(reduction='mean')

data_handler = DataHandlerBERT("data/Ratings_Warriner_et_al.csv")
for noise in [0.01, 0.033]:
    for prompt in PROMPTS:

        folder = f"results/stdiff_even_E_{int(100000*noise)}/{prompt.replace(' ','_')}"
        os.makedirs(folder, exist_ok=True)

        z_0 = data_handler.model.get_learned_conditioning([prompt])
        z_0.requires_grad = False

        start_code = torch.randn(
            [N_SAMPLES, 4, 512 // 8, 512 // 8],
            device=device,
        )

        for aff_idx in range(3):
            for aff_val in [0.1, 0.5, 0.9]:
            # for aff_val in [n/10 for n in range(11)]+[None]:
                v_name = f"{aff_names[aff_idx]}_{round(100*aff_val)}" if not aff_val is None else "no_aff"
                aff_val = torch.tensor(aff_val, device=device, requires_grad=False) if not aff_val is None else None

                print(f"Generating {prompt} with affect {v_name}...")

                zz = torch.zeros_like(z_0)
                
                for channel in range(77):
                    print_progress_bar(channel+1, 77, channel+1, suffix= "-- Channel:")

                    path = f"data/bert_nets/data_ch_{channel}.pkl"
                    data_handler.load_data(savepath=path)

                    z = copy.deepcopy(z_0[:, channel, :])
                    if not aff_val is None:
                        z += noise*torch.std(z)*torch.rand_like(z)

                        with torch.no_grad():
                            mlp = MLP([64, 32], do=True, sig=False, h0=768).to(device)
                            mlp.load_state_dict(torch.load(f'data/bert_nets/model_ch_{channel}.pt'))                   

                        if channel != 0: #channel 0 has no info
                            z.requires_grad = True

                            opt = torch.optim.Adam([z], lr=0.15)

                            # w2 = 1# if channel ==1 else 0.1

                            v_0 = mlp(z_0[0, channel, :])
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

                zz = zz.to('cpu')

                stable_diffuser = StableDiffuser()
                for batch in range(int(np.ceil(N_SAMPLES/3))):
                    stable_diffuser.initialize(prompt=prompt, start_code = start_code[3*batch:3*(batch+1),:,:,:])
                    stable_diffuser.override_zz(zz)
                    stable_diffuser.run_diffusion(alt_savepath=folder, im_name = f"_{v_name}", batch_n=batch)

                    torch.cuda.empty_cache()