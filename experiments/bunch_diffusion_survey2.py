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
AFF_WEIGHT = 50
FOLDER = "results/stdiff_survey4"
RECOMPUTE_MEANS = False
N_SAMPLES = 12
PROMPTS = [
    "City",
    "Grassland",
    "Beach",
    "Mountain"
    # "Tiger",
    # "Elephant",
    # "Lion",
    # "House on fire",
    # "Puppy",
    # "Storm",
    # "House overlooking the ocean",
    # "Puppy",
    # "Tiger",
    # "Elephant",
    # "Crocodile",
    # "Snake",
    # "Spider",
    # "Wasp"
]
           

# MAIN starts here
aff_names = ["V", "A", "D"]

criterion = torch.nn.MSELoss(reduction='mean')

data_handler = DataHandlerBERT("data/Ratings_Warriner_et_al.csv")
for prompt in PROMPTS:

    folder = f"{FOLDER}/{prompt.replace(' ','_')}"
    os.makedirs(folder, exist_ok=True)

    z_0 = data_handler.model.get_learned_conditioning([prompt])
    z_0.requires_grad = False

    start_code = torch.randn(
        [N_SAMPLES, 4, 512 // 8, 512 // 8],
    )

    for aff_idx in range(3):
        for aff_val in [0.1, 0.5, 0.9]:
            v_name = f"{aff_names[aff_idx]}_{round(100*aff_val)}"
            aff_val = torch.tensor(aff_val, device=device, requires_grad=False)

            print(f"Generating {prompt} with affect {v_name}...")


            zz = torch.zeros_like(z_0)
            total_affect = 0
            
            for channel in range(1, 77):
                aff_val.requires_grad = False
                aff_val = aff_val.detach()
                print_progress_bar(channel+1, 77, channel+1, suffix= "-- Channel:")

                path = f"data/bert_nets/data_ch_{channel}.pkl"
                data_handler.load_data(savepath=path)

                with torch.no_grad():
                    mlp = MLP([64, 32], do=True, sig=False, h0=768).to(device)
                    mlp.load_state_dict(torch.load(f'data/bert_nets/model_ch_{channel}.pt'))

                z = copy.deepcopy(z_0[:, channel, :])

                if True: #aff_val != 0.5:
                    z += 0.01*torch.std(z)*torch.rand_like(z)

                    if channel != 0: #channel 0 has no info
                        z.requires_grad = True

                        opt = torch.optim.Adam([z], lr=0.1)

                        for iter in range(MAX_ITER):
                            opt.zero_grad()
                            loss = 0
                            loss += criterion(z, z_0[:, channel, :])
                            det_aff_val = aff_val.detach()
                            loss += AFF_WEIGHT * criterion(
                                mlp(data_handler.scaler_Z.scale(z))[:, aff_idx].unsqueeze(0), det_aff_val
                            )
                            loss.backward()
                            opt.step()

                        total_affect += mlp(data_handler.scaler_Z.scale(z))[:, aff_idx].detach().item()
                
                else:
                    print("NO AFF")

                with torch.no_grad():
                    zz[0, channel, :] = copy.deepcopy(z.detach())
                
                torch.cuda.empty_cache()
            
            print(f"Mean Aff = {total_affect/76}, Target: {aff_val.item()}")

            zz = zz.to('cpu')

            stable_diffuser = StableDiffuser()
            for batch in range(int(np.ceil(N_SAMPLES/3))):
                stable_diffuser.initialize(prompt=prompt, start_code = start_code[3*batch:3*(batch+1),:,:,:])
                stable_diffuser.override_zz(zz)
                stable_diffuser.run_diffusion(alt_savepath=folder, im_name = f"_{v_name}", batch_n=batch)
