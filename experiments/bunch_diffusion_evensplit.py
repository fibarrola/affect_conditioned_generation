import torch
import copy
from src.mlp import MLP
from stable_diffusion.scripts.stable_diffuser import StableDiffuser
from src.data_handler_bert import DataHandlerBERT
from src.utils import print_progress_bar
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_ITER = 2000
AFF_WEIGHT = 1000
FOLDER = "results/stdiff_even_1"
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
        device=device,
    )

    for aff_idx in range(3):
        for aff_val in [0.00, 0.25, 0.33, 0.5, 0.67, 0.75, 1.00]:
            v_name = f"{aff_names[aff_idx]}_{round(100*aff_val)}"
            aff_val = torch.tensor(aff_val, device=device, requires_grad=False)

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

                    opt = torch.optim.Adam([z], lr=0.2)

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
            stable_diffuser.initialize(prompt=prompt, start_code = start_code)
            stable_diffuser.override_zz(zz)
            stable_diffuser.run_diffusion(alt_savepath=folder, im_name = f"_{v_name}")
