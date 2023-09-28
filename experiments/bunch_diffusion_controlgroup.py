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
FOLDER = "results/stdiff_extreme_8"
RECOMPUTE_MEANS = False
N_SAMPLES = 6
PROMPTS = [
    "Tiger",
    "Elephant",
    "Lion",
    "House on fire",
    "Puppy",
    "Storm",
    "House overlooking the ocean"
]
           

# MAIN starts here
aff_names = {
    "V": ["happy", "", "sad"],
    "A": ["excited", "", "calm"],
    "D": ["in control","", "controlled"],
}

criterion = torch.nn.MSELoss(reduction='mean')

data_handler = DataHandlerBERT("data/Ratings_Warriner_et_al.csv")
for prompt in PROMPTS:

    folder = f"{FOLDER}/{prompt.replace(' ','_')}"
    os.makedirs(folder, exist_ok=True)

    

    start_code = torch.randn(
        [N_SAMPLES, 4, 512 // 8, 512 // 8],
    )

    for aff_dim in aff_names:
        for aff_idx in [0, 1, 2]:
            aff_feel = aff_names[aff_dim][aff_idx]

            mod_prompt = f"{prompt} that makes me feel {aff_names[aff_dim][aff_idx]}" if aff_idx != 1 else prompt
            print(f"Generating {mod_prompt}...")


            stable_diffuser = StableDiffuser()
            for batch in range(int(np.ceil(N_SAMPLES/3))):
                stable_diffuser.initialize(prompt=mod_prompt, start_code = start_code[3*batch:3*(batch+1),:,:,:])
                stable_diffuser.run_diffusion(alt_savepath=folder, im_name = f"_{aff_dim}_{aff_idx}", batch_n=batch)
