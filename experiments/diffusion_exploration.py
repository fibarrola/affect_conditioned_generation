import torch
import copy
from src.mlp import MLP
from stable_diffusion.scripts.stable_diffuser import StableDiffuser
from src.data_handler_bert_v2 import DataHandlerBERT, load_model_from_config
from src.utils import print_progress_bar, renum_path
from omegaconf import OmegaConf
import os
import numpy as np
from dotenv import load_dotenv

param_env = "bert.env"

load_dotenv(param_env)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_ITER = 500
AFF_WEIGHT = 500
FOLDER = renum_path("results/exploration_0")
RECOMPUTE_MEANS = False
N_SAMPLES = 12
AFFECT_VALS = [0., 1.]
PROMPTS = [
    "Lion",
    "Elephant",
    "Tiger",
    "Puppy",
    "Snake",
    "Spider",
    "Wasp",   
    "House on fire",
    "Storm",
    "House overlooking the ocean",
    "Crocodile",
]

aux_prompts = [
    "happy",
    "unhappy",
    "calm",
    "excited",
    "submissive",
    "dominant"
]


config = OmegaConf.load(os.environ.get("SD_CONFIG"))
model = load_model_from_config(config, os.environ.get("SD_MODEL"))

# MAIN starts here
aff_names = ["V", "A", "D"]

criterion = torch.nn.MSELoss(reduction='mean')

data_handler = DataHandlerBERT()
for prompt in PROMPTS:

    folder = f"{FOLDER}/{prompt.replace(' ','_')}"
    os.makedirs(folder, exist_ok=True)

    z_0 = model.get_learned_conditioning([prompt])
    z_0.requires_grad = False

    start_code = torch.randn([N_SAMPLES, 4, 512 // 8, 512 // 8],)

    for aff_idx in range(3):
        for aff_val in AFFECT_VALS:
            v_name = f"{aff_names[aff_idx]}_{round(100*aff_val)}"
            aff_val = torch.tensor(aff_val, device=device, requires_grad=False)

            print(f"Generating {prompt} with affect {v_name}...")

            zz = torch.zeros_like(z_0)
            total_affect = 0

            for channel in range(77):  # channel 0 has no info
                aff_val.requires_grad = False
                aff_val = aff_val.detach()
                print_progress_bar(channel + 1, 77, channel + 1, suffix="-- Channel:")

                path = f"{os.environ.get('MODEL_PATH')}/data_ch_{channel}.pkl"

                z = copy.deepcopy(z_0[:, channel, :])

                if channel != 0:

                    with torch.no_grad():
                        mlp = MLP(
                            h0=int(os.environ.get('IMG_SIZE')),
                            use_dropout=os.environ.get('USE_DROPOUT'),
                            use_sigmoid=os.environ.get('USE_SIGMOID'),
                        ).to('cuda:0')
                        mlp.load_state_dict(
                            torch.load(
                                f"{os.environ.get('MODEL_PATH')}/model_ch_{channel}.pt"
                            )
                        )

                    z += 0.01 * torch.std(z) * torch.rand_like(z)
                    z.requires_grad = True

                    opt = torch.optim.Adam([z], lr=0.2)

                    for iter in range(MAX_ITER):
                        opt.zero_grad()
                        loss = 0
                        loss += criterion(z, z_0[:, channel, :])
                        det_aff_val = aff_val.detach()
                        loss += AFF_WEIGHT * criterion(
                            mlp(z)[:, aff_idx].unsqueeze(0), det_aff_val
                        )
                        loss.backward()
                        opt.step()

                    total_affect += mlp(z)[:, aff_idx].detach().item()

                with torch.no_grad():
                    zz[0, channel, :] = copy.deepcopy(z.detach())

                torch.cuda.empty_cache()

            print(f"Mean Aff = {total_affect/76}, Target: {aff_val.item()}")

            zz = zz.to('cpu')

            stable_diffuser = StableDiffuser()
            for batch in range(int(np.ceil(N_SAMPLES / 3))):
                stable_diffuser.initialize(
                    prompt=prompt,
                    start_code=start_code[3 * batch : 3 * (batch + 1), :, :, :],
                )
                stable_diffuser.override_zz(zz)
                stable_diffuser.run_diffusion(
                    alt_savepath=folder, im_name=f"_{v_name}", batch_n=batch
                )

    stable_diffuser = StableDiffuser()
    for batch in range(int(np.ceil(N_SAMPLES / 3))):
        stable_diffuser.initialize(
            prompt=prompt,
            start_code=start_code[3 * batch : 3 * (batch + 1), :, :, :],
        )
        stable_diffuser.run_diffusion(
            alt_savepath=folder, im_name=f"_no_aff", batch_n=batch
        )

    
    for aux_prompt in aux_prompts:

        for batch in range(int(np.ceil(N_SAMPLES / 3))):
            stable_diffuser.initialize(
                prompt=f"{prompt} that makes me feel very {aux_prompt}",
                start_code=start_code[3 * batch : 3 * (batch + 1), :, :, :],
            )
            stable_diffuser.run_diffusion(
                alt_savepath=folder, im_name=f"_{aux_prompt}", batch_n=batch
            )