from tqdm.notebook import tqdm
from PIL import ImageFile
from src.affective_generator2 import AffectiveGenerator
import torch
import os


ImageFile.LOAD_TRUNCATED_IMAGES = True

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

affective_generator = AffectiveGenerator()
for trial in range(N_TRIALS):
    # seed_everything(trial)
    noise_0 = torch.randint(
        affective_generator.n_toks,
        [affective_generator.toksY * affective_generator.toksX],
        device=device,
    )
    for prompt in PROMPTS:
        os.makedirs(f"results/vqgan_R1/{prompt.replace(' ','_')}", exist_ok=True)

        for v in vv:
            # seed_everything(trial)
            affective_generator.initialize(
                prompts=prompt,
                v=vv[v],
                savepath =f"results/vqgan_R1/{prompt.replace(' ','_')}/{trial}_{v}.png",
                seed=trial,
                noise_0=noise_0,
            )
            i = 0
            with tqdm() as pbar:
                while True:
                    affective_generator.train(i, AFF_WEIGHT)
                    if i == MAX_ITER:
                        break
                    i += 1
                    pbar.update()


# vv = {
#     'high_E': [1.0, None, None],
#     'low_E': [0.0, None, None],
#     'high_P': [None, 1.0, None],
#     'low_P': [None, 0.0, None],
#     'high_A': [None, None, 1.0],
#     'low_A': [None, None, 0.0],
# }

# affective_generator = AffectiveGenerator()
# for trial in range(N_TRIALS):
#     noise_0 = torch.randint(
#         affective_generator.n_toks,
#         [affective_generator.toksY * affective_generator.toksX],
#         device=device,
#     )
#     for prompt in PROMPTS:
#         seed_everything(trial)

#         img0_path = f"results/vqgan_split_noned2/{trial}_{prompt.replace(' ','_')}_img0.png"

#         # Build starting image
#         affective_generator.initialize(
#                 prompts=prompt,
#                 img_savedir = img0_path,
#                 seed=trial,
#                 noise_0=noise_0,
#             )
#         i = 0
#         with tqdm() as pbar:
#             while True:
#                 affective_generator.train(i)
#                 if i == 25:
#                     break
#                 i += 1
#                 pbar.update()

#         for v in vv:
#             seed_everything(trial)
#             affective_generator.initialize(
#                 prompts=prompt,
#                 v=vv[v],
#                 img_0 = img0_path,
#                 img_savedir = f"results/vqgan_split_noned2/{trial}_{prompt.replace(' ','_')}_{v}.png",
#                 seed=trial,
#                 noise_0=noise_0,
#             )
#             i = 0
#             with tqdm() as pbar:
#                 while True:
#                     affective_generator.train(i,AFF_WEIGHT)
#                     if i == MAX_ITER:
#                         break
#                     i += 1
#                     pbar.update()
