# @title Carga de bibliotecas y definiciones

from tqdm.notebook import tqdm
from PIL import ImageFile
from src.affective_generator import AffectiveGenerator
import torch
from pytorch_lightning import seed_everything

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = "cuda:0" if torch.cuda.is_available() else "cpu"

PROMPTS = [
    'Waves hitting the rocks',
    'The sea at nightfall',
    'A dark forest',
    'A windy night',
    'flaming landscape',
]
MAX_ITER = 1500
AFF_WEIGHT = 1
N_TRIALS = 3
vv = {
    'high_E': [1.0, 0.5, 0.5],
    'low_E': [0.0, 0.5, 0.5],
    'high_P': [0.5, 1.0, 0.5],
    'low_P': [0.5, 0.0, 0.5],
    'high_A': [0.5, 0.5, 1.0],
    'low_A': [0.5, 0.5, 0.0],
}

affective_generator = AffectiveGenerator()
for trial in range(N_TRIALS):
    noise_0 = torch.randint(
        affective_generator.n_toks,
        [affective_generator.toksY * affective_generator.toksX],
        device=device,
    )
    for prompt in PROMPTS:
        seed_everything(trial)

        img0_path = f"results/vqgan_split3/{trial}_{prompt.replace(' ','_')}_img0.png"

        # Build starting image
        affective_generator.initialize(
                prompts=prompt,
                img_savedir = img0_path,
                seed=trial,
                noise_0=noise_0,
            )
        i = 0
        with tqdm() as pbar:
            while True:
                affective_generator.train(i)
                if i == 25:
                    break
                i += 1
                pbar.update()

        for v in vv:
            seed_everything(trial)
            affective_generator.initialize(
                prompts=prompt,
                v=vv[v],
                img_0 = img0_path,
                img_savedir = f"results/vqgan_split3/{trial}_{prompt.replace(' ','_')}_{v}.png",
                seed=trial,
                noise_0=noise_0,
            )
            i = 0
            with tqdm() as pbar:
                while True:
                    affective_generator.train(i,AFF_WEIGHT)
                    if i == MAX_ITER:
                        break
                    i += 1
                    pbar.update()


vv = {
    'high_E': [1.0, None, None],
    'low_E': [0.0, None, None],
    'high_P': [None, 1.0, None],
    'low_P': [None, 0.0, None],
    'high_A': [None, None, 1.0],
    'low_A': [None, None, 0.0],
}
affective_generator = AffectiveGenerator()
for trial in range(N_TRIALS):
    noise_0 = torch.randint(
        affective_generator.n_toks,
        [affective_generator.toksY * affective_generator.toksX],
        device=device,
    )
    for prompt in PROMPTS:
        seed_everything(trial)

        img0_path = f"results/vqgan_split_noned2/{trial}_{prompt.replace(' ','_')}_img0.png"

        # Build starting image
        affective_generator.initialize(
                prompts=prompt,
                img_savedir = img0_path,
                seed=trial,
                noise_0=noise_0,
            )
        i = 0
        with tqdm() as pbar:
            while True:
                affective_generator.train(i)
                if i == 25:
                    break
                i += 1
                pbar.update()

        for v in vv:
            seed_everything(trial)
            affective_generator.initialize(
                prompts=prompt,
                v=vv[v],
                img_0 = img0_path,
                img_savedir = f"results/vqgan_split_noned2/{trial}_{prompt.replace(' ','_')}_{v}.png",
                seed=trial,
                noise_0=noise_0,
            )
            i = 0
            with tqdm() as pbar:
                while True:
                    affective_generator.train(i,AFF_WEIGHT)
                    if i == MAX_ITER:
                        break
                    i += 1
                    pbar.update()