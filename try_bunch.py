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
    'A strange animal',
    'A windy night',
    'flaming landscape',
]
MAX_ITER = 1500
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
        for v in vv:
            seed_everything(trial)
            affective_generator.initialize(
                prompts=prompt,
                v=vv[v],
                im_suffix=f"{trial}_{v}",
                seed=trial,
                noise_0=noise_0,
            )
            i = 0

            with tqdm() as pbar:
                while True:
                    affective_generator.train(i)
                    if i == MAX_ITER:
                        break
                    i += 1
                    pbar.update()
