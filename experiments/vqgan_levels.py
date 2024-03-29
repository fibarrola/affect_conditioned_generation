from src.affective_generator import AffectiveGenerator
import torch
import os
from tqdm.notebook import tqdm
from src.utils import renum_path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_ITER = 500
AFF_WEIGHT = 7
FOLDER = "results/vqgan_survey_0"
N_SAMPLES = 3
AFFECT_VALS = [0.03, 0.25, 0.5, 0.75, 0.97, None]
PROMPTS = [
    "A castle",
    "A university",
    "A storm",
    "A Village",
    "An earthquake",
    "A canyon",
    "A ship",
    "pirates",
    "A Mountain",
    "A House",
    "The Sea",
    "A Beach",
    "A City",
    "A house overlooking the sea",
    "A house on fire",
    "A volcano",
]

# MAIN starts here
folder = renum_path(FOLDER)

aff_names = ["V", "A", "D"]

criterion = torch.nn.MSELoss(reduction='mean')

affective_generator = AffectiveGenerator()

for prompt in PROMPTS:
    os.makedirs(f"{folder}/{prompt.replace(' ','_')}", exist_ok=True)

    for sample in range(N_SAMPLES):
        noise_0 = torch.randint(
            affective_generator.n_toks,
            [affective_generator.toksY * affective_generator.toksX],
            device=device,
        )
        default_affect = affective_generator.get_affect(prompt)[0]
        dists = torch.min(
            0.95 * torch.ones((3), device=device) - default_affect,
            default_affect - 0.05 * torch.ones((3), device=device),
        )
        for aff_idx in range(3):
            print(
                f"Generating {prompt} -- default affect {round(100*default_affect[0].item())}..."
            )
            for aff_val in AFFECT_VALS:
                aff_name = str(round(100 * aff_val)) if aff_val is not None else "noaff"
                v_name = f"{aff_names[aff_idx]}_{aff_name}"
                print(f"Generating {prompt} with affect {v_name}...")
                aff_vec = [None, None, None]
                if aff_val is not None:
                    aff_vec[aff_idx] = aff_val
                affective_generator.initialize(
                    prompts=prompt,
                    v=aff_vec,
                    savepath=f"{folder}/{prompt.replace(' ','_')}/{sample}_{v_name}.png",
                    seed=20 + sample,
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
