from src.affective_generator2 import AffectiveGenerator
import torch
import os
from src.clipdraw import CLIPAffDraw
import pydiffvg
from pytorch_lightning import seed_everything
from src.utils import checked_path

device = "cuda:0" if torch.cuda.is_available() else "cpu"

MAX_ITER = 800
AFF_WEIGHT = 0.3
FOLDER = "results/clipdraw_R1"
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

affective_generator = AffectiveGenerator()

for prompt in PROMPTS:
    os.makedirs(f"{FOLDER}/{prompt.replace(' ','_')}", exist_ok=True)

    for sample in range(N_SAMPLES):
        noise_0 = torch.randint(
            affective_generator.n_toks,
            [affective_generator.toksY * affective_generator.toksX],
            device=device,
        )
        default_affect = affective_generator.get_affect(prompt)[0]
        dists = torch.min(0.95*torch.ones((3), device=device)-default_affect, default_affect-0.05*torch.ones((3), device=device))
        for aff_idx in range(3):
            print(f"Generating {prompt} -- default affect {round(100*default_affect[0].item())}...")
            for tick in range(5):
                aff_val = default_affect[aff_idx]-(1-0.5*tick)*dists[aff_idx]
                v_name = f"{aff_names[aff_idx]}_{round(100*aff_val.item())}"
                print(f"Generating {prompt} with affect {v_name}...")

                seed_everything(1)

                cicada = CLIPAffDraw(aff_weight=AFF_WEIGHT)
                cicada.process_text(
                    prompt=prompt,
                    use_affect= tick != 2,
                    neg_prompt_1="Written words.",
                    neg_prompt_2="Text",
                    v=[aff_val],
                    aff_idx=aff_idx,
                )
                cicada.add_random_shapes(256)
                cicada.initialize_variables()
                cicada.initialize_optimizer()

                # Run the main optimization loop
                for t in range(500):
                    cicada.run_epoch(t)
                filepath = checked_path(
                    f"{FOLDER}/{prompt.replace(' ','_')}/{sample}_{v_name}",
                    "png",
                )

                pydiffvg.imwrite(
                    cicada.img, filepath, gamma=1,
                )


