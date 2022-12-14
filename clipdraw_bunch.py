from src.clipdraw import CLIPAffDraw
import torch
import pydiffvg
import datetime
import time
from pathlib import Path
import argparse
from pytorch_lightning import seed_everything
import os
from src.utils import checked_path


parser = argparse.ArgumentParser(description='Sketching Agent Args')

# CLIP prompts
parser.add_argument("--prompt", type=str, help="what to draw", default="A red chair.")
parser.add_argument("--neg_prompt", type=str, default="Written words.")
parser.add_argument("--neg_prompt_2", type=str, default="Text.")
parser.add_argument("--use_neg_prompts", type=bool, default=True)
parser.add_argument("--normalize_clip", type=bool, default=True)

# Canvas parameters
parser.add_argument(
    "--num_paths", type=int, help="number of strokes to add", default=512
)
parser.add_argument("--canvas_h", type=int, help="canvas height", default=224)
parser.add_argument("--canvas_w", type=int, help="canvas width", default=224)
parser.add_argument("--max_width", type=int, help="max px width", default=15)

# Algorithm parameters
parser.add_argument(
    "--num_iter", type=int, help="maximum algorithm iterations", default=1500
)
parser.add_argument(
    "--num_trials", type=int, help="number of times to run the algorithm", default=1
)
parser.add_argument(
    "--num_augs",
    type=int,
    help="number of augmentations for computing semantic loss",
    default=4,
)
# Saving
parser.add_argument(
    "--save_path", type=str, help="subfolder for saving results", default="clipdraw2"
)

args = parser.parse_args()


device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'

# Build dir if does not exist & make sure using a
# trailing / or not does not matter
save_path = Path("results/").joinpath(args.save_path)
save_path.mkdir(parents=True, exist_ok=True)
save_path = str(save_path) + '/'


t0 = time.time()

PROMPTS = [
    'Waves hitting the rocks',
    'The sea at nightfall',
    'A dark forest',
    'A windy night',
    'flaming landscape',
]
vv = {
    'high_E': [1.0, 0.5, 0.5],
    'low_E': [0.0, 0.5, 0.5],
    'high_P': [0.5, 1.0, 0.5],
    'low_P': [0.5, 0.0, 0.5],
    'high_A': [0.5, 0.5, 1.0],
    'low_A': [0.5, 0.5, 0.0],
}

for prompt in PROMPTS:
    for v in vv:
        seed_everything(1)

        cicada = CLIPAffDraw()
        cicada.process_text(
            prompt=prompt, neg_prompt_1="Written words.", neg_prompt_2="Text", v=vv[v],
        )

        time_str = (datetime.datetime.today() + datetime.timedelta(hours=11)).strftime(
            "%Y_%m_%d_%H_%M_%S"
        )

        cicada.add_random_shapes(args.num_paths)
        cicada.initialize_variables()
        cicada.initialize_optimizer()

        # Run the main optimization loop
        for t in range(args.num_iter):

            if (t + 1) % (args.num_iter // 50) == 0:
                print(
                    'Step: {} \t Loss: {:.3f} \t Semantic Loss: {:.3f} \t Affective Loss: {:.3f}'.format(
                        t + 1,
                        cicada.losses['global'],
                        cicada.losses['semantic'],
                        cicada.losses['affective'],
                    )
                )
            cicada.run_epoch(t)

        filepath = checked_path(f"{save_path}{prompt.replace(' ','_')}_{v}_0", "png")

        pydiffvg.imwrite(
            cicada.img, filepath, gamma=1,
        )

        time_sec = round(time.time() - t0)
        print(
            f"Elapsed time: {time_sec//60} min, {time_sec-60*(time_sec//60)} seconds."
        )
