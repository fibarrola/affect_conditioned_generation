from src.clipdraw import CLIPAffDraw
import torch
import pydiffvg
import datetime
import time
from pathlib import Path
import argparse
from pytorch_lightning import seed_everything
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
parser.add_argument("--max_width", type=int, help="max px width", default=5)

# Algorithm parameters
parser.add_argument(
    "--num_iter", type=int, help="maximum algorithm iterations", default=2000
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
    "--save_path", type=str, help="subfolder for saving results", default="clipdraw11"
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
    'A volcano',
    'A large rainforest',
    'butterflys',
    # 'Going downriver',
    'A remote island',
    'A treasure map',
    'An old temple',
    'A dream',
    # 'A cloudy day in the field',
]
# vv = {
#     'no_aff': [None, None, None],
#     'high_E': [1.0, 0.5, 0.5],
#     'low_E': [0.0, 0.5, 0.5],
#     'high_P': [0.5, 1.0, 0.5],
#     'low_P': [0.5, 0.0, 0.5],
#     'high_A': [0.5, 0.5, 1.0],
#     'low_A': [0.5, 0.5, 0.0],
# }
Vs = {
    'high_E': [0, 1.0],
    'low_E': [0, 0.0],
    'high_P': [1, 1.0],
    'low_P': [1, 0.0],
    'high_A': [2, 1.0],
    'low_A': [2, 0.0],
    'no_aff': [-1, None],
}
N_TRIALS = 3
for trial in range(N_TRIALS):
    for aff_weight in [0.1, 1]:
        for prompt in PROMPTS:
            for v_name in Vs:
                seed_everything(1)

                cicada = CLIPAffDraw(aff_weight=aff_weight)
                cicada.process_text(
                    prompt=prompt,
                    neg_prompt_1="Written words.",
                    neg_prompt_2="Text",
                    v=[Vs[v_name][1]],
                    aff_idx=Vs[v_name][0],
                )

                time_str = (
                    datetime.datetime.today() + datetime.timedelta(hours=11)
                ).strftime("%Y_%m_%d_%H_%M_%S")

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
                filepath = checked_path(
                    f"{save_path}w{10*aff_weight}_{prompt.replace(' ','_')}_{trial}_{v_name}_0",
                    "png",
                )

                pydiffvg.imwrite(
                    cicada.img, filepath, gamma=1,
                )
                time_sec = round(time.time() - t0)
                print(
                    f"Elapsed time: {time_sec//60} min, {time_sec-60*(time_sec//60)} seconds."
                )
