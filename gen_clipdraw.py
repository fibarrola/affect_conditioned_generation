from src.clipdraw import CLIPAffDraw
import pydiffvg
from pathlib import Path
import argparse
import os

parser = argparse.ArgumentParser(description='Sketching Agent Args')

# CLIP prompts
parser.add_argument("--prompt", type=str, help="what to draw", default="A red chair.")
parser.add_argument(
    "--num_paths", type=int, help="number of strokes to add", default=512
)
parser.add_argument("--max_width", type=int, help="max px width", default=20)
parser.add_argument(
    "--num_iter", type=int, help="maximum algorithm iterations", default=1500
)
parser.add_argument(
    "--E", type=float, help="Evaluation (bad<good) in [0,1]", default=0.5,
)
parser.add_argument(
    "--P", type=float, help="Potency (weak<strong) in [0,1]", default=0.5,
)
parser.add_argument(
    "--A", type=float, help="Activity (calm<exciting) in [0,1]", default=0.5,
)
parser.add_argument(
    "--save_path", type=str, help="subfolder for saving results", default="clipdraw"
)
args = parser.parse_args()


# Build dir if does not exist & make sure using a
# trailing / or not does not matter
save_path = Path("results/").joinpath(args.save_path)
save_path.mkdir(parents=True, exist_ok=True)
save_path = str(save_path) + '/'

cicada = CLIPAffDraw()
cicada.process_text(
    prompt=args.prompt,
    neg_prompt_1="Written words.",
    neg_prompt_2="Text",
    v=[args.E, args.P, args.A],
)

cicada.add_random_shapes(args.num_paths)
cicada.initialize_variables(max_width=args.max_width)
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

k = 0
while os.path.exists(f"{save_path}{args.prompt.replace(' ','_')}_{k}.png"):
    k += 1
pydiffvg.imwrite(
    cicada.img, f"{save_path}{args.prompt.replace(' ','_')}_{k}.png", gamma=1,
)
