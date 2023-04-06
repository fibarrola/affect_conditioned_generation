import argparse
from tqdm.notebook import tqdm
from PIL import ImageFile
from src.affective_generator import AffectiveGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Text Affect Args')
parser.add_argument(
    "--prompt",
    type=str,
    help="Prompt of which to get affect score",
    default="A flaming landscape",
)
parser.add_argument(
    "--V", type=float, help="Valence, in [0,1]", default=None,
)
parser.add_argument(
    "--A", type=float, help="Arousal, in [0,1]", default=None,
)
parser.add_argument(
    "--D", type=float, help="Dominance, in [0,1]", default=None,
)
parser.add_argument(
    "--max_iter", type=int, help="max VQGAN iterations", default=1500,
)
parser.add_argument(
    "--outdir", type=str, help="sample output directory", default="results",
)
config = parser.parse_args()
# vv = [
#     [0, 0.9, 0],
#     [0, 0.7, 0],
#     [0, 0.5, 0],
#     [0, 0.3, 0],
#     [0, 0.1, 0],
#     [None, 0.9, None],
#     [None, 0.7, None],
#     [None, 0.5, None],
#     [None, 0.3, None],
#     [None, 0.1, None],
# ]
# vv = [
#     [0.9, 0, 0],
#     [0.7, 0, 0],
#     [0.5, 0, 0],
#     [0.3, 0, 0],
#     [0.1, 0, 0],
#     [0.9, None, None],
#     [0.7, None, None],
#     [0.5, None, None],
#     [0.3, None, None],
#     [0.1, None, None],
# ]
vv = [
    [0, 0, 0.9],
    [0, 0, 0.7],
    [0, 0, 0.5],
    [0, 0, 0.3],
    [0, 0, 0.1],
    [None, None, 0.9],
    [None, None, 0.7],
    [None, None, 0.5],
    [None, None, 0.3],
    [None, None, 0.1],
]
prompts = [
    "the sea at nightfall",
    "a forest",
    "flaming landscape",
    "a city skyline",
]

for prompt in prompts:
    for v in vv:
        print(f"running {prompt} with v={v}")

        affective_generator = AffectiveGenerator()
        affective_generator.initialize(
            prompts=prompt, v=v, outdir=config.outdir
        )
        i = 0
        try:
            with tqdm() as pbar:
                while True:
                    affective_generator.train(i)
                    if i == config.max_iter:
                        break
                    i += 1
                    pbar.update()
        except KeyboardInterrupt:
            pass