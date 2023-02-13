# @title Carga de bibliotecas y definiciones

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


affective_generator = AffectiveGenerator()
affective_generator.initialize(
    prompts=config.prompt, v=[config.V, config.A, config.D], outdir=config.outdir
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
