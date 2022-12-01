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
    "--E", type=float, help="Evaluation (bad<good) in [0,1]", default=None,
)
parser.add_argument(
    "--P", type=float, help="Potency (weak<strong) in [0,1]", default=None,
)
parser.add_argument(
    "--A", type=float, help="Activity (calm<exciting) in [0,1]", default=None,
)
parser.add_argument(
    "--max_iter", type=int, help="Activity (calm<exciting) in [0,1]", default=2500,
)
config = parser.parse_args()


affective_generator = AffectiveGenerator()
affective_generator.initialize(prompts=config.prompt, v=[config.E, config.P, config.A])
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
