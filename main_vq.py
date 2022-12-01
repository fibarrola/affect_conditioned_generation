# @title Carga de bibliotecas y definiciones
 
import argparse
from tqdm.notebook import tqdm
from PIL import ImageFile
from pytorch_lightning import seed_everything
from src.affective_generator import AffectiveGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

seed_everything(42)

parser = argparse.ArgumentParser(description='Text Affect Args')
parser.add_argument(
    "--prompt",
    type=str,
    help="Prompt of which to get affect score",
    default="A flaming landscape",
)
parser.add_argument(
    "--EPA",
    type=str,
    help='''
        "e,p,a" format
    ''',
    default=None,
)
config = parser.parse_args()

# A = config.EPA.split(',')
config.v = [0.5,0.,0.5]

affective_generator = AffectiveGenerator()
affective_generator.initialize(prompts='Waves hitting the shoreline', v=[1.,0.5,0.5])
max_iter = 2500
i = 0
try:
    with tqdm() as pbar:
        while True:
            affective_generator.train(i)
            if i == max_iter:
                break
            i += 1
            pbar.update()
except KeyboardInterrupt:
    pass
