# @title Carga de bibliotecas y definiciones
 
from tqdm.notebook import tqdm
from PIL import ImageFile
from pytorch_lightning import seed_everything
from src.affective_generator import AffectiveGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

seed_everything(42)

PROMPT = 'Waves hitting the shoreline'
MAX_ITER = 2500
vv = {
    'high_E': [1., 0.5, 0.5],
    'low_E': [0., 0.5, 0.5],
    'high_P': [0.5, 1., 0.5],
    'low_P': [0.5, 0., 0.5],
    'high_A': [0.5, 0.5, 1.],
    'low_A': [0.5, 0.5, 0.],
}

affective_generator = AffectiveGenerator()
try:
    for v in vv:
        affective_generator.initialize(prompts='Waves hitting the rocks', v=vv[v], im_suffix=str(v))
        i = 0
        
        with tqdm() as pbar:
            while True:
                affective_generator.train(i)
                if i == MAX_ITER:
                    break
                i += 1
                pbar.update()
except KeyboardInterrupt:
    pass
