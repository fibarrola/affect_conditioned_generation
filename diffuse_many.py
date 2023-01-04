import pandas as pd
import torch
from src.utils import imread, square_crop
import clip
import pickle
import sys
from os import listdir
from os.path import isfile, join
from src.mlp import MLP
from src.utils import N_max_elements
import numpy as np
import cv2
from torchvision.transforms import Resize
from src.utils import N_max_elements

from stable_diffusion.scripts.stable_diffuser import StableDiffuser
device = "cuda:0" if torch.cuda.is_available() else "cpu"

NUM_IMGS = 12
BATCH_SIZE = 5
MAX_ITER = 5

resize = Resize(size=224)
clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
mlp = MLP([64, 32]).to(device)
mlp.load_state_dict(torch.load('data/model_mixed.pt'))
with open('data/data_handler_mixed.pkl', 'rb') as f:
    data_handler = pickle.load(f)

PROMPTS = [
    'A dog in the forest',
    'An old building',
    'A fish swimming in the sea',
    'A tree on a hilltop',
    'A meal on a white plate'
]
Vs = {
    'high_E': [1.0, 0.5, 0.5],
    'low_E': [0.0, 0.5, 0.5],
    'high_P': [0.5, 1.0, 0.5],
    'low_P': [0.5, 0.0, 0.5],
    'high_A': [0.5, 0.5, 1.0],
    'low_A': [0.5, 0.5, 0.0],
}
## Initialization
stable_diffuser = StableDiffuser(outdir="results/st_diff_010")
with torch.no_grad():
    for prompt in PROMPTS:
        start_code = torch.randn([NUM_IMGS, 4, 512 // 8, 512 // 8], device='cpu')
        for v_name in Vs:
            num_imgs = 0
            while num_imgs<NUM_IMGS:
                stable_diffuser.initialize(prompt=prompt, start_code=start_code[num_imgs:min(num_imgs+BATCH_SIZE, NUM_IMGS),:,:,:])
                with open(f"data/diff_embeddings/{prompt.replace(' ','_')}_{v_name}_010.pkl", 'rb') as f:
                    z = pickle.load(f)
                stable_diffuser.override_zz(z)
                stable_diffuser.run_diffusion(suffix=v_name)
                num_imgs += BATCH_SIZE
  