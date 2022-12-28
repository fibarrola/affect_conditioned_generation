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

from stable_diffusion.scripts.txt2img_mod import StableDiffuser
device = "cuda:0" if torch.cuda.is_available() else "cpu"

NUM_IMGS = 100
BATCH_SIZE = 6

stable_diffuser = StableDiffuser()
start_code = torch.randn([NUM_IMGS, 4, 512 // 8, 512 // 8], device='cpu')
num_imgs = 0
while num_imgs<NUM_IMGS:
    stable_diffuser.initialize(prompt="A dog in the forest", start_code=start_code[num_imgs:min(num_imgs+BATCH_SIZE, NUM_IMGS),:,:,:])
    stable_diffuser.run_diffusion()
    num_imgs += BATCH_SIZE

# assert False
# device = "cuda:0" if torch.cuda.is_available() else "cpu"

# PATH = "results/diffusion/dog_in_forest"
# img_names = listdir(PATH)

# mlp = MLP([64, 32]).to(device)
# mlp.load_state_dict(torch.load('data/model_mixed.pt'))

# with open('data/data_handler_mixed.pkl', 'rb') as f:
#     data_handler = pickle.load(f)

# xx = torch.empty((len(img_names), 3, 224, 224), device = device)
# with torch.no_grad():
#     for k, img_name in enumerate(img_names):
#         x = imread(f"results/diffusion/dog_in_forest/{img_name}", min_size=224)
#         x = torch.tensor(x).permute(2,0,1)
#         xx[k, :, :, :] = x
#     print('xx shape', xx.shape)
#     clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
#     zz = clip_model.encode_image(xx).to(torch.float32)
#     zz = data_handler.scaler_Z.scale(zz)
#     affs = mlp(zz).to('cpu')

# print('zz shape', zz.shape)
# print('aff shape', affs.shape)

# for k, img_name in enumerate(img_names):
#     # print(f"{img_name}, E {affs[k,0].item()}, P {affs[k,1].item()}, A {affs[k,2].item()}")
#     print("{}, E {:.2f}, P {:.2f}, A {:.2f}".format(img_name, affs[k,0].item(), affs[k,1].item(), affs[k,2].item()))

# for k, aff_name in enumerate(['E', 'P', 'A']): 
#     print('')
#     ind = torch.argmax(affs[:,k])
#     print(f'Max {aff_name}: {img_names[ind]}')
#     ind = torch.argmin(affs[:,k])
#     print(f'Min {aff_name}: {img_names[ind]}')

# for k, aff_name in enumerate(['E', 'P', 'A']): 
#     print('')
#     vals, inds = N_max_elements(affs[:,k], 3)
#     print(f'Max {aff_name}: {img_names[inds[0]]}, {img_names[inds[1]]}, {img_names[inds[2]]}')
#     vals, inds = N_max_elements(2-affs[:,k], 3)
#     print(f'Min {aff_name}: {img_names[inds[0]]}, {img_names[inds[1]]}, {img_names[inds[2]]}')