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
BATCH_SIZE = 6
MAX_ITER = 5

resize = Resize(size=224)
clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
mlp = MLP([64, 32]).to(device)
mlp.load_state_dict(torch.load('data/model_mixed.pt'))
with open('data/data_handler_mixed.pkl', 'rb') as f:
    data_handler = pickle.load(f)

Vs = {
    'high_E': 0,
    'low_E': 0,
    'high_P': 1,
    'low_P': 1,
    'high_A': 2,
    'low_A': 2,
}
PROMPTS = [
    'A dog in the forest',
    'An old building',
    'A fish swimming in the sea',
    'A tree on a hilltop',
    'A meal on a white plate'
]
## Initialization
stable_diffuser = StableDiffuser()
num_imgs = 0
with torch.no_grad():
    for prompt in PROMPTS:
        for v_name in Vs:
            start_code = torch.randn([NUM_IMGS, 4, 512 // 8, 512 // 8], device='cpu')
            init_images = torch.empty([NUM_IMGS, 3, 224, 224], device=device)
            while num_imgs<NUM_IMGS:
                stable_diffuser.initialize(prompt=prompt, start_code=start_code[num_imgs:min(num_imgs+BATCH_SIZE, NUM_IMGS),:,:,:])
                stable_diffuser.run_diffusion(save=False)
                images = stable_diffuser.img_batch
                images = resize(images)
                init_images[num_imgs:num_imgs+images.shape[0],:,:,:] = images
                num_imgs += BATCH_SIZE
                zz = clip_model.encode_image(init_images).to(torch.float32)
                vv = mlp(data_handler.scaler_Z.scale(zz))

            for i in range(MAX_ITER):
                _, high_inds = N_max_elements(vv[:,Vs[v_name]], 6)
                _, low_inds = N_max_elements(2-vv[:,Vs[v_name]], 6)
                center_high = torch.mean(start_code[tuple(high_inds), :,:,:], dim=0).unsqueeze(0)
                center_low = torch.mean(start_code[tuple(low_inds), :,:,:], dim=0).unsqueeze(0)
                diff = torch.norm(center_high-center_low)
                if v_name[0] =='h':
                    new_noise = center_high + (0.2*diff/np.sqrt(4*64*64))*torch.randn([6, 4, 512 // 8, 512 // 8], device='cpu')
                    # new_noise = start_code[tuple(high_inds), :,:,:] + 0.1*torch.randn_like(start_code[tuple(high_inds), :,:,:] )
                    start_code[tuple(low_inds), :,:,:] = new_noise
                else:
                    new_noise = center_low + (0.2*diff/np.sqrt(4*64*64))*torch.randn([6, 4, 512 // 8, 512 // 8], device='cpu')
                    # new_noise = start_code[tuple(low_inds), :,:,:] + 0.1*torch.randn_like(start_code[tuple(low_inds), :,:,:] )
                    start_code[tuple(high_inds), :,:,:] = new_noise

                stable_diffuser.initialize(prompt=prompt, start_code=new_noise)
                stable_diffuser.run_diffusion(save=False)
                images = stable_diffuser.img_batch
                images = resize(images)
                zz = clip_model.encode_image(images).to(torch.float32)
                if v_name[0] =='h':
                    vv[tuple(low_inds),:] = mlp(data_handler.scaler_Z.scale(zz))
                else:
                    vv[tuple(high_inds),:] = mlp(data_handler.scaler_Z.scale(zz))
            
            if v_name[0] =='h':
                _, high_inds = N_max_elements(vv[:,Vs[v_name]], 6)
                new_noise = start_code[tuple(high_inds), :,:,:]
            else:
                _, low_inds = N_max_elements(2-vv[:,Vs[v_name]], 6)
                new_noise = start_code[tuple(low_inds), :,:,:]

            stable_diffuser.initialize(prompt=prompt, start_code=new_noise)
            stable_diffuser.run_diffusion(save=True)


    


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