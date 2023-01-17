import torch
import clip
import os
import plotly.graph_objects as go
from src.utils import imread


PATH = "results/diff_large_set4/single_dim_4/"
PROMPTS = [
    # 'A dog in the forest',
    # 'A crocodile',
    # 'A colourful wild animal',
    # 'A dark forest',
    # 'A forest',
    # 'A house overlooking the sea',
    # 'A large wild animal',
    # 'A spaceship',
    # 'An elephant',
    # 'An UFO',
    # 'The sea at night',
    'The sea at sunrise',
]
PROMPT2 = 'An exciting sea at sunrise'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

prompt = PROMPTS[0]

img_names = os.listdir(PATH+prompt.replace(' ','_'))

tokens = clip.tokenize(PROMPT2).to(device)
z0 = clip_model.encode_text(tokens)

for dim in ['E', 'P', 'A']:
    xx1 = []
    xx2 = []    
    with torch.no_grad():
        for img_name in img_names:
            if img_name[5] == dim:
                x = imread(f"{PATH}{prompt.replace(' ','_')}/{img_name}", min_size=224)
                x = torch.tensor(x)
                x = x.permute(2, 0, 1).unsqueeze(0).to(device)
                z = clip_model.encode_image(x)
                xx1.append(torch.norm(z-z0).detach().item())
            elif img_name[4] == dim:
                x = imread(f"{PATH}{prompt.replace(' ','_')}/{img_name}", min_size=224)
                x = torch.tensor(x)
                x = x.permute(2, 0, 1).unsqueeze(0).to(device)
                z = clip_model.encode_image(x)
                xx2.append(torch.norm(z-z0).detach().item())

    fig = go.Figure()
    fig.add_trace(go.Box(y=xx1, name=f"||z_high{dim}-z_new||"))
    fig.add_trace(go.Box(y=xx2, name=f"||z_low{dim}-z_new||"))
    fig.update_layout(title=f"z = {prompt}, z_new = {PROMPT2}")
    fig.show()
