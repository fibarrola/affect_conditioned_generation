import pandas as pd
import torch
from utils import imread, square_crop
import clip
import pickle
import sys

device = "cuda:0" if torch.cuda.is_available() else "cpu"

df = pd.read_csv('./data/image_scores.csv')

# ----------- Target values ------------
vals = list(df['valmn'][:])
pots = list(df['aromn'][:])
acts = list(df['dom1mn'][:])
for k in range(len(acts)):
    if acts[k] == '.':
        acts[k] = float(df['dom2mn'][k])
    else:
        acts[k] = float(acts[k])

V = torch.tensor([vals, pots, acts], dtype=torch.float32)

with open("data/img_Vmn.pkl", "wb") as f:
    pickle.dump(V, f)

vals = list(df['valsd'][:])
pots = list(df['arosd'][:])
acts = list(df['dom1sd'][:])
for k in range(len(acts)):
    if acts[k] == '.':
        acts[k] = float(df['dom2sd'][k])
    else:
        acts[k] = float(acts[k])

V = torch.tensor([vals, pots, acts], dtype=torch.float32)

with open("data/img_Vsd.pkl", "wb") as f:
    pickle.dump(V, f)

sys.exit()


# ----------- Image latents ------------
ids = [int(x) for x in list(df['IAPS'])]

zz = []
for i, img_id in enumerate(ids):
    if i % 10 == 0:
        print(i)
    try:
        x = imread(f"./data/Images/{img_id}.jpg", min_size=224)
    except:
        x = imread(f"./data/Images/{img_id}.1.jpg", min_size=224)
    x = square_crop(x)
    with torch.no_grad():
        x = torch.tensor(x)
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
        z = clip_model.encode_image(x).to('cpu')
        zz.append(z)

zz = torch.cat(zz, 0).to("cpu")
print(zz)
print(zz.shape)

with open("data/img_Z.pkl", "wb") as f:
    pickle.dump(zz, f)
