from src.mlp import MLP
import torch
import pickle
from src.utils import imread, square_crop
import clip
import pandas as pd
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Image Affect Args')
parser.add_argument(
    "--img_id",
    type=int,
    help="id of image to evaluate",
    default=2030,
)
config = parser.parse_args()

with open('data/data_handler_mixed.pkl', 'rb') as f:
    data_handler = pickle.load(f)

mlp = MLP([64, 32]).to('cuda:0')
mlp.load_state_dict(torch.load('outputs/model_mixed.pt'))

df = pd.read_csv('./data/image_scores.csv')
df_i = df[df['IAPS']==img_id]
target_v = [df_i['valmn'].item(), df_i['aromn'].item(), df_i['dom1mn'].item()]

x = imread(f"./data/Images/{config.img_id}.jpg", min_size=224)
x = square_crop(x)

with torch.no_grad():
    x = torch.tensor(x)
    x = x.permute(2, 0, 1).unsqueeze(0).to(device)
    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
    z = clip_model.encode_image(x)
    z = z.unsqueeze(0)
    z = data_handler.scaler_Z.scale(z)
    mlp.eval()
    output = mlp(z)
    output = data_handler.scaler_V.unscale(output)

target_v = [float(x) for x in target_v]
output = [x.item() for x in output.to('cpu')[0,0]]
print("target: $E={:.2f}, P={:.2f}, A={:.2f}$".format(target_v[0], target_v[1], target_v[2]))
print("output: $E={:.2f}, P={:.2f}, A={:.2f}$".format(output[0], output[1], output[2]))
