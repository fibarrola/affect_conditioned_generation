from src.mlp import MLP
import torch
import pickle
from src.utils import imread, square_crop
import clip
import argparse

device = "cuda:0" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser(description='Text Affect Args')
parser.add_argument(
    "--prompt",
    type=str,
    help="Prompt of which to get affect score",
    default="A close-up snake",
)
parser.add_argument(
    "--img_path", type=str, help="Path of image to evaluate", default=None,
)
parser.add_argument(
    "--format",
    type=str,
    help="display format: 'score', 'uniform' or 'latex'",
    default='score',
)
config = parser.parse_args()

with open('data/data_handler_mixed.pkl', 'rb') as f:
    data_handler = pickle.load(f)

with torch.no_grad():
    mlp = MLP([64, 32]).to('cuda:0')
    mlp.load_state_dict(torch.load('data/model_mixed.pt'))
    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

    if config.img_path is None:
        print(f'Computing affect scores for "{config.prompt}"...')
        tokens = clip.tokenize(config.prompt).to(device)
        z = clip_model.encode_text(tokens).to(torch.float32)
        z = data_handler.scaler_Z.scale(z)

    else:
        print(f'Computing affect scores for "{config.img_path}"...')
        x = imread(config.img_path, min_size=224)
        x = square_crop(x)
        x = torch.tensor(x)
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        z = clip_model.encode_image(x)
        z = data_handler.scaler_Z.scale(z)

    mlp.eval()
    output = mlp(z)
    if config.format in ['score', 'latex']:
        output = data_handler.scaler_V.unscale(output)

output = [x.item() for x in output.to('cpu')[0]]

if config.format in ['score', 'uniform']:
    print("V = {:.2f}, A = {:.2f}, D = {:.2f}".format(output[0], output[1], output[2]))
else:
    print(
        "& ${:.2f}$ & ${:.2f}$ & ${:.2f}$ \\\\".format(output[0], output[1], output[2])
    )
