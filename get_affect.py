from src.mlp import MLP
from src.data_handler_bert_v2 import DataHandlerBERT, load_model_from_config
from omegaconf import OmegaConf
import os
import torch
import pickle
from src.utils import imread, square_crop
import clip
import argparse
from dotenv import load_dotenv

param_env = "bert.env"

load_dotenv(param_env)

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
args = parser.parse_args()

with open('data/data_handler_mixed.pkl', 'rb') as f:
    data_handler = pickle.load(f)

with torch.no_grad():
    mlp = MLP().to('cuda:0')
    mlp.load_state_dict(torch.load('data/model_mixed.pt'))
    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

    if args.img_path is None:
        print(f'Computing affect scores for "{args.prompt}"...')
        tokens = clip.tokenize(args.prompt).to(device)
        z = clip_model.encode_text(tokens).to(torch.float32)
        z = data_handler.scaler_Z.scale(z)

    else:
        print(f'Computing affect scores for "{args.img_path}"...')
        x = imread(args.img_path, min_size=224)
        x = square_crop(x)
        x = torch.tensor(x)
        x = x.permute(2, 0, 1).unsqueeze(0).to(device)
        z = clip_model.encode_image(x)
        z = data_handler.scaler_Z.scale(z)

    mlp.eval()
    output = mlp(z)
    if args.format in ['score', 'latex']:
        output = data_handler.scaler_V.unscale(output)

output = [x.item() for x in output.to('cpu')[0]]

if args.format in ['score', 'uniform']:
    print("V = {:.2f}, A = {:.2f}, D = {:.2f}".format(output[0], output[1], output[2]))
else:
    print(
        "& ${:.2f}$ & ${:.2f}$ & ${:.2f}$ \\\\".format(output[0], output[1], output[2])
    )


data_handler = DataHandlerBERT()

mlp = MLP(
    h0=int(os.environ.get('IMG_SIZE')),
    use_dropout=False,
    use_sigmoid=os.environ.get('USE_SIGMOID'),
).to('cuda:0')

config = OmegaConf.load(os.environ.get("SD_CONFIG"))
model = load_model_from_config(config, os.environ.get("SD_MODEL"))
z_0 = model.get_learned_conditioning([args.prompt]).to('cuda:0')

affect = torch.tensor([[0.0, 0.0, 0.0]]).to('cuda:0')
for channel in range(77):

    mlp.load_state_dict(
        torch.load(f"{os.environ.get('MODEL_PATH')}/model_ch_{channel}.pt")
    )
    affect += mlp(z_0[:, channel, :])

print(affect / 77)
