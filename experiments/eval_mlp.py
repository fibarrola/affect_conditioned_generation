from src.mlp import MLP
import torch
import pickle
import argparse
from torch.nn.functional import sigmoid
import clip

parser = argparse.ArgumentParser(description='Training Parameters')
parser.add_argument("--num_epochs", type=int, default=2500)
parser.add_argument(
    "--scaling",
    type=str,
    default="uniform",
    help="scaling for input and output data. Can be 'uniform', 'whiten', 'normalize' or 'none'",
)
parser.add_argument("--lr", type=float, help="learning rate", default=0.00007)
parser.add_argument(
    "--layer_dims", type=str, help="layer dimensions. Separate with |", default="64|32"
)
parser.add_argument(
    "--use_dropout", type=bool, help="Use dropout for training?", default=True
)
parser.add_argument(
    "--use_sigmoid",
    type=bool,
    help="Use sigmoid at the end of last layer?",
    default=False,
)
config = parser.parse_args()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

with open('data/data_handler_mixed.pkl', 'rb') as f:
    data_handler = pickle.load(f)

with torch.no_grad():
    mlp = MLP([64, 32]).to('cuda:0')
    mlp.load_state_dict(torch.load('data/model_mixed.pt'))
    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

with open('data/data_handler_mixed.pkl', 'wb') as f:
    pickle.dump(data_handler, f)


def threshold_count(x, y, t):
    d = torch.abs(x - y) - t
    d = d.view(-1, 1).squeeze(1)
    d = sigmoid(10000 * d)
    r = torch.sum(d) / len(d)
    return r


# monitor losses
l1_loss_txt = 0
l1_loss_img = 0
txt_losses = [0, 0, 0]
img_losses = [0, 0, 0]
r_txt = 0
r_img = 0

######################
# validate the model #
######################
with torch.no_grad():
    mlp.eval()  # prep model for evaluation
for data, label, sds in data_handler.txt_test_loader:
    output = mlp(data)
    l1_loss_txt += torch.sum(torch.abs(output - label)).item()
    loss_by_component = torch.sum(torch.abs(output - label), dim=0)
    for i in range(3):
        txt_losses[i] += loss_by_component[i].item()

    label = data_handler.scaler_V.unscale(label)
    output = data_handler.scaler_V.unscale(output)
    r_txt += threshold_count(output, label, sds) * data.size(0)


for data, label, sds in data_handler.img_test_loader:
    output = mlp(data)
    l1_loss_img += torch.sum(torch.abs(output - label)).item()
    loss_by_component = torch.sum(torch.abs(output - label), dim=0)
    for i in range(3):
        img_losses[i] += loss_by_component[i].item()

    label = data_handler.scaler_V.unscale(label)
    output = data_handler.scaler_V.unscale(output)
    r_img += threshold_count(output, label, sds) * data.size(0)

txt_losses = [x / len(data_handler.txt_test_loader.sampler) for x in txt_losses]
img_losses = [x / len(data_handler.img_test_loader.sampler) for x in img_losses]
l1_loss_txt = l1_loss_txt / (3 * len(data_handler.txt_test_loader.sampler))
l1_loss_img = l1_loss_img / (3 * len(data_handler.img_test_loader.sampler))
r_txt = r_txt / len(data_handler.txt_test_loader.sampler)
r_img = r_img / len(data_handler.img_test_loader.sampler)


print(
    'txt Loss: {:.3f}, \timg Loss: {:.3f} \ttxt r: {:.4f}, \timg r: {:.4f}'.format(
        l1_loss_txt,
        l1_loss_img,
        1 - r_txt,
        1 - r_img,
    )
)
print(txt_losses)
print(img_losses)
