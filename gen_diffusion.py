import pickle
import torch
import copy
import argparse
from src.mlp import MLP
from stable_diffusion.scripts.stable_diffuser import StableDiffuser


parser = argparse.ArgumentParser(description='Affect-Conditioned Stable Diffusion')

parser.add_argument("--prompt", type=str, help="what to draw", default="A forest.")
parser.add_argument("--reg", type=int, help="regularization parameter", default=0.3)
parser.add_argument("--max_iter", type=int, help="Z search iterations", default=500)
parser.add_argument(
    "--V", type=float, help="Valence, in [0,1]", default=None,
)
parser.add_argument(
    "--A", type=float, help="Arousal, in [0,1]", default=None,
)
parser.add_argument(
    "--D", type=float, help="Dominance, in[0,1]", default=None,
)
parser.add_argument(
    "--save_path", type=str, help="subfolder for saving results", default="st_diff"
)
args = parser.parse_args()


device = "cuda:0" if torch.cuda.is_available() else "cpu"

target_v = [args.V, args.A, args.D]
target_dims = [k for k in range(3) if not target_v[k] is None]
target_v = torch.tensor([0.5 if v is None else v for v in target_v], device=device, requires_grad=False)

with open(f'data/bert_nets/data_handler_bert_{0}.pkl', 'rb') as f:
    data_handler = pickle.load(f)

mlp = MLP([64, 32], do=True, sig=False, h0=768).to(device)
criterion = torch.nn.MSELoss(reduction='mean')

z_0 = data_handler.model.get_learned_conditioning([args.prompt])
z_0.requires_grad = False

zz = torch.zeros_like(z_0)
for channel in range(77):
    print(f"Adjusting Channel {channel} ...")

    with open(f'data/bert_nets/data_handler_bert_{channel}.pkl', 'rb') as f:
        data_handler = pickle.load(f)
    with torch.no_grad():
        mlp.load_state_dict(torch.load(f'data/bert_nets/model_{channel}.pt'))

    z = copy.deepcopy(z_0[:, channel, :])
    z.requires_grad = True

    opt = torch.optim.Adam([z], lr=0.1)

    v_0 = mlp(z_0[0, channel, :])
    if len(target_dims) > 0:
        for iter in range(args.max_iter):
            opt.zero_grad()
            loss = 0
            loss += criterion(z, z_0[:, channel, :])
            for dim in target_dims:
                loss += args.reg * criterion(
                    mlp(data_handler.scaler_Z.scale(z))[:, dim], target_v[dim]
                )
            loss.backward()
            opt.step()

    with torch.no_grad():
        zz[0, channel, :] = copy.deepcopy(z.detach())

zz = zz.to('cpu')

stable_diffuser = StableDiffuser(outdir=f"results/{args.save_path}/")
stable_diffuser.initialize(prompt=args.prompt,)
stable_diffuser.override_zz(zz)
stable_diffuser.run_diffusion()
