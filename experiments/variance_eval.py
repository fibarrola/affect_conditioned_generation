import torch
from src.utils import imread
import clip
import pickle
import os
from src.mlp import MLP


device = "cuda:0" if torch.cuda.is_available() else "cpu"

PATH = "results/diff_no_aff"
img_names = os.listdir(PATH)

with open('data/data_handler_mixed.pkl', 'rb') as f:
    data_handler = pickle.load(f)

mlp = MLP([64, 32]).to(device)
mlp.load_state_dict(torch.load('data/model_mixed.pt'))

clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)
for img_name in img_names:
    filenames = os.listdir(f"{PATH}/{img_name}")
    with torch.no_grad():
        xx = torch.empty((len(filenames), 3, 224, 224), device=device)
        for n, filename in enumerate(filenames):
            x = imread(f"{PATH}/{img_name}/{filename}", min_size=224)
            x = torch.tensor(x).permute(2, 0, 1)
            xx[n, :, :, :] = x

        zz = clip_model.encode_image(xx).to(torch.float32)
        zz = data_handler.scaler_Z.scale(zz)
        # sys.exit()
        affs = mlp(zz).to('cpu')

        means = torch.mean(affs, dim=0).to('cpu').tolist()
        sdevs = torch.std(affs, dim=0).to('cpu').tolist()
        print(f"\n {img_name.replace('_',' ')}")
        print(
            "means: \tE {:.3f} \tP {:.3f} \tA {:.3f}".format(
                means[0], means[1], means[2]
            )
        )
        print(
            "sdevs: \tE {:.3f} \tP {:.3f} \tA {:.3f}".format(
                sdevs[0], sdevs[1], sdevs[2]
            )
        )

    with open(f'results/dif_no_aff_encodings/aff_{img_name}.pkl', 'wb') as f:
        pickle.dump(affs, f)
    with open(f'results/dif_no_aff_encodings/zz_{img_name}.pkl', 'wb') as f:
        pickle.dump(zz.to('cpu'), f)

    # torch.cuda.empty_cache()
    # sys.exit()
