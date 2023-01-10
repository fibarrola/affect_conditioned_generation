import torch
import clip
import pickle
from src.mlp import MLP
from torchvision.transforms import Resize
from stable_diffusion.scripts.stable_diffuser import StableDiffuser


device = "cuda:0" if torch.cuda.is_available() else "cpu"

NUM_IMGS = 100
BATCH_SIZE = 6

resize = Resize(size=224)
clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
mlp = MLP([64, 32]).to(device)
mlp.load_state_dict(torch.load('data/model_mixed.pt'))
with open('data/data_handler_mixed.pkl', 'rb') as f:
    data_handler = pickle.load(f)

PROMPTS = [
    # "A dog in the forest",
    # "A meal on a white plate",
    # "A tree on a hilltop",
    "A forest",
    "A dark forest",
    "A house overlooking the sea",
    "An old house overlooking the sea",
    "A colorful wild animal",
    "A large wild animal",
    "A spaceship",
    "An UFO",
    "The sea at night",
    "The sea at sunrise",
    "An elephant",
    "A crocodile",
]
Vs = {
    'high_E': [1.0, 0.5, 0.5],
    'low_E': [0.0, 0.5, 0.5],
    'high_P': [0.5, 1.0, 0.5],
    'low_P': [0.5, 0.0, 0.5],
    'high_A': [0.5, 0.5, 1.0],
    'low_A': [0.5, 0.5, 0.0],
    'no_aff': [],
}
## Initialization
stable_diffuser = StableDiffuser(outdir="results/diff_large_set")
with torch.no_grad():
    start_code = torch.randn([NUM_IMGS, 4, 512 // 8, 512 // 8], device='cpu')
    for v_name in Vs:
        for prompt in PROMPTS:
            num_imgs = 0
            while num_imgs < NUM_IMGS:
                stable_diffuser.initialize(
                    prompt=prompt,
                    start_code=start_code[
                        num_imgs : min(num_imgs + BATCH_SIZE, NUM_IMGS), :, :, :
                    ],
                )
                if not v_name == 'no_aff':
                    with open(
                        f"data/diff_embeddings/{prompt.replace(' ','_')}_{v_name}_010_03.pkl",
                        'rb',
                    ) as f:
                        z = pickle.load(f)
                    stable_diffuser.override_zz(z)
                stable_diffuser.run_diffusion(suffix=v_name)
                num_imgs += BATCH_SIZE
