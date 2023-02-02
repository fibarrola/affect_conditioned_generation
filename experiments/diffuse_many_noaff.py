import torch
import clip
import pickle
from src.mlp import MLP
from torchvision.transforms import Resize
from stable_diffusion.scripts.stable_diffuser import StableDiffuser


device = "cuda:0" if torch.cuda.is_available() else "cpu"

NUM_IMGS = 6
BATCH_SIZE = 6

resize = Resize(size=224)
clip_model = clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
mlp = MLP([64, 32]).to(device)
mlp.load_state_dict(torch.load('data/model_mixed.pt'))
with open('data/data_handler_mixed.pkl', 'rb') as f:
    data_handler = pickle.load(f)

PROMPTS = [
    # "A happy forest",
    "forest that makes you happy"
    # "A dark forest",
    # "A house overlooking the sea",
    # "An old house overlooking the sea",
    # "A colorful wild animal",
    # "A large wild animal",
    # "A spaceship",
    # "An UFO",
    # "The sea at night",
    # "The sea at sunrise",
    # "An elephant",
    # "A crocodile",
]
## Initialization
stable_diffuser = StableDiffuser(outdir="results/diff_no_aff")
with torch.no_grad():
    for prompt in PROMPTS:
        start_code = torch.randn([NUM_IMGS, 4, 512 // 8, 512 // 8], device='cpu')
        num_imgs = 0
        while num_imgs < NUM_IMGS:
            stable_diffuser.initialize(
                prompt=prompt,
                start_code=start_code[
                    num_imgs : min(num_imgs + BATCH_SIZE, NUM_IMGS), :, :, :
                ],
            )
            stable_diffuser.run_diffusion()
            num_imgs += BATCH_SIZE
