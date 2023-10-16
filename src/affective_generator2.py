import torch
import torch.nn.functional as F
import os
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm
import clip
from PIL import ImageFile, Image
import pickle
from src.mlp import MLP
from src.vqclip_utils import (
    MakeCutouts,
    Prompt,
    parse_prompt,
    resize_image,
    vector_quantize,
    clamp_with_grad,
    add_stegano_data,
    add_xmp_data,
)
from src.vqclip_utils import load_vqgan_model

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AffectiveGenerator:
    @torch.no_grad()
    def __init__(
        self,
        vqg_name="vqgan_model/vqgan_imagenet_f16_16384",
        im_size=[480, 480],
        cutn=64,
    ):
        self.mlp = MLP().to('cuda:0')
        self.mlp.load_state_dict(torch.load('data/model_mixed.pt'))
        with open('data/data_handler_mixed.pkl', 'rb') as f:
            self.data_handler = pickle.load(f)
        self.vqg_model = load_vqgan_model(vqg_name + '.yaml', vqg_name + '.ckpt').to(
            device
        )
        self.im_size = im_size
        self.clip_model = (
            clip.load('ViT-B/32', jit=False)[0].eval().requires_grad_(False).to(device)
        )
        self.e_dim = self.vqg_model.quantize.e_dim

        cut_size = self.clip_model.visual.input_resolution
        self.make_cutouts = MakeCutouts(cut_size, cutn, cut_pow=1.0)
        self.z_min = self.vqg_model.quantize.embedding.weight.min(dim=0).values[
            None, :, None, None
        ]
        self.z_max = self.vqg_model.quantize.embedding.weight.max(dim=0).values[
            None, :, None, None
        ]
        self.n_toks = self.vqg_model.quantize.n_e
        f = 2 ** (self.vqg_model.decoder.num_resolutions - 1)
        self.toksX, self.toksY = im_size[0] // f, im_size[1] // f
        self.sideX, self.sideY = self.toksX * f, self.toksY * f

    @torch.no_grad()
    def process_epa(self, v, prompts):
        if v[0] is None and v[1] is None and v[2] is None:
            return []
        if v[0] is None or v[1] is None or v[2] is None:
            tokens = clip.tokenize(prompts).to(device)
            z = self.clip_model.encode_text(tokens).to(torch.float32)
            z = self.data_handler.scaler_Z.scale(z)

            v0 = self.mlp(z)
            for k in range(3):
                if not (v[k]):
                    v[k] = v0[0, k].to('cpu').item()

        return torch.matmul(
            torch.ones((self.make_cutouts.cutn, 1), device=device),
            torch.tensor([v], device=device, requires_grad=False),
        )

    @torch.no_grad()
    def get_affect(self, prompt):
        tokens = clip.tokenize([prompt]).to(device)
        z = self.clip_model.encode_text(tokens).to(torch.float32)
        z = self.data_handler.scaler_Z.scale(z)
        return self.mlp(z)

    @torch.no_grad()
    def initialize(
        self,
        prompts,
        v=[None, None, None],
        img_0=None,
        target_imgs=None,
        seed=None,
        lr=0.1,
        noise_prompt_seeds=[],
        noise_prompt_weights=[],
        outdir='results',
        noise_0=[],
        savepath=None,
    ):

        # avoid overwriting
        if savepath is None:
            k = 0
            while os.path.exists(f"{outdir}/{prompts.replace(' ','_')}_{k}.png"):
                k += 1
            self.img_savedir = f"{outdir}/{prompts.replace(' ','_')}_{k}.png"
        else:
            self.img_savedir = savepath

        # split prompt if multiple
        prompts = [prompt.strip() for prompt in prompts.split("|")]

        self.target_affect = self.process_epa(v, prompts)

        torch.manual_seed(seed) if seed else torch.seed()

        target_imgs = target_imgs.split("|") if target_imgs else []
        target_imgs = [image.strip() for image in target_imgs]

        if img_0:
            pil_image = Image.open(img_0).convert('RGB')
            pil_image = pil_image.resize((self.sideX, self.sideY), Image.LANCZOS)
            self.z, *_ = self.vqg_model.encode(
                TF.to_tensor(pil_image).to(device).unsqueeze(0) * 2 - 1
            )
        else:
            noise_0 = (
                noise_0
                if len(noise_0) > 0
                else torch.randint(
                    self.n_toks, [self.toksY * self.toksX], device=device
                )
            )
            one_hot = F.one_hot(noise_0, self.n_toks).float()
            self.z = one_hot @ self.vqg_model.quantize.embedding.weight
            self.z = self.z.view([-1, self.toksY, self.toksX, self.e_dim]).permute(
                0, 3, 1, 2
            )
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        self.opt = torch.optim.Adam([self.z], lr=lr)

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        self.pMs = []

        for prompt in prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.clip_model.encode_text(clip.tokenize(txt).to(device)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        for prompt in target_imgs:
            path, weight, stop = parse_prompt(prompt)
            img = resize_image(
                Image.open(path).convert('RGB'), (self.sideX, self.sideY)
            )
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
            embed = self.clip_model.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(device))

        for seed, weight in zip(noise_prompt_seeds, noise_prompt_weights):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.clip_model.visual.output_dim]).normal_(
                generator=gen
            )
            self.pMs.append(Prompt(embed, weight).to(device))

    def synth(self, z):
        z_q = vector_quantize(
            z.movedim(1, 3), self.vqg_model.quantize.embedding.weight
        ).movedim(3, 1)
        return clamp_with_grad(self.vqg_model.decode(z_q).add(1).div(2), 0, 1)

    @torch.no_grad()
    def checkin(self, i, losses):
        losses_str = ', '.join(f'{loss.item():g}' for loss in losses)
        tqdm.write(f'i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}')
        out = self.synth(self.z)
        TF.to_pil_image(out[0].cpu()).save(self.img_savedir)
        add_stegano_data(self.img_savedir)
        add_xmp_data(self.img_savedir)

    def ascend_txt(self, init_weight=0, aff_weight=1):
        global i
        out = self.synth(self.z)
        iii = self.clip_model.encode_image(
            self.normalize(self.make_cutouts(out))
        ).float()

        result = []

        # if init_weight:
        #     result.append(F.mse_loss(self.z, self.z_orig) * init_weight / 2)

        for prompt in self.pMs:
            result.append(prompt(iii))

        if len(self.target_affect) >= 1:
            z = self.data_handler.scaler_Z.scale(iii)
            result.append(aff_weight * F.mse_loss(self.mlp(z), self.target_affect))

        return result

    def train(self, iter=0, aff_weight=1):
        self.opt.zero_grad()
        lossAll = self.ascend_txt(aff_weight)
        if iter % 50 == 0:
            self.checkin(iter, lossAll)
        loss = sum(lossAll)
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))
