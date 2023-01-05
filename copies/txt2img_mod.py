import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class StableDiffuser:
    def __init__(
            self,
            config = "stable_diffusion/configs/stable-diffusion/v1-inference.yaml",
            ckpt = "stable_diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
            outdir = "stable_diffusion/outputs/txt2img-samples",
        ):
        config = OmegaConf.load(f"{config}")
        model = load_model_from_config(config, f"{ckpt}")
        self.model = model.to(device)

        self.sampler = PLMSSampler(self.model)

        os.makedirs(outdir, exist_ok=True)
        self.outpath = outdir

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        self.wm_encoder = WatermarkEncoder()
        self.wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    @torch.no_grad()
    def initialize(self, prompt, start_code=None, precision="autocast", C=4, W=512, H=512, f=8, n_samples=3):
        assert prompt is not None
        self.C = C
        self.f = f
        self.W = W
        self.H = H
        if start_code is None:
            self.start_code = torch.randn([self.n_samples, self.C, self.H // self.f, self.W // self.f], device=device)
            self.n_samples = n_samples
            self.batch_size = n_samples
        #     start_code = torch.randn([1, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
        #     start_code = torch.cat([start_code for k in range(opt.n_samples)], dim=0)
        else:
            self.start_code = start_code.to(device)
            self.n_samples = start_code.shape[0]
            self.batch_size = start_code.shape[0]
        
        self.data = [self.batch_size * [prompt]]
        self.sample_path = os.path.join(self.outpath, prompt.replace(' ','_'))
        os.makedirs(self.sample_path, exist_ok=True)
        self.base_count = len(os.listdir(self.sample_path))
        self.precision_scope = autocast if precision=="autocast" else nullcontext

    @torch.no_grad()
    def run_diffusion(self, scale=7.5, ddim_steps=50, ddim_eta=0., save=True, suffix=''):
        with self.precision_scope('cuda'):
            with self.model.ema_scope():
                tic = time.time()
                all_samples = list()
                for prompts in tqdm(self.data, desc="data"):
                    uc = None
                    if scale != 1.0:
                        uc = self.model.get_learned_conditioning(self.batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = self.model.get_learned_conditioning(prompts)
                    shape = [self.C, self.H // self.f, self.W // self.f]
                    samples_ddim, _ = self.sampler.sample(S=ddim_steps,
                                                        conditioning=c,
                                                        batch_size=self.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=uc,
                                                        eta=ddim_eta,
                                                        x_T=self.start_code)

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    self.img_batch = x_samples_ddim

                    if save:
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_checked_image = x_samples_ddim
                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, self.wm_encoder)
                            img.save(os.path.join(self.sample_path, f"{self.base_count:05}.png"))
                            self.base_count += 1

                toc = time.time()

