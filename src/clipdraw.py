from src.cicada_utils import treebranch_initialization, get_augment_trans
from src.cicada_drawing import Drawing
import clip
import torch
import pydiffvg
import pickle
from src.mlp import MLP
import torch.nn.functional as F


pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(torch.cuda.is_available())
pydiffvg.set_device(torch.device('cuda:0') if torch.cuda.is_available() else 'cpu')


class CLIPAffDraw:
    def __init__(self, canvas_w=224, canvas_h=224, normalize_clip=True, num_augs=4):
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model, preprocess = clip.load('ViT-B/32', self.device, jit=False)
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.augment_trans = get_augment_trans(canvas_w, normalize_clip)
        self.drawing = Drawing(canvas_w, canvas_h)
        self.drawing_area = {'x0': 0, 'x1': 1, 'y0': 0, 'y1': 1}
        self.num_augs = num_augs
        self.mlp = MLP([64, 32]).to('cuda:0')
        self.mlp.load_state_dict(torch.load('data/model_mixed.pt'))
        with open('data/data_handler_mixed.pkl', 'rb') as f:
            self.data_handler = pickle.load(f)

    @torch.no_grad()
    def process_text(
        self, prompt, neg_prompt_1=None, neg_prompt_2=None, v=[0.5, 0.5, 0.5]
    ):
        self.target_affect = torch.matmul(
            torch.ones((self.num_augs, 1), device=self.device),
            torch.tensor([v], device=self.device, requires_grad=False),
        )
        self.use_neg_prompts = not (neg_prompt_1 is None)
        tokens = clip.tokenize(prompt).to(self.device)
        self.text_features = self.model.encode_text(tokens)
        if self.use_neg_prompts:
            neg_tokens_1 = clip.tokenize(neg_prompt_1).to(self.device)
            neg_tokens_2 = clip.tokenize(neg_prompt_2).to(self.device)
            self.text_features_neg1 = self.model.encode_text(neg_tokens_1)
            self.text_features_neg2 = self.model.encode_text(neg_tokens_2)

    def add_random_shapes(self, num_rnd_traces):
        '''
        This will NOT discard existing shapes
        ---
        input:
            num_rnd_traces: Int;
        '''
        shapes, shape_groups = treebranch_initialization(
            self.drawing, num_rnd_traces, self.drawing_area,
        )
        self.drawing.add_shapes(shapes, shape_groups, fixed=False)

    def initialize_variables(self, max_width=40):
        self.max_width = max_width
        self.points_vars = []
        self.stroke_width_vars = []
        self.color_vars = []
        for trace in self.drawing.traces:
            trace.shape.points.requires_grad = True
            self.points_vars.append(trace.shape.points)
            trace.shape.stroke_width.requires_grad = True
            self.stroke_width_vars.append(trace.shape.stroke_width)
            trace.shape_group.stroke_color.requires_grad = True
            self.color_vars.append(trace.shape_group.stroke_color)

        self.render = pydiffvg.RenderFunction.apply

    def initialize_optimizer(self):
        self.points_optim = torch.optim.Adam(self.points_vars, lr=0.2)
        self.width_optim = torch.optim.Adam(self.stroke_width_vars, lr=0.2)
        self.color_optim = torch.optim.Adam(self.color_vars, lr=0.02)

    def build_img(self, t, shapes=None, shape_groups=None):
        if not shapes:
            shapes = [trace.shape for trace in self.drawing.traces]
            shape_groups = [trace.shape_group for trace in self.drawing.traces]
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.canvas_w, self.canvas_h, shapes, shape_groups
        )
        img = self.render(self.canvas_w, self.canvas_h, 2, 2, t, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(
            img.shape[0], img.shape[1], 3, device=pydiffvg.get_device()
        ) * (1 - img[:, :, 3:4])
        img = img[:, :, :3].unsqueeze(0).permute(0, 3, 1, 2)  # NHWC -> NCHW
        return img

    def run_epoch(self, t):
        self.points_optim.zero_grad()
        self.width_optim.zero_grad()
        self.color_optim.zero_grad()

        img = self.build_img(t)

        self.img = img.cpu().permute(0, 2, 3, 1).squeeze(0)

        loss_sem = 0
        loss_aff = 0

        img_augs = []
        for n in range(self.num_augs):
            img_augs.append(self.augment_trans(img))
        im_batch = torch.cat(img_augs)
        img_features = self.model.encode_image(im_batch)
        for n in range(self.num_augs):
            loss_sem -= torch.cosine_similarity(
                self.text_features, img_features[n : n + 1], dim=1
            )
            if self.use_neg_prompts:
                loss_sem += (
                    torch.cosine_similarity(
                        self.text_features_neg1, img_features[n : n + 1], dim=1
                    )
                    * 0.3
                )
                loss_sem += (
                    torch.cosine_similarity(
                        self.text_features_neg2, img_features[n : n + 1], dim=1
                    )
                    * 0.3
                )
        loss_aff = F.mse_loss(
            self.mlp(img_features.to(torch.float32)), self.target_affect
        )

        loss = loss_sem + 5 * loss_aff

        # Backpropagate the gradients.
        loss.backward()

        # Take a gradient descent step.
        self.points_optim.step()
        self.width_optim.step()
        self.color_optim.step()
        for trace in self.drawing.traces:
            trace.shape.stroke_width.data.clamp_(1.0, self.max_width)
            trace.shape_group.stroke_color.data.clamp_(0.0, 1.0)

        self.losses = {
            'global': loss.to('cpu').item(),
            'semantic': loss_sem.to('cpu').item(),
            'affective': loss_aff.to('cpu').item(),
        }
